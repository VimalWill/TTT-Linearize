"""Build a striped TTT model in MAD and check it trains. Needs a GPU.

This is the step that cannot run on a machine without triton, which MAD's
RMSNorm imports unconditionally. It checks the three things the adapter is
responsible for: that a striped stack constructs, that sharing links the
right blocks and removes their state, and that gradient reaches the writer's
fast weights through a reader's loss.

    PYTHONPATH=.:third_party/mad-lab python LinearTTT/mad/smoke.py
"""
import argparse

import torch

from LinearTTT.mad.register import (
    link_shared_memories, register_ttt, total_state_dim, ttt_blocks,
)


def build(dim, vocab, length, heads, chunk, share):
    from mad.model import LanguageModel
    from mad.model.layers import Mlp

    registry = register_ttt()
    ttt, attention = registry['ttt']['module'], registry['mh-attention']['module']

    # 3:1 striped, matching hybrid3 and MAD's own optimal ratio of 25%
    # attention: three TTT layers then one attention layer, each followed by
    # a channel mixer.
    layers, configs = [], []
    for kind in (ttt, ttt, ttt, attention):
        layers += [kind, Mlp]
        base = dict(dim=dim, max_length=length)
        if kind is ttt:
            base.update(num_heads=heads, chunk_size=chunk)
        else:
            base.update(n_heads=heads, rotary_emb_dim=8)
        configs += [base, dict(dim=dim, max_length=length)]

    model = LanguageModel(vocab_size=vocab, layers=layers, layer_cfgs=configs,
                          dim=dim, max_length=length)
    private = total_state_dim(model)
    if share:
        link_shared_memories(model, [[0, 1, 2]])
    return model, private


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--vocab', type=int, default=64)
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=32)
    ap.add_argument('--steps', type=int, default=20)
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, private = build(a.dim, a.vocab, a.length, a.heads, a.chunk, share=True)
    model = model.to(device)
    blocks = ttt_blocks(model)
    shared = total_state_dim(model)
    print(f'TTT blocks {len(blocks)}  |  readers '
          f'{sum(b._share_src is not None for b in blocks)}')
    print(f'total fixed state: {private} private -> {shared} shared '
          f'({private / max(shared, 1):.1f}x reduction)')
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')

    # A few steps of a trivially learnable task: copy the previous token. The
    # point is that the loss moves and the writer's memory gets gradient, not
    # that the task is interesting.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first, last = None, None
    for step in range(a.steps):
        ids = torch.randint(0, a.vocab, (4, a.length), device=device)
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, a.vocab), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    writer_grad = blocks[0].w0.grad
    print(f'loss {first:.4f} -> {last:.4f} over {a.steps} steps')
    print(f"writer fast-weight gradient: "
          f"{'present' if writer_grad is not None and writer_grad.abs().max() > 0 else 'MISSING'}")


if __name__ == '__main__':
    main()
