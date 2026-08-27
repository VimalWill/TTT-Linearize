"""Inner-loop loss vs. number of inner-loop iterations.

LaCT takes exactly one gradient step per chunk. This sweeps that count: for each
N, the fast weight takes N steps on a chunk before the sequence advances.

Inner loss is 1 - cos(f(W_{<i}; k_i), v_i), held out: chunk i is scored with
weights fit only on earlier chunks.

    python3 diag_iters.py --ckpt $TTT_CKPT_DIR/ttt_at/best
"""

import argparse

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import LinearTTT  # noqa: F401
from LinearTTT.model.LinearizeLlama import LinearizeLlama as lz
from Training.train import build_model_config


def silu_backprop(dy, x):
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def multi_step(n_steps, losses):
    """block_causal_lact_swiglu with n_steps inner updates per chunk."""
    def op(w0, w1, w2, q, k, v, lr0, lr1, lr2,
           chunk_size=2048, use_muon=False, momentum=None):
        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        n0 = w0.norm(dim=2, keepdim=True)
        n1 = w1.norm(dim=2, keepdim=True)
        n2 = w2.norm(dim=2, keepdim=True)
        if momentum is not None:
            m0 = torch.zeros_like(w0); m1 = torch.zeros_like(w1); m2 = torch.zeros_like(w2)

        vT = v.transpose(1, 2).float()
        qT = q.transpose(1, 2).float()
        out = torch.zeros_like(vT)
        held_out, e_index, seq_len = [], 0, k.shape[1]

        for i in range(0, seq_len - chunk_size, chunk_size):
            s_index, e_index = i, i + chunk_size
            ki = k[:, s_index:e_index, :].float()
            vi = vT[:, :, s_index:e_index]
            qi = qT[:, :, s_index:e_index]
            lr1i = lr1[:, s_index:e_index, :].float()
            lr2i = lr2[:, s_index:e_index, :].float()
            lr0i = lr0[:, s_index:e_index, :].float()
            kT = ki.transpose(1, 2)

            out[:, :, s_index:e_index] = torch.bmm(
                w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))

            # held out: W here was fit only on chunks < i
            pred = torch.bmm(w1, F.silu(torch.bmm(w0, kT)) * torch.bmm(w2, kT))
            held_out.append(
                (1.0 - F.cosine_similarity(pred, vi, dim=1).mean()).item())

            for _ in range(n_steps):
                gate_before_act = torch.bmm(w0, kT)
                hidden_before_mul = torch.bmm(w2, kT)
                hidden = F.silu(gate_before_act) * hidden_before_mul

                dhidden = torch.bmm(w1.transpose(1, 2), vi)
                dhidden_before_mul = dhidden * F.silu(gate_before_act)
                dgate_before_act = silu_backprop(
                    dhidden * hidden_before_mul, gate_before_act)

                dw1 = torch.bmm(vi, hidden.transpose(1, 2) * lr1i)
                dw0 = torch.bmm(dgate_before_act, ki * lr0i)
                dw2 = torch.bmm(dhidden_before_mul, ki * lr2i)

                if momentum is not None:
                    mi = momentum[:, s_index:e_index, :].float().mean(dim=1, keepdim=True)
                    dw0 = dw0 + m0 * mi; dw1 = dw1 + m1 * mi; dw2 = dw2 + m2 * mi
                    m0, m1, m2 = dw0, dw1, dw2

                w0 = w0 + dw0; w1 = w1 + dw1; w2 = w2 + dw2
                w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * n0
                w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * n1
                w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * n2

        qi = qT[:, :, e_index:seq_len]
        out[:, :, e_index:seq_len] = torch.bmm(
            w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))
        losses.append(sum(held_out) / len(held_out))
        return out.transpose(1, 2).to(v.dtype)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--sequences', type=int, default=8,
                    help='held-out sequences to average over')
    ap.add_argument('--iters', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument('--out', default='ttt_inner_iters')
    ap.add_argument('--verify', action='store_true',
                    help='check that the N=1 replay reproduces the real operator: '
                         'runs one sequence through both and compares the logits')
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model_config = build_model_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(args.base)

    from datasets import load_dataset
    stream = load_dataset(config.data.path, split='validation', streaming=True)
    need = args.seq_len * args.sequences
    toks = []
    for doc in stream:
        toks += tok.encode(doc['text'], add_special_tokens=False)
        if len(toks) >= need:
            break
    seqs = [torch.tensor(toks[i * args.seq_len:(i + 1) * args.seq_len]).unsqueeze(0).cuda()
            for i in range(args.sequences)]
    print(f'{len(seqs)} held-out sequences x {args.seq_len} tokens\n')

    original = lz.block_causal_lact_swiglu

    if args.verify:
        # The replay is only trustworthy if N=1 reproduces the shipped operator.
        with torch.no_grad():
            ref = model(input_ids=seqs[0], use_cache=False).logits.float()
            lz.block_causal_lact_swiglu = multi_step(1, [])
            try:
                mine = model(input_ids=seqs[0], use_cache=False).logits.float()
            finally:
                lz.block_causal_lact_swiglu = original
        d = (ref - mine).abs()
        rel = (d.max() / ref.abs().max()).item()
        agree = (ref.argmax(-1) == mine.argmax(-1)).float().mean().item()
        print(f'VERIFY  max|diff| {d.max():.4e}  relative {rel:.4e}  '
              f'argmax agreement {agree:.4%}')
        print('        (nonzero diff is expected: replay is fp32, operator is bf16)\n')

    results = []
    print(f'{"iters":>6} {"inner loss":>12}')
    for n in args.iters:
        losses = []
        lz.block_causal_lact_swiglu = multi_step(n, losses)
        try:
            with torch.no_grad():
                for ids in seqs:
                    model(input_ids=ids, use_cache=False)
        finally:
            lz.block_causal_lact_swiglu = original
        inner = sum(losses) / len(losses)
        results.append((n, inner))
        print(f'{n:>6} {inner:>12.4f}')

    with open(f'{args.out}.csv', 'w') as f:
        f.write('iters,inner_loss\n')
        for n, inner in results:
            f.write(f'{n},{inner:.6f}\n')
    print(f'\nwrote {args.out}.csv')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ns = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(ns, [r[1] for r in results], color='#1f4e79', lw=2, marker='o', ms=6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel('Inner-Loop Iterations')
    ax.set_ylabel('Inner-Loop Loss')
    ax.grid(alpha=0.25)
    fig.savefig(f'{args.out}.png', dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
