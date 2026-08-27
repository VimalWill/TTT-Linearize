"""TTT loss per token during the forward pass.

The LaCT inner loop ascends <f(W;k), v> -- a dot-product objective, not squared
reconstruction error (there is no residual term in the operator). So the loss is
reported as 1 - cos(f(W;k_t), v_t): lower is better, 0 is perfect alignment.

The operator is apply-then-update, so token t is read out with fast weights fit
only on chunks strictly before its own. Every point is therefore held out -- the
curve is within-sequence generalization, not a training loss.

Within a chunk the fast weights are frozen, so the curve is a staircase: flat
across a chunk, stepping down at each boundary where an update lands.

    python3 diag_inner.py --ckpt $TTT_CKPT_DIR/ttt_at/best
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


def token_loss(w0, w1, w2, ki, vi):
    """1 - cos(f(W;k_t), v_t) for every token in the chunk. -> [l]"""
    kT = ki.transpose(1, 2)
    pred = torch.bmm(w1, F.silu(torch.bmm(w0, kT)) * torch.bmm(w2, kT))
    return (1.0 - F.cosine_similarity(pred, vi, dim=1)).mean(dim=0)


def instrumented(records):
    """Replay of block_causal_lact_swiglu recording per-token loss.

    Mirrors third_party/LaCT/lact_llm/lact_model/ttt_operation.py; the only
    addition is the measurement.
    """
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
        per_token, e_index, seq_len = [], 0, k.shape[1]

        for i in range(0, seq_len - chunk_size, chunk_size):
            s_index, e_index = i, i + chunk_size
            ki = k[:, s_index:e_index, :].float()
            vi = vT[:, :, s_index:e_index]
            qi = qT[:, :, s_index:e_index]
            lr1i = lr1[:, s_index:e_index, :].float()
            lr2i = lr2[:, s_index:e_index, :].float()
            lr0i = lr0[:, s_index:e_index, :].float()

            out[:, :, s_index:e_index] = torch.bmm(
                w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))

            # W here was fit on chunks < i, so every token below is held out
            per_token.append(token_loss(w0, w1, w2, ki, vi).cpu())

            kT = ki.transpose(1, 2)
            gate_before_act = torch.bmm(w0, kT)
            hidden_before_mul = torch.bmm(w2, kT)
            hidden = F.silu(gate_before_act) * hidden_before_mul

            dhidden = torch.bmm(w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act)
            dgate_before_act = silu_backprop(dhidden * hidden_before_mul, gate_before_act)

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

        # tail chunk: read out with the final weights, still never fit on
        ki = k[:, e_index:seq_len, :].float()
        vi = vT[:, :, e_index:seq_len]
        qi = qT[:, :, e_index:seq_len]
        out[:, :, e_index:seq_len] = torch.bmm(
            w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))
        per_token.append(token_loss(w0, w1, w2, ki, vi).cpu())

        records.append(torch.cat(per_token))
        return out.transpose(1, 2).to(v.dtype)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--out', default='ttt_loss_per_token')
    ap.add_argument('--smooth', type=int, default=1,
                    help='moving-average window in tokens (1 = raw)')
    ap.add_argument('--control', choices=['shuffle', 'mixed', 'random'], default=None,
                    help="mixed: every chunk from a DIFFERENT document. random: uniform "
                         "token ids. shuffle: reorder chunks of one document (weak -- "
                         "document identity survives). If the loss still falls under "
                         "mixed/random, the curve is warm-up, not learning.")
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
    cs = model_config.lact_chunk_size
    n_chunks = args.seq_len // cs
    g = torch.Generator().manual_seed(0)

    if args.control == 'random':
        ids = torch.randint(0, model_config.vocab_size, (1, args.seq_len),
                            generator=g).cuda()
        print('CONTROL random: uniform token ids -- nothing to learn')
    elif args.control == 'mixed':
        stream = load_dataset(config.data.path, split='validation', streaming=True)
        pieces = []
        for doc in stream:
            t = tok.encode(doc['text'], add_special_tokens=False)
            if len(t) >= cs:
                pieces.append(t[:cs])
            if len(pieces) == n_chunks:
                break
        ids = torch.tensor([x for p in pieces for x in p]).unsqueeze(0).cuda()
        print(f'CONTROL mixed: {len(pieces)} chunks from {len(pieces)} different documents')
    else:
        stream = load_dataset(config.data.path, split='validation', streaming=True)
        toks = []
        for doc in stream:
            toks += tok.encode(doc['text'], add_special_tokens=False)
            if len(toks) >= args.seq_len:
                break
        ids = torch.tensor(toks[:args.seq_len]).unsqueeze(0).cuda()
        if args.control == 'shuffle':
            perm = torch.randperm(n_chunks, generator=g)
            ids = torch.cat([ids[:, p * cs:(p + 1) * cs] for p in perm], dim=1)
            print('CONTROL shuffle: chunk order destroyed, document identity kept')
        else:
            print(f'held-out PG19 validation text: {ids.shape[1]} tokens')

    records = []
    original = lz.block_causal_lact_swiglu
    lz.block_causal_lact_swiglu = instrumented(records)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        lz.block_causal_lact_swiglu = original

    loss = torch.stack(records)                       # [layers, tokens]
    print(f'\n{loss.shape[0]} layers x {loss.shape[1]} tokens, '
          f'chunk_size={cs} ({n_chunks - 1} inner updates)\n')
    print(f'{"layer":>5} {"first chunk":>12} {"last chunk":>11} {"delta":>9}')
    for li in range(loss.shape[0]):
        a, b = loss[li, :cs].mean().item(), loss[li, -cs:].mean().item()
        print(f'{li:>5} {a:>12.4f} {b:>11.4f} {b - a:>+9.4f}')

    tag = f'_{args.control}' if args.control else ''
    with open(f'{args.out}{tag}.csv', 'w') as f:
        f.write('layer,token,loss\n')
        for li in range(loss.shape[0]):
            for t in range(loss.shape[1]):
                f.write(f'{li},{t},{loss[li, t]:.6f}\n')
    print(f'\nwrote {args.out}{tag}.csv')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    y = loss
    if args.smooth > 1:
        k = torch.ones(1, 1, args.smooth) / args.smooth
        y = F.conv1d(loss.unsqueeze(1), k, padding=args.smooth // 2).squeeze(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap('viridis')
    xs = range(y.shape[1])
    for li in range(y.shape[0]):
        ax.plot(xs, y[li], color=cmap(li / max(y.shape[0] - 1, 1)), lw=0.8, alpha=0.8)
    for b in range(cs, loss.shape[1], cs):
        ax.axvline(b, color='0.85', lw=0.5, zorder=0)
    ax.set_xlabel('token position in sequence')
    ax.set_ylabel(r'$1-\cos(f(W;k_t),\,v_t)$')
    ax.set_title('TTT loss per token during the forward pass'
                 + (f'  (CONTROL: {args.control})' if args.control else ''))
    ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, y.shape[0] - 1))
    fig.colorbar(sm, ax=ax, label='layer')
    fig.savefig(f'{args.out}{tag}.png', dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}{tag}.png')


if __name__ == '__main__':
    main()
