"""Does the fast weight actually learn the document as it reads it?

The LaCT inner loop ascends <f(W;k), v> -- a dot-product objective, NOT squared
reconstruction error (there is no residual term anywhere in the operator). So the
right measure of inner-loop learning is *alignment* between f(W;k_i) and v_i.

Because the operator is apply-then-update, chunk i is read out with weights fit
only on chunks 0..i-1. Measuring alignment on chunk i under W_{i-1} is therefore
held-out generalization *within a single sequence* -- exactly the claim "test-time
training" makes.

Reports, per layer and per chunk:
  loss    1 - cos(f(W_{i-1}; k_i), v_i)   falling => the state is learning
  lr      mean per-token inner learning rate the model chose for this chunk
  |dw|/|w|  relative size of the fast-weight update

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


def instrumented(records):
    """Replay of block_causal_lact_swiglu that records per-chunk statistics.

    Mirrors third_party/LaCT/lact_llm/lact_model/ttt_operation.py exactly; the
    only additions are the measurements.
    """
    def op(w0, w1, w2, q, k, v, lr0, lr1, lr2,
           chunk_size=2048, use_muon=False, momentum=None):
        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        w0_norm = w0.norm(dim=2, keepdim=True)
        w1_norm = w1.norm(dim=2, keepdim=True)
        w2_norm = w2.norm(dim=2, keepdim=True)
        if momentum is not None:
            m0 = torch.zeros_like(w0); m1 = torch.zeros_like(w1); m2 = torch.zeros_like(w2)

        qT, vT = q.transpose(1, 2).float(), v.transpose(1, 2).float()
        out = torch.zeros_like(vT)
        rows, e_index, seq_len = [], 0, k.shape[1]

        for n, i in enumerate(range(0, seq_len - chunk_size, chunk_size)):
            s_index, e_index = i, i + chunk_size
            ki = k[:, s_index:e_index, :].float()
            vi = vT[:, :, s_index:e_index]
            qi = qT[:, :, s_index:e_index]
            lr1i = lr1[:, s_index:e_index, :].float()
            lr2i = lr2[:, s_index:e_index, :].float()
            lr0i = lr0[:, s_index:e_index, :].float()

            out[:, :, s_index:e_index] = torch.bmm(
                w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))

            kT = ki.transpose(1, 2)
            gate_before_act = torch.bmm(w0, kT)
            hidden_before_mul = torch.bmm(w2, kT)
            hidden = F.silu(gate_before_act) * hidden_before_mul

            # ---- measurement: W_{i-1} was fit on chunks < i, so this is held out
            pred = torch.bmm(w1, hidden)                       # [B, dv, l]
            # the inner rule ascends <f(W;k), v>, so report it as a decreasing loss
            loss = (1.0 - F.cosine_similarity(pred, vi, dim=1).mean()).item()

            dhidden = torch.bmm(w1.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, hidden.transpose(1, 2) * lr1i)
            dw0 = torch.bmm(dgate_before_act, ki * lr0i)
            dw2 = torch.bmm(dhidden_before_mul, ki * lr2i)

            if momentum is not None:
                mi = momentum[:, s_index:e_index, :].float().mean(dim=1, keepdim=True)
                dw0 = dw0 + m0 * mi; dw1 = dw1 + m1 * mi; dw2 = dw2 + m2 * mi
                m0, m1, m2 = dw0, dw1, dw2

            rows.append((n, loss,
                         torch.cat([lr0i, lr1i, lr2i]).mean().item(),
                         (dw1.norm() / w1.norm()).item()))

            w0 = w0 + dw0; w1 = w1 + dw1; w2 = w2 + dw2
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        qi = qT[:, :, e_index:seq_len]
        out[:, :, e_index:seq_len] = torch.bmm(
            w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi))
        records.append(rows)
        return out.transpose(1, 2).to(v.dtype)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 31])
    ap.add_argument('--out', default='ttt_learning_curve',
                    help='writes <out>.csv and <out>.png')
    ap.add_argument('--control', choices=['shuffle', 'mixed', 'random'], default=None,
                    help="shuffle: reorder chunks of one document (tests order only -- "
                         "weak, since document identity survives). mixed: each chunk "
                         "from a DIFFERENT document (tests document identity). random: "
                         "uniform random token ids (strongest null). If alignment still "
                         "rises under mixed/random, the curve is initialisation warm-up, "
                         "not learning.")
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model_config = build_model_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16).eval()

    # Real held-out prose. A repeated sentence would be learned trivially and
    # would produce a beautiful curve that means nothing.
    # Stream only the validation split -- load_data() would pull all 200 training
    # books for one sequence, which is how we got rate-limited (429).
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.base)
    cs = model_config.lact_chunk_size
    n_chunks = args.seq_len // cs
    g = torch.Generator().manual_seed(0)

    if args.control == 'random':
        ids = torch.randint(0, model_config.vocab_size, (1, args.seq_len),
                            generator=g).cuda()
        print('CONTROL random: uniform token ids -- nothing to learn')
    elif args.control == 'mixed':
        # One chunk from each of n_chunks DIFFERENT documents. Every chunk is real
        # prose, but no two share a document, so there is no consistent thing for
        # the fast weight to accumulate.
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

    print(f'\nchunk_size={model_config.lact_chunk_size} seq_len={args.seq_len} '
          f'-> {len(records[0])} inner updates per sequence')
    print('loss = 1 - cos(f(W_fit_on_chunks<i ; k_i), v_i) -- held out within the sequence\n')

    for li in args.layers:
        if li >= len(records):
            continue
        print(f'--- layer {li} ---')
        print(f'{"chunk":>6} {"loss":>9} {"lr":>10} {"|dw1|/|w1|":>11}')
        for n, loss, lr, dw in records[li]:
            print(f'{n:>6} {loss:>9.4f} {lr:>10.5f} {dw:>11.4f}')
        first, last = records[li][0][1], records[li][-1][1]
        print(f'  first {first:+.4f} -> last {last:+.4f}   delta {last - first:+.4f}\n')

    tag = f'_{args.control}' if args.control else ''
    csv_path = f'{args.out}{tag}.csv'
    with open(csv_path, 'w') as f:
        f.write('layer,chunk,loss,lr,dw1_rel\n')
        for li, rows in enumerate(records):
            for n, loss, lr, dw in rows:
                f.write(f'{li},{n},{loss:.6f},{lr:.6f},{dw:.6f}\n')
    print(f'wrote {csv_path}  ({len(records)} layers x {len(records[0])} chunks)')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available -- CSV written, skipping plot')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.get_cmap('viridis')
    for li, rows in enumerate(records):
        c = cmap(li / max(len(records) - 1, 1))
        xs = [r[0] for r in rows]
        for ax, j in zip(axes, (1, 2, 3)):
            ax.plot(xs, [r[j] for r in rows], color=c, lw=1, alpha=0.75)
    for ax, title, ylab in zip(
            axes,
            ('TTT loss (held out within sequence)', 'inner learning rate', 'update magnitude'),
            (r'$1-\cos(f(W_{<i};k_i),\,v_i)$', 'mean lr', r'$\|dw_1\|/\|w_1\|$')):
        ax.set_title(title); ax.set_xlabel('inner-loop chunk'); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, len(records) - 1))
    fig.colorbar(sm, ax=axes, label='layer', fraction=0.02)
    fig.suptitle('TTT inner-loop learning'
                 + (f'  (CONTROL: {args.control})' if args.control else ''))
    png = f'{args.out}{tag}.png'
    fig.savefig(png, dpi=140, bbox_inches='tight')
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
