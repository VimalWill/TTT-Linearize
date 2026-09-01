"""TTT loss per token during the forward pass.

Under `ttt_inner_loss='dot'` (LaCT) the inner loop ascends <f(W;k), v> -- a
dot-product objective, not squared reconstruction error (there is no residual
term in the operator), and the magnitude is pinned by the Eq. 8 renorm. So the
loss is reported as 1 - cos(f(W;k_t), v_t): lower is better, 0 is perfect
alignment.

Under `ttt_inner_loss='l2'` (Atlas) it descends 1/2||f(W;k) - v||^2 with a
retention gate in place of the renorm, so the magnitude is meaningful and the
objective itself is reported, scale-normalised as
||f(W;k_t) - v_t||^2/||v_t||^2. It also prints ||f(k)||/||v||, since an l2 run
that cannot close the magnitude gap is the failure mode this pairing exists to
fix. The two losses are not comparable across settings; compare within one.

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
from LinearTTT.model.LinearizeLlama.ttt_ops import zeropower_via_newtonschulz5
from Training.train import build_model_config


def silu_backprop(dy, x):
    sigma = torch.sigmoid(x)
    return dy * sigma * (1 + x * (1 - sigma))


def token_loss(w0, w1, w2, ki, vi, inner_loss):
    """Per-token inner-loop loss, and ||f(k)||/||v||. -> ([l], [l])"""
    kT = ki.transpose(1, 2)
    pred = torch.bmm(w1, F.silu(torch.bmm(w0, kT)) * torch.bmm(w2, kT))
    ratio = pred.norm(dim=1) / (vi.norm(dim=1) + 1e-8)      # [b*h, l]
    # mean AND max over heads: the mean hides heads whose readout is already
    # far above ||v||, which are the ones that blow up first.
    mag = torch.stack([ratio.mean(dim=0), ratio.amax(dim=0)])
    if inner_loss == 'l2':
        loss = (((pred - vi) ** 2).sum(dim=1)
                / ((vi ** 2).sum(dim=1) + 1e-8)).mean(dim=0)
    else:
        loss = (1.0 - F.cosine_similarity(pred, vi, dim=1)).mean(dim=0)
    return loss, mag


def instrumented(records, mags, inner_loss='dot'):
    """Replay of block_causal_lact_swiglu recording per-token loss.

    Mirrors third_party/LaCT/lact_llm/lact_model/ttt_operation.py; the only
    addition is the measurement.
    """
    def op(w0, w1, w2, q, k, v, lr0, lr1, lr2,
           chunk_size=2048, use_muon=False, momentum=None, retention=None):
        w0, w1, w2 = w0.float(), w1.float(), w2.float()
        n0 = w0.norm(dim=2, keepdim=True)
        n1 = w1.norm(dim=2, keepdim=True)
        n2 = w2.norm(dim=2, keepdim=True)
        if momentum is not None:
            m0 = torch.zeros_like(w0); m1 = torch.zeros_like(w1); m2 = torch.zeros_like(w2)

        vT = v.transpose(1, 2).float()
        qT = q.transpose(1, 2).float()
        out = torch.zeros_like(vT)
        per_token, per_mag, e_index, seq_len = [], [], 0, k.shape[1]

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
            l_, m_ = token_loss(w0, w1, w2, ki, vi, inner_loss)
            per_token.append(l_.cpu()); per_mag.append(m_.cpu())

            kT = ki.transpose(1, 2)
            gate_before_act = torch.bmm(w0, kT)
            hidden_before_mul = torch.bmm(w2, kT)
            hidden = F.silu(gate_before_act) * hidden_before_mul

            # the l2 bias seeds the backward pass with the residual, not v
            err = vi - torch.bmm(w1, hidden) if inner_loss == 'l2' else vi

            dhidden = torch.bmm(w1.transpose(1, 2), err)
            dhidden_before_mul = dhidden * F.silu(gate_before_act)
            dgate_before_act = silu_backprop(dhidden * hidden_before_mul, gate_before_act)

            dw1 = torch.bmm(err, hidden.transpose(1, 2) * lr1i)
            dw0 = torch.bmm(dgate_before_act, ki * lr0i)
            dw2 = torch.bmm(dhidden_before_mul, ki * lr2i)

            if momentum is not None:
                mi = momentum[:, s_index:e_index, :].float().mean(dim=1, keepdim=True)
                dw0 = dw0 + m0 * mi; dw1 = dw1 + m1 * mi; dw2 = dw2 + m2 * mi
                m0, m1, m2 = dw0, dw1, dw2

            if use_muon:
                # NS discards its input's magnitude, so the per-token lr folded
                # into dw has to be reapplied as Atlas Eq. 32's eta_t. Mirrors
                # ttt_l2.py; without it the replay reports dynamics the real
                # operator does not have.
                dw0 = zeropower_via_newtonschulz5(dw0).float() * lr0i.mean(dim=1, keepdim=True)
                dw1 = zeropower_via_newtonschulz5(dw1).float() * lr1i.mean(dim=1, keepdim=True)
                dw2 = zeropower_via_newtonschulz5(dw2).float() * lr2i.mean(dim=1, keepdim=True)

            if retention is not None:
                # Atlas Eq. 32, in place of LaCT Eq. 8
                a_i = retention[:, s_index:e_index, :].float().mean(dim=1, keepdim=True)
                w0 = w0 * a_i + dw0; w1 = w1 * a_i + dw1; w2 = w2 * a_i + dw2
            else:
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
        l_, m_ = token_loss(w0, w1, w2, ki, vi, inner_loss)
        per_token.append(l_.cpu()); per_mag.append(m_.cpu())

        records.append(torch.cat(per_token))
        mags.append(torch.cat(per_mag, dim=1))
        return out.transpose(1, 2).to(v.dtype)
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--out', default='ttt_loss_per_token')
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

    records, mags = [], []
    inner_loss = getattr(model_config, 'ttt_inner_loss', 'dot')
    op_name = ('block_causal_lact_swiglu_l2' if inner_loss == 'l2'
               else 'block_causal_lact_swiglu')
    print(f'inner objective: {inner_loss} '
          f'({"||f(k)-v||^2/||v||^2" if inner_loss == "l2" else "1 - cos(f(k), v)"})')
    original = getattr(lz, op_name)
    setattr(lz, op_name, instrumented(records, mags, inner_loss))
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        setattr(lz, op_name, original)

    loss = torch.stack(records)                       # [layers, tokens]
    n_iter = loss.shape[1] // cs
    # one point per inner-loop iteration: mean over layers and over the tokens
    # in that chunk. Iteration i is read out with weights fit on chunks < i.
    per_iter = loss[:, :n_iter * cs].view(loss.shape[0], n_iter, cs).mean(dim=(0, 2))

    print(f'\n{loss.shape[0]} layers, chunk_size={cs}, {n_iter} inner-loop iterations\n')
    mag = torch.stack(mags)                    # [layers, 2, tokens]
    mag_iter = mag[:, :, :n_iter * cs].view(
        mag.shape[0], 2, n_iter, cs).mean(dim=3)
    mean_iter = mag_iter[:, 0].mean(dim=0)     # mean over layers
    max_iter = mag_iter[:, 1].amax(dim=0)      # worst head, worst layer
    print(f'{"iteration":>10} {"loss":>10} {"|f|/|v| mean":>14} {"|f|/|v| max":>13}')
    for i, (l, mu, mx) in enumerate(zip(per_iter.tolist(),
                                        mean_iter.tolist(), max_iter.tolist())):
        print(f'{i:>10} {l:>10.4f} {mu:>14.4f} {mx:>13.4f}')
    print(f'\n  {per_iter[0]:.4f} -> {per_iter[-1]:.4f}   delta {per_iter[-1] - per_iter[0]:+.4f}')

    tag = f'_{args.control}' if args.control else ''
    with open(f'{args.out}{tag}.csv', 'w') as f:
        f.write('iteration,loss\n')
        for i, l in enumerate(per_iter.tolist()):
            f.write(f'{i},{l:.6f}\n')
    print(f'\nwrote {args.out}{tag}.csv')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(n_iter), per_iter, color='#1f4e79', lw=1.8, marker='o', ms=4)
    ax.set_xlabel('Inner-Loop Iteration')
    ax.set_ylabel('Inner-Loop Loss')
    if args.control:
        ax.set_title(f'control: {args.control}')
    ax.grid(alpha=0.25)
    fig.savefig(f'{args.out}{tag}.png', dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}{tag}.png')


if __name__ == '__main__':
    main()
