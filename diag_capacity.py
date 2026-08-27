"""Memory capacity of the TTT fast weight, Atlas-style.

diag_inner.py measures the *online* curve: 15 updates as the model reads once.
This instead fixes a set of (k, v) pairs from a real forward pass and runs the
LaCT update rule over them for many iterations, so you see how well the fast
weight can fit a given number of items when optimised to convergence.

Sweeping --items shows where capacity saturates: a fixed-size SwiGLU fast weight
should fit 512 pairs easily and 8192 poorly, and the gap is the memory limit that
bounds long-context performance.

The inner rule ascends <f(W;k), v>, so loss is plotted as 1 - cos(f(W;k), v):
lower is better, 0 is perfect alignment.

    python3 diag_capacity.py --ckpt $TTT_CKPT_DIR/ttt_at/best --layer 16
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


def capture(store, layer_idx, original):
    """Grab the real (w, q, k, v, lr) handed to the operator for one layer.

    Every layer still runs the real operator -- returning zeros would disable the
    TTT branch in layers below the target and corrupt the hidden states the target
    layer sees.
    """
    calls = {'n': 0}

    def op(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs):
        if calls['n'] == layer_idx:
            store.update(dict(w0=w0.detach().float(), w1=w1.detach().float(),
                              w2=w2.detach().float(), k=k.detach().float(),
                              v=v.detach().float(),
                              lr0=lr0.detach().float(), lr1=lr1.detach().float(),
                              lr2=lr2.detach().float()))
        calls['n'] += 1
        return original(w0, w1, w2, q, k, v, lr0, lr1, lr2, **kwargs)
    return op


def fit(store, n_items, iters, lr_scale):
    """Run the LaCT update rule over the same n_items pairs for `iters` passes."""
    w0, w1, w2 = store['w0'].clone(), store['w1'].clone(), store['w2'].clone()
    n0 = w0.norm(dim=2, keepdim=True)
    n1 = w1.norm(dim=2, keepdim=True)
    n2 = w2.norm(dim=2, keepdim=True)

    ki = store['k'][:, :n_items, :]                    # [B, n, dk]
    vi = store['v'][:, :n_items, :].transpose(1, 2)    # [B, dv, n]
    lr0 = store['lr0'][:, :n_items, :] * lr_scale
    lr1 = store['lr1'][:, :n_items, :] * lr_scale
    lr2 = store['lr2'][:, :n_items, :] * lr_scale
    kT = ki.transpose(1, 2)

    curve = []
    for _ in range(iters):
        gate_before_act = torch.bmm(w0, kT)
        hidden_before_mul = torch.bmm(w2, kT)
        hidden = F.silu(gate_before_act) * hidden_before_mul

        pred = torch.bmm(w1, hidden)
        curve.append((1.0 - F.cosine_similarity(pred, vi, dim=1).mean()).item())

        dhidden = torch.bmm(w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate_before_act = silu_backprop(dhidden * hidden_before_mul, gate_before_act)

        w1 = w1 + torch.bmm(vi, hidden.transpose(1, 2) * lr1)
        w0 = w0 + torch.bmm(dgate_before_act, ki * lr0)
        w2 = w2 + torch.bmm(dhidden_before_mul, ki * lr2)

        # the operator's row-norm projection: capacity is bounded by construction
        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * n0
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * n1
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * n2
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--items', type=int, nargs='+', default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--lr-scale', type=float, default=1.0,
                    help='multiply the model-chosen inner lr (its own lr is tuned for '
                         'a single pass, so convergence may need more)')
    ap.add_argument('--out', default='ttt_capacity')
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
    toks = []
    for doc in stream:
        toks += tok.encode(doc['text'], add_special_tokens=False)
        if len(toks) >= args.seq_len:
            break
    ids = torch.tensor(toks[:args.seq_len]).unsqueeze(0).cuda()

    store, original = {}, lz.block_causal_lact_swiglu
    lz.block_causal_lact_swiglu = capture(store, args.layer, original)
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    finally:
        lz.block_causal_lact_swiglu = original
    del model
    torch.cuda.empty_cache()

    d_in = store['k'].shape[-1]
    d_h = store['w0'].shape[1]
    n_heads = store['k'].shape[0]
    params = n_heads * (2 * d_h * d_in + d_h * d_in)
    print(f'\nlayer {args.layer}: {n_heads} heads, d_in={d_in}, d_h={d_h}')
    print(f'fast weight holds {params:,} scalars total '
          f'({params // n_heads:,} per head)\n')

    curves = {}
    with torch.no_grad():
        for n in args.items:
            if n > store['k'].shape[1]:
                continue
            c = fit(store, n, args.iters, args.lr_scale)
            curves[n] = c
            print(f'{n:>6} items: loss {c[0]:.4f} -> {c[-1]:.4f}   '
                  f'(min {min(c):.4f} at iter {c.index(min(c))})')

    with open(f'{args.out}.csv', 'w') as f:
        f.write('items,iter,loss\n')
        for n, c in curves.items():
            for i, l in enumerate(c):
                f.write(f'{n},{i},{l:.6f}\n')
    print(f'\nwrote {args.out}.csv')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, c in curves.items():
        ax.plot(c, lw=1.2, label=f'{n} items')
    ax.set_xlabel('iteration'); ax.set_ylabel(r'$1-\cos(f(W;k),\,v)$')
    ax.set_title(f'TTT fast-weight memory capacity (layer {args.layer})')
    ax.legend(title='pairs to memorise'); ax.grid(alpha=0.25)
    fig.savefig(f'{args.out}.png', dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}.png')


if __name__ == '__main__':
    main()
