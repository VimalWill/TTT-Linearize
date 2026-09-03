"""Zero-shot capacity reduction: does the measured allocation beat the controls
without any retraining?

Fusing normally changes d_h, so a fused model cannot load the trained
checkpoint. But zeroing hidden channels of the SwiGLU fast weights is
equivalent to a narrower d_h while keeping every shape identical, so the
allocation can be tested on the checkpoint you already have.

The zeroing is stable through the inner loop, which is what makes this a real
capacity reduction rather than a perturbation that heals: with w0[h,j,:] = 0 the
gate is silu(0) = 0, so hidden[j] = 0, so dw1[:,j] = err @ hidden[j]^T = 0 and
both dw0[j] and dw2[j] vanish through hidden_before_mul. The channel is dead for
the whole sequence.

Channels are ranked per head by
    ||w0[h,j,:]|| * ||w2[h,j,:]|| * ||w1[h,:,j]||
i.e. how much the channel is driven on the input side times how much it
contributes on the output side, and the top-k are kept. This is magnitude-based
structured pruning of the SwiGLU hidden dimension.

    python3 diag_capacity.py --cfg Configs/ttt_ar_l2.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_l2/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_l2/best_ckpt \
        --arms measured inverted uniform

CAVEAT: zero-shot pruning is a LOWER bound. Retraining recovers some of the
loss, and it may recover different amounts per arm, so a ranking here need not
survive stage 1 + stage 2. Read this as a fast screen, not as the result.
"""

import argparse
import copy
import math

import torch
import yaml
from omegaconf import OmegaConf

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from Training.dataloader import load_data
from diag_common import load_model
from diag_retrieval import ar_masks, score_ar, ttt_layers


def channel_rank(m):
    """-> [H, d_h] score per hidden channel, higher = more load-bearing."""
    w0, w1, w2 = m.w0.detach().float(), m.w1.detach().float(), m.w2.detach().float()
    inp = w0.norm(dim=2) * w2.norm(dim=2)      # [H, d_h] driven on the input side
    out = w1.norm(dim=1)                       # [H, d_h] contributed on the output
    return inp * out


@torch.no_grad()
def prune(m, keep_frac):
    """Zero all but the top keep_frac of hidden channels, per head."""
    d_h = m.w0.shape[1]
    k = max(1, int(round(d_h * keep_frac)))
    if k >= d_h:
        return d_h
    idx = channel_rank(m).argsort(dim=1, descending=True)[:, k:]   # [H, d_h-k]
    for h in range(m.w0.shape[0]):
        dead = idx[h]
        m.w0[h, dead, :] = 0
        m.w2[h, dead, :] = 0
        m.w1[h, :, dead] = 0
    return k


def arm_vector(path, n_layers):
    r = yaml.safe_load(open(path))['model'].get('ttt_inter_multi')
    if r is None:
        raise SystemExit(f'{path} has no per-layer ttt_inter_multi')
    if not isinstance(r, list) or len(r) != n_layers:
        raise SystemExit(f'{path}: expected {n_layers} entries, got {r if not isinstance(r,list) else len(r)}')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_ar_l2.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--arms', nargs='+', default=['measured', 'inverted', 'uniform'])
    ap.add_argument('--arm-cfg', default='Configs/ttt_at_{}.yml')
    ap.add_argument('--seqs', type=int, default=4)
    ap.add_argument('--seq-len', type=int, default=32768)
    ap.add_argument('--edges', type=int, nargs='+', default=[512, 2048, 8192])
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--out', default='capacity')
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    cfg.model.max_length = args.seq_len
    model_config = build_model_config(cfg)
    chunk, window = model_config.lact_chunk_size, model_config.window_size

    loader = load_data(cfg)['validation']
    seqs, masks, edges = [], [], list(args.edges) + [math.inf]
    for b in loader:
        for row in b['input_ids']:
            if len(seqs) >= args.seqs:
                break
            seqs.append(row.cpu())
            masks.append(ar_masks(row.cpu(), edges, window, chunk)[0])
        if len(seqs) >= args.seqs:
            break
    names = [n for n in masks[0] if n != 'other'] + ['other']
    print(f'{len(seqs)} x {seqs[0].shape[0]} tokens; window {window}, chunk {chunk}')

    model = load_model(args.ckpt, model_config, args.adapter)
    mods = ttt_layers(model)
    pristine = [(m.w0.detach().clone(), m.w1.detach().clone(), m.w2.detach().clone())
                for m in mods]

    base, cnt = score_ar(model, seqs, masks, args.batch)
    names = [n for n in names if cnt[n]]
    far = [n for n in names if n not in ('other', 'ar_same_chunk')][-1]
    print(f'\n{"arm":<10}{"sum r":>7}' + ''.join(f'{n:>15}' for n in names))
    print(f'{"full":<10}{32.0:>7.2f}' + ''.join(f'{base[n]:>15.4f}' for n in names))

    rows = {}
    for arm in args.arms:
        r = arm_vector(args.arm_cfg.format(arm), len(mods))
        with torch.no_grad():                      # restore, then prune
            for m, (a, b_, c) in zip(mods, pristine):
                m.w0.copy_(a); m.w1.copy_(b_); m.w2.copy_(c)
        ks = [prune(m, f) for m, f in zip(mods, r)]
        ce, _ = score_ar(model, seqs, masks, args.batch)
        rows[arm] = ce
        print(f'{arm:<10}{sum(r):>7.2f}' + ''.join(f'{ce[n]:>15.4f}' for n in names))
        print(f'{"":<10}{"delta":>7}' + ''.join(f'{ce[n]-base[n]:>+15.4f}' for n in names)
              + f'   d_h kept: {min(ks)}-{max(ks)}')

    with torch.no_grad():
        for m, (a, b_, c) in zip(mods, pristine):
            m.w0.copy_(a); m.w1.copy_(b_); m.w2.copy_(c)

    print(f'\nranking at {far} (lower delta = better allocation):')
    for arm, ce in sorted(rows.items(), key=lambda kv: kv[1][far]):
        print(f'  {arm:<10}{ce[far]-base[far]:+.4f}')
    print('\nzero-shot only -- retraining recovers some of this, possibly unevenly '
          'across arms.')
    with open(f'{args.out}.csv', 'w') as f:
        f.write('arm,' + ','.join(names) + '\n')
        f.write('full,' + ','.join(f'{base[n]:.5f}' for n in names) + '\n')
        for arm, ce in rows.items():
            f.write(f'{arm},' + ','.join(f'{ce[n]:.5f}' for n in names) + '\n')
    print(f'wrote {args.out}.csv')


if __name__ == '__main__':
    main()
