"""Figures for the learned reader alignment maps.

Reads ttt_params.pt directly rather than the summary JSON, because the two
claims worth plotting are distributional and mean/min/max cannot show them:
that the spectrum is near-isometric with thin tails, and that the orthogonal
and symmetric parts of the polar decomposition are comparable.

    python plot_reader_maps.py \
      --ckpt NAME=/path/to/stage2/best_ckpt [--ckpt NAME2=...] \
      --out-dir /path/to/figures
"""
import argparse
import re
from pathlib import Path

import torch

PREFIX = re.compile(r'^base_model\.model\.')
KEY = re.compile(r'layers\.(\d+)\..*ttt_reader_alignment\.weight$')


def load(ckpt):
    sd = torch.load(Path(ckpt) / 'ttt_params.pt', map_location='cpu', weights_only=True)
    maps = {int(m.group(1)): v.float()
            for k, v in sd.items() if (m := KEY.search(PREFIX.sub('', k)))}
    if not maps:
        raise SystemExit(f'no reader maps in {ckpt}')
    return dict(sorted(maps.items()))


def stats(chi):
    h, d, _ = chi.shape
    eye = torch.eye(d).expand(h, d, d)
    delta = chi - eye
    sv = torch.linalg.svdvals(chi)                       # [h, d] descending
    # polar chi = Q P: Q = U V^T orthogonal, P = V S V^T symmetric PSD
    U, S, Vh = torch.linalg.svd(chi)
    Q = U @ Vh
    P = Vh.transpose(1, 2) @ torch.diag_embed(S) @ Vh
    return dict(sv=sv, disp=delta.flatten(1).norm(dim=1), d=d,
                q=(Q - eye).flatten(1).norm(dim=1),
                p=(P - eye).flatten(1).norm(dim=1))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    metavar='NAME=PATH', help='repeatable; NAME labels the series')
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    runs = {}
    for spec in a.ckpt:
        if '=' not in spec:
            raise SystemExit(f'--ckpt wants NAME=PATH, got {spec}')
        name, path = spec.split('=', 1)
        runs[name] = {l: stats(c) for l, c in load(path).items()}
        print(f'{name}: layers {sorted(runs[name])}')

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(v) for v in runs.values())))
    styles = ['-', '--', ':']

    # (a) spectrum: the near-isometric claim. Median over heads with IQR band.
    ax = axes[0]
    ax.axhspan(0.9, 1.1, color='0.9', zorder=0, label=r'$\sigma \in [0.9, 1.1]$')
    ax.axhline(1.0, color='0.4', lw=0.8, zorder=1)
    for s, (name, layers) in zip(styles, runs.items()):
        for c, (layer, st) in zip(colours, layers.items()):
            sv = st['sv'].numpy()
            x = np.arange(sv.shape[1])
            ax.plot(x, np.median(sv, 0), s, color=c, lw=1.3,
                    label=f'{name} L{layer}' if len(runs) > 1 else f'L{layer}')
            if len(runs) == 1:
                ax.fill_between(x, *np.percentile(sv, [25, 75], axis=0), color=c, alpha=0.18)
    ax.set_xlabel('singular value index'); ax.set_ylabel(r'$\sigma$')
    ax.set_title('(a) spectrum is near-isometric')
    ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 2.1)

    # (b) depth gradient: per-head points, not four means.
    ax = axes[1]
    for off, (name, layers) in enumerate(runs.items()):
        for c, (layer, st) in zip(colours, layers.items()):
            y = st['disp'].numpy() / (2 * st['d']) ** 0.5
            x = np.full_like(y, layer) + (off - (len(runs) - 1) / 2) * 0.6
            ax.scatter(x, y, s=7, color=c, alpha=0.5, edgecolors='none')
            ax.plot([x.mean() - 0.3, x.mean() + 0.3], [y.mean()] * 2, color='k', lw=1.6)
    ax.set_xlabel('follower layer'); ax.set_ylabel(r'$\|\chi-I\|_F / \sqrt{2d}$')
    ax.set_title('(b) correction grows with distance from the writer')
    if len(runs) > 1:
        ax.text(0.02, 0.96, ' / '.join(runs), transform=ax.transAxes, fontsize=7, va='top')

    # (c) polar: rotation and stretch are comparable, so neither term dominates.
    ax = axes[2]
    hi = 0
    for m, (name, layers) in zip(['o', '^', 's'], runs.items()):
        for c, (layer, st) in zip(colours, layers.items()):
            q, p = st['q'].numpy(), st['p'].numpy()
            ax.scatter(q, p, s=9, marker=m, color=c, alpha=0.55, edgecolors='none')
            hi = max(hi, q.max(), p.max())
    ax.plot([0, hi * 1.05], [0, hi * 1.05], color='0.5', lw=0.8, ls='--')
    ax.set_xlabel(r'$\|Q-I\|_F$  (rotation)'); ax.set_ylabel(r'$\|P-I\|_F$  (stretch)')
    ax.set_title('(c) polar parts are comparable')
    ax.set_xlim(0, hi * 1.05); ax.set_ylim(0, hi * 1.05)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'reader_maps.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {out}/reader_maps.png and .pdf')


if __name__ == '__main__':
    main()
