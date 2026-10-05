"""What rotation does chi_l actually apply, and is it more than noise?

plot_reader_maps.py reports ||Q-I||_F, which has no units and no null. This
script works in degrees and carries a control.

A real orthogonal Q has eigenvalues on the unit circle in conjugate pairs
e^{+-i theta}: each pair is a plane of R^d that Q turns by theta. So the
rotation IS a list of angles, and the three panels are that list.

The control matters. For chi = I + E with E small, Q ~= I + skew(E) and
P ~= I + sym(E), and for iid E the two have nearly equal norm
(||sym||^2/||skew||^2 = 1 + 2/(d-1) = 1.016 at d=128). So "the orthogonal and
symmetric parts are comparable" is what ANY small perturbation gives, and a
figure showing it establishes nothing. Every panel here therefore overlays a
null: per head, a Gaussian E rescaled to that head's measured ||chi - I||_F,
pushed through the same polar decomposition. Learned curves that sit on the
null are noise; separation is structure.

    python plot_reader_rotation.py \
      --ckpt NAME=/path/to/stage2/best_ckpt [--ckpt NAME2=...] \
      --out-dir /path/to/figures
"""
import argparse
import re
from pathlib import Path

import torch

PREFIX = re.compile(r'^base_model\.model\.')
KEY = re.compile(r'layers\.(\d+)\..*ttt_reader_alignment\.weight$')
PROBES = 256
SEED = 0


def load(ckpt):
    sd = torch.load(Path(ckpt) / 'ttt_params.pt', map_location='cpu', weights_only=True)
    maps = {int(m.group(1)): v.double()
            for k, v in sd.items() if (m := KEY.search(PREFIX.sub('', k)))}
    if not maps:
        raise SystemExit(f'no reader maps in {ckpt}')
    return dict(sorted(maps.items()))


def orthogonal_factor(chi):
    """Q from the right polar decomposition chi = Q P, in float64."""
    u, _, vh = torch.linalg.svd(chi)
    return u @ vh


def matched_null(chi, generator):
    """I + E per head, with ||E||_F set to that head's measured ||chi - I||_F.

    Matching head by head rather than using one pooled scale keeps the null
    honest for the depth gradient: layer 31 moves further than layer 1, and a
    null that ignored that would manufacture separation at the far layers.
    """
    h, d, _ = chi.shape
    eye = torch.eye(d, dtype=chi.dtype).expand(h, d, d)
    target = (chi - eye).flatten(1).norm(dim=1)
    e = torch.randn(h, d, d, dtype=chi.dtype, generator=generator)
    e = e * (target / e.flatten(1).norm(dim=1)).view(h, 1, 1)
    return eye + e


def angles(chi):
    """|arg lambda| in degrees for every eigenvalue of Q, sorted descending.

    Conjugate pairs contribute the same angle twice, so a d-vector holds d/2
    rotation planes listed twice over. The duplication is uniform and both the
    learned maps and the null go through this function, so the shapes compare.
    """
    lam = torch.linalg.eigvals(orthogonal_factor(chi))
    return lam.angle().abs().rad2deg().sort(dim=1, descending=True).values


def probe_angles(chi, generator):
    """Angle between an isotropic unit probe v and its image chi v, in degrees.

    chi acts on readouts, not on basis vectors, so this is the quantity a
    downstream o_proj sees. Isotropic probes are a stand-in for the real
    readout distribution, which is not in the checkpoint -- they say how far a
    TYPICAL direction moves, not how far the directions that matter move.
    """
    h, d, _ = chi.shape
    v = torch.randn(h, d, PROBES, dtype=chi.dtype, generator=generator)
    v = v / v.norm(dim=1, keepdim=True)
    u = chi @ v
    cos = (v * u).sum(dim=1) / u.norm(dim=1).clamp(min=1e-30)
    return cos.clamp(-1, 1).arccos().rad2deg()


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

    g = torch.Generator().manual_seed(SEED)
    runs = {}
    for spec in a.ckpt:
        if '=' not in spec:
            raise SystemExit(f'--ckpt wants NAME=PATH, got {spec}')
        name, path = spec.split('=', 1)
        per_layer = {}
        for layer, chi in load(path).items():
            null = matched_null(chi, g)
            per_layer[layer] = dict(
                eig=torch.linalg.eigvals(orthogonal_factor(chi)),
                ang=angles(chi), ang_null=angles(null),
                probe=probe_angles(chi, g), probe_null=probe_angles(null, g))
        runs[name] = per_layer
        print(f'{name}: layers {sorted(per_layer)}')

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = sorted({l for v in runs.values() for l in v})
    colour = dict(zip(layers, plt.cm.viridis(np.linspace(0.1, 0.85, len(layers)))))
    style = dict(zip(runs, ['-', '--', ':']))
    marker = dict(zip(runs, ['o', '^', 's']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) the literal picture: where Q sends the unit circle. One run, so the
    # panel reads as a figure rather than a pile; the rest is in (b) and (c).
    ax = axes[0]
    first = next(iter(runs))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color='0.75', lw=0.8, zorder=0)
    for layer, st in runs[first].items():
        lam = st['eig'].flatten().numpy()
        ax.scatter(lam.real, lam.imag, s=3, color=colour[layer], alpha=0.25,
                   edgecolors='none', label=f'L{layer}')
    ax.scatter([1], [0], s=30, marker='x', color='k', zorder=3, label='identity')
    ax.set_aspect('equal'); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r'Re $\lambda(Q)$'); ax.set_ylabel(r'Im $\lambda(Q)$')
    ax.set_title(f'(a) rotation planes of $Q$ ({first})')
    leg = ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # (b) the control. Learned angle profile against the matched null: if the
    # learned curve is steeper the rotation is concentrated in a few planes,
    # if it lies on the null the "rotation" is an artefact of displacement.
    ax = axes[1]
    for name, per_layer in runs.items():
        for layer, st in per_layer.items():
            x = np.arange(1, st['ang'].shape[1] + 1)
            ax.plot(x, st['ang'].mean(0).numpy(), style[name], color=colour[layer], lw=1.4,
                    label=f'{name} L{layer}' if len(runs) > 1 else f'L{layer}')
            ax.plot(x, st['ang_null'].mean(0).numpy(), style[name], color=colour[layer],
                    lw=1.0, alpha=0.35)
    ax.set_xlabel('eigen-direction (sorted; conjugates twice)')
    ax.set_ylabel(r'rotation angle $|\arg\lambda|$ (deg)')
    ax.set_title('(b) angle profile vs matched null (faint)')
    ax.legend(fontsize=7, ncol=2)

    # (c) what a readout actually experiences, learned vs null, per layer.
    ax = axes[2]
    pos, ticks = [], []
    for i, layer in enumerate(layers):
        for j, (name, per_layer) in enumerate(runs.items()):
            if layer not in per_layer:
                continue
            st = per_layer[layer]
            x = i + (j - (len(runs) - 1) / 2) * 0.26
            med = st['probe'].median().item()
            lo, hi = np.percentile(st['probe'].numpy(), [25, 75])
            ax.plot([x, x], [lo, hi], color=colour[layer], lw=2.4, solid_capstyle='butt')
            ax.scatter([x], [med], s=34, marker=marker[name], color=colour[layer],
                       edgecolors='k', linewidths=0.5, zorder=3)
            ax.scatter([x], [st['probe_null'].median().item()], s=22, marker='_',
                       color='0.35', zorder=4)
        pos.append(i); ticks.append(f'L{layer}')
    ax.set_xticks(pos); ax.set_xticklabels(ticks)
    ax.set_xlabel('follower layer (equal spacing; the gap L1->L29 is not to scale)')
    ax.set_ylabel(r'$\angle(v, \chi v)$, isotropic probes (deg)')
    ax.set_title('(c) angle a typical readout is turned through')
    handles = [plt.Line2D([], [], ls='', marker=marker[n], color='0.3', label=n) for n in runs]
    handles.append(plt.Line2D([], [], ls='', marker='_', color='0.35', label='matched null'))
    ax.legend(handles=handles, fontsize=7, loc='upper left')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'reader_rotation.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {out}/reader_rotation.png and .pdf')

    print(f'\n{"run":>10} {"layer":>6} {"median probe angle":>20} {"null":>8} '
          f'{"top-8 planes":>14} {"null":>8}')
    for name, per_layer in runs.items():
        for layer, st in per_layer.items():
            print(f'{name:>10} {layer:>6} {st["probe"].median():>19.2f}deg '
                  f'{st["probe_null"].median():>7.2f} '
                  f'{st["ang"][:, :8].mean():>13.2f}deg {st["ang_null"][:, :8].mean():>7.2f}')


if __name__ == '__main__':
    main()
