"""What rotation does chi_l actually apply, and is it more than noise?

An earlier version of this analysis reported ||Q-I||_F, which has no units
and no null. This script works in degrees and carries a control.

A real orthogonal Q has eigenvalues on the unit circle in conjugate pairs
e^{+-i theta}: each pair is a plane of R^d that Q turns by theta. So the
rotation has a spectrum of plane angles. The main figure plots paired, actual
held-out reader inputs and outputs in a selected invariant plane, alongside the
distinct-plane angle profiles. A second figure shows original -> stretch ->
rotate on those same real activations, projected onto that plane.

The control matters. For chi = I + E with E small, Q ~= I + skew(E) and
P ~= I + sym(E), and for iid E the two have nearly equal norm
(||sym||^2/||skew||^2 = 1 + 2/(d-1) = 1.016 at d=128). So "the orthogonal and
symmetric parts are comparable" is what ANY small perturbation gives, and a
figure showing it alone does not distinguish learned structure from noise.
The angle profile therefore overlays a null: per head, a Gaussian E rescaled to that head's measured ||chi - I||_F,
pushed through the same polar decomposition. Separation indicates structure
relative to this particular null; overlap does not establish that a map is noise.

The activation panels use captured pre-alignment reader activations from held-out
prompts, paired with the actual module outputs. No synthetic vectors are used.

    python plot_reader_rotation.py \
      --ckpt NAME=/path/to/stage2/best_ckpt [--ckpt NAME2=...] \
      --out-dir /path/to/figures
"""
import argparse
import json
import re
from pathlib import Path

import torch
from analyze_reader_maps import polar_factors

PREFIX = re.compile(r'^base_model\.model\.')
KEY = re.compile(r'layers\.(\d+)\..*ttt_reader_alignment\.weight$')
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
    q = orthogonal_factor(chi)
    if q.shape[-1] % 2 or (torch.linalg.det(q) < 0).any():
        raise ValueError('Distinct-plane profiles require even-dimensional proper rotations; '
                         'reflection heads must be analyzed separately')
    lam = torch.linalg.eigvals(q)
    return lam.angle().abs().rad2deg().sort(dim=1, descending=True).values


def planes(ang):
    """Effective number of rotation planes, by participation ratio on angles.

    (sum theta)^2 / sum theta^2 over the d/2 distinct planes. A map that turns
    k planes equally and leaves the rest alone reads k; one that spreads the
    same displacement over everything reads d/2. Reported against the null
    rather than against d/2, since finite samples never reach the ceiling.
    """
    half = ang[:, ::2]                                   # drop the conjugate copy
    return (half.sum(1) ** 2 / half.pow(2).sum(1).clamp(min=1e-30)).mean()


def strongest_plane(q):
    """An orthonormal basis B for a strongest invariant rotation plane of Q."""
    q = q.double()
    if q.shape[0] < 2 or torch.linalg.det(q) < 0:
        raise ValueError('A rotation plane requires a proper rotation of dimension >= 2')
    lam, vectors = torch.linalg.eig(q)
    # A pair of -1 eigenvalues is a 180-degree plane, stronger than any
    # complex pair. For an orthogonal matrix that eigenspace is real.
    if ((lam.real < -1 + 1e-12) & (lam.imag.abs() < 1e-9)).sum() >= 2:
        _, eigenvectors = torch.linalg.eigh((q + q.T) / 2)
        basis = eigenvectors[:, :2]
    elif (lam.imag > 1e-9).any():
        scores = torch.where(lam.imag > 1e-9, lam.angle(), -torch.ones_like(lam.real))
        v = vectors[:, scores.argmax()]
        basis, _ = torch.linalg.qr(torch.stack((v.real, v.imag), dim=1))
    else:  # Identity: any plane is invariant.
        basis = torch.eye(q.shape[0], dtype=q.dtype)[:, :2]
    reduced = basis.T @ q @ basis
    if torch.atan2(reduced[1, 0], reduced[0, 0]) < 0:
        basis[:, 1] *= -1
        reduced = basis.T @ q @ basis
    if not torch.allclose(q @ basis, basis @ reduced, atol=1e-7, rtol=1e-7):
        raise ValueError('Could not extract an invariant plane')
    return basis, reduced


def select_example(per_layer, layer=None, head=None):
    """Default: most-rotated layer, median head by its strongest plane angle."""
    if layer is None:
        layer = max(per_layer, key=lambda l: float(per_layer[l]['ang'][:, 0].mean()))
    if layer not in per_layer:
        raise ValueError(f'Example layer {layer} has no reader map')
    st = per_layer[layer]
    if head is None:
        head = int(st['ang'][:, 0].argsort()[st['ang'].shape[0] // 2])
    if head < 0 or head >= st['chi'].shape[0]:
        raise ValueError(f'Example head {head} is out of range')
    chi = st['chi'][head]
    q, p, _ = polar_factors(chi)
    basis, q2 = strongest_plane(q)
    p2 = basis.T @ p @ basis
    # Stretch can send a vector outside Q's invariant plane. The displayed
    # stages are exact orthogonal projections, not a claim of a closed 2D map.
    pb = p @ basis
    leakage = float((pb - basis @ p2).square().sum() / pb.square().sum().clamp(min=1e-30))
    angle = float(torch.atan2(q2[1, 0], q2[0, 0]).rad2deg())
    return dict(layer=layer, head=head, angle_degrees=angle, q2=q2, p2=p2,
                stretch_out_of_plane_energy_fraction=leakage)


def setup_plane(ax, radius=1.3):
    ax.set_aspect('equal')
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.axhline(0, color='.88', lw=.8, zorder=0)
    ax.axvline(0, color='.88', lw=.8, zorder=0)
    ax.set_xlabel('Plane coordinate 1')
    ax.set_ylabel('Plane coordinate 2')


def save_figure(fig, out, stem):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {out}/{stem}.png and .pdf')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', required=True,
                    metavar='NAME=PATH', help='repeatable; NAME labels the series')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--activations', action='append', required=True, metavar='NAME=FILE',
                    help='real captured readouts .npz; repeat once per --ckpt run')
    ap.add_argument('--activation-layer', type=int,
                    help='layer shown; default picks strongest selected captured layer per run')
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
            per_layer[layer] = dict(chi=chi, ang=angles(chi), ang_null=angles(null))
        runs[name] = per_layer
        print(f'{name}: layers {sorted(per_layer)}')

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    layers = sorted({l for v in runs.values() for l in v})
    colour = dict(zip(layers, plt.cm.viridis(np.linspace(0.1, 0.85, len(layers)))))
    activation_files = {}
    activation_metadata = {}
    for spec in a.activations:
        if '=' not in spec:
            raise SystemExit(f'--activations wants NAME=FILE, got {spec}')
        name, filename = spec.split('=', 1)
        if name in activation_files:
            raise SystemExit(f'duplicate activation data for {name}')
        activation_files[name] = np.load(filename)
        activation_metadata[name] = json.loads(Path(filename).with_suffix('.json').read_text())
    if set(activation_files) != set(runs):
        raise SystemExit(f'Activation names {sorted(activation_files)} must match checkpoint names {sorted(runs)}')

    actual = {}
    for name, data in activation_files.items():
        available = sorted(int(k.split('_', 1)[1]) for k in data.files if k.startswith('readout_'))
        if not available:
            raise ValueError(f'{name}: no captured readout activations')
        layer = a.activation_layer if a.activation_layer is not None else max(
            available, key=lambda l: float(runs[name][l]['ang'][:, 0].mean()))
        if layer not in available or layer not in runs[name]:
            raise ValueError(f'{name}: layer {layer} has no captured activation/map')
        meta = activation_metadata[name]['layers'][str(layer)]
        # Arrays are [samples, heads, dim] and bases [heads, dim, 2]. The
        # scatter can only draw one plane, so it draws the display head; the
        # table below summarises every head.
        display = meta['display_head']
        x = data[f'readout_{layer}'][:, display]
        p = data[f'stretched_{layer}'][:, display]
        y = data[f'aligned_{layer}'][:, display]
        basis = data[f'basis_{layer}'][display]
        pos = data[f'token_position_{layer}']
        pos_fraction = data[f'position_fraction_{layer}']
        ids = data[f'example_id_{layer}']
        coords = [v @ basis for v in (x, p, y)]
        # Raw coordinates are ~0.03 long, so a scatter of them is a smudge and
        # the rotation reads as a shear inside it. Two rescalings fix that
        # without touching the geometry:
        #   unit  -- each vector by its OWN norm, so radius is the square root
        #            of that vector's plane energy and the only thing left to
        #            see is the angle. This is the rotation panel.
        #   staged-- all three stages by the ORIGINAL norm, so the stretch
        #            stays visible as a change in radius. This is the stages
        #            figure, where losing the stretch would defeat the point.
        norms = [np.linalg.norm(v, axis=1, keepdims=True).clip(1e-30) for v in (x, p, y)]
        unit = [(v / n) @ basis for v, n in zip((x, p, y), norms)]
        staged = [(v / norms[0]) @ basis for v in (x, p, y)]
        # Every captured layer, summarised over heads. The capture already
        # reduced each head over samples, so this is a spread across heads:
        # the quantity that says whether one low head is a quiet reader or an
        # unlucky draw.
        turns = {}
        for other in available:
            if other not in runs[name]:
                continue
            per_head = activation_metadata[name]['layers'][str(other)]['per_head']
            t = np.array([i['median_turn_degrees'] for i in per_head])
            e = np.array([i['median_energy_in_plane'] for i in per_head
                          if i['has_plane']])
            turns[other] = dict(
                turn=float(np.median(t)), turn_iqr=np.percentile(t, [25, 75]),
                energy=float(np.median(e)), energy_iqr=np.percentile(e, [25, 75]),
                heads=len(t), planar=len(e))
        actual[name] = dict(layer=layer, head=display, readout=x, stretched=p,
            aligned=y, positions=pos, position_fractions=pos_fraction, ids=ids,
            coordinates=coords, unit_coordinates=unit, staged_coordinates=staged,
            metadata=meta, map_angle=meta['display_rotation_angle_degrees'], turns=turns,
            # The panels draw ONE head, so they must quote that head's energy.
            # The over-heads median belongs in the table, not under a scatter.
            head_energy=next(i['median_energy_in_plane'] for i in meta['per_head']
                             if i['head'] == display))
        print(f'{name}: plotting {len(x)} readouts from L{layer}, head {display}; '
              f'median plane energy over {meta["heads"]} heads '
              f'{meta["median_energy_in_plane_over_heads"]:.1%}')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(len(runs), 2, figsize=(13, 4.8 * len(runs)), squeeze=False,
                             gridspec_kw={'width_ratios': [1, 1.5]})
    for row, (name, per_layer) in enumerate(runs.items()):
        act = actual[name]
        ax = axes[row, 0]
        original, mapped = act['unit_coordinates'][0], act['unit_coordinates'][2]
        setup_plane(ax, 1.15)
        circle = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(circle), np.sin(circle), color='.8', lw=.9, zorder=1)
        ax.scatter(original[:, 0], original[:, 1], s=15, color='.55', alpha=.45,
                   label='Readout, before the map', zorder=2)
        ax.scatter(mapped[:, 0], mapped[:, 1], s=18, color='#147d92', alpha=.72,
                   label='After the reader map', zorder=3)
        # Circular mean of each cloud, drawn to the unit circle. The angle the
        # two rays subtend IS the median turn quoted in the title, so the
        # reader can measure the claim off the picture.
        rays = []
        for points, shade in ((original, '.35'), (mapped, '#0d5c6b')):
            direction = (points / np.linalg.norm(points, axis=1, keepdims=True).clip(1e-30)).mean(0)
            direction = direction / max(np.linalg.norm(direction), 1e-30)
            rays.append(direction)
            ax.annotate('', xy=direction, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=shade, lw=2, alpha=.9))
        # np.cross on 2-vectors is deprecated in numpy 2; the z-component is
        # the only one a plane has, so write it out.
        cross = rays[0][0] * rays[1][1] - rays[0][1] * rays[1][0]
        turn = np.degrees(np.arctan2(cross, np.dot(*rays)))
        ax.set_title(f"{name}, layer {act['layer']} head {act['head']}\n"
                     f"{act['head_energy']:.0%} of readout energy "
                     f"in this plane; mean direction turns {abs(turn):.0f} degrees")
        ax.set_xlabel('Plane coordinate 1 (unit-normalized)')
        ax.set_ylabel('Plane coordinate 2 (unit-normalized)')
        ax.legend(fontsize=8, loc='upper left')

        ax = axes[row, 1]
        for layer, st in per_layer.items():
            ang, null = st['ang'][:, ::2], st['ang_null'][:, ::2]
            xaxis = np.arange(1, ang.shape[1] + 1)
            ax.plot(xaxis, ang.mean(0).numpy(), color=colour[layer], lw=2, label=f'L{layer}')
            ax.plot(xaxis, null.mean(0).numpy(), '--', color=colour[layer], lw=1.2, alpha=.4)
        ax.set_xlabel('Distinct rotation plane (ranked within each head)')
        ax.set_ylabel('Mean rotation angle across heads (degrees)')
        ax.set_title(f'{name}: learned turn per plane, against chance (dashed)')
        ax.grid(axis='y', alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], ls='--', color='.5', alpha=.5))
        ax.legend(handles, labels + ['Chance, same total change'], fontsize=9)
    fig.suptitle('Reader maps acting on held-out model activations', fontsize=17, y=1.01)
    fig.text(.5, .01, 'Each dot is one readout from a held-out document, before and after the '
             'reader map, shown in the plane the map turns hardest.',
             ha='center', fontsize=9, color='.35')
    fig.tight_layout(rect=(0, .045, 1, .98), h_pad=2.5)
    save_figure(fig, out, 'reader_rotation')
    plt.close(fig)

    # Stages figure, drawn to publication style: a serif face, limits fitted
    # to the data rather than a fixed frame, the previous stage ghosted behind
    # the current one so the motion is visible inside a single panel, and one
    # shared colour bar instead of a legend per panel.
    with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'stix',
                         'axes.spines.top': True, 'axes.spines.right': True,
                         'axes.edgecolor': '.2', 'axes.linewidth': .8,
                         'xtick.direction': 'out', 'ytick.direction': 'out',
                         'font.size': 11}):
        fig, axes = plt.subplots(len(runs), 3, figsize=(13.2, 3.5 * len(runs)),
                                 squeeze=False, sharex='row', sharey='row')
        cmap = plt.get_cmap('viridis')
        titles = ['Readout', 'After stretch', 'After rotation']
        for row, (name, act) in enumerate(actual.items()):
            # Raw plane coordinates: with limits fitted to the data there is
            # nothing to gain from rescaling, and the true spread stays visible.
            coords = act['coordinates']
            colors = cmap(act['position_fractions'])
            allpts = np.concatenate(coords)
            lo, hi = allpts.min(0), allpts.max(0)
            pad = (hi - lo).max() * .08
            picks = np.linspace(0, len(coords[0]) - 1, min(10, len(coords[0]))).round().astype(int)
            for col, (ax, points, title) in enumerate(zip(axes[row], coords, titles)):
                if col:
                    before = coords[col - 1]
                    ax.scatter(before[:, 0], before[:, 1], s=9, color='.82',
                               linewidths=0, zorder=1)
                    for i in picks:
                        ax.annotate('', xy=points[i], xytext=before[i], zorder=2,
                                    arrowprops=dict(arrowstyle='->', color='.45',
                                                    lw=.7, shrinkA=0, shrinkB=0))
                ax.scatter(points[:, 0], points[:, 1], c=colors, s=22, alpha=.85,
                           linewidths=0, zorder=3)
                ax.set_xlim(lo[0] - pad, hi[0] + pad)
                ax.set_ylim(lo[1] - pad, hi[1] + pad)
                ax.set_aspect('equal', adjustable='box')
                if row == 0:
                    ax.set_title(title, pad=9)
            axes[row, 0].set_ylabel(f"{name}, layer {act['layer']} head {act['head']}\n"
                                    f"Plane coordinate 2")
        fig.supxlabel('Plane coordinate 1', y=.04)
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axes, fraction=.02, pad=.015)
        bar.set_ticks([0, 1]); bar.set_ticklabels(['early', 'late'])
        bar.set_label('Token position', labelpad=8)
        bar.outline.set_linewidth(.8)
        energies = ', '.join(f"{n}: {a['head_energy']:.0%}" for n, a in actual.items())
        fig.text(.5, -.02, 'Each dot is one readout from a held-out document; the previous '
                 f'stage is ghosted in grey. Readout energy in this plane -- {energies}.',
                 ha='center', fontsize=9.5, color='.35')
    save_figure(fig, out, 'reader_transform_stages')
    plt.close(fig)
    metadata = dict(actual_activations=True, synthetic_vectors=False,
        activation_runs={name: activation_metadata[name] for name in activation_metadata},
        selected_layers={name: act['layer'] for name, act in actual.items()},
        interpretation='Projection onto the strongest invariant plane of Q; '
                       'captured full reader outputs are used in the final stage.')
    (out / 'reader_activation_figure.json').write_text(json.dumps(metadata, indent=2) + '\n')

    heads = {n: next(iter(a['turns'].values()))['heads'] for n, a in actual.items() if a['turns']}
    print(f'\n  activation columns: median over heads [inter-quartile range], '
          f'each head a median over samples')
    print(f'{"run":>10} {"layer":>6} {"turn (deg)":>21} {"energy in plane":>23} '
          f'{"top-8":>8} {"null":>7} {"effective":>10} {"null":>7} {"ratio":>6}')
    for name, per_layer in runs.items():
        for layer, st in per_layer.items():
            n, n0 = planes(st['ang']), planes(st['ang_null'])
            c = actual[name]['turns'].get(layer)
            turn = (f"{c['turn']:6.1f} [{c['turn_iqr'][0]:4.1f},{c['turn_iqr'][1]:5.1f}]"
                    if c else f'{"-":>19}')
            energy = (f"{c['energy']:5.1%} [{c['energy_iqr'][0]:4.1%},{c['energy_iqr'][1]:5.1%}]"
                      if c else f'{"-":>21}')
            print(f'{name:>10} {layer:>6} {turn:>21} {energy:>23} '
                  f'{st["ang"][:, ::2][:, :8].mean():>8.2f} '
                  f'{st["ang_null"][:, ::2][:, :8].mean():>7.2f} '
                  f'{n:>10.1f} {n0:>7.1f} {n / n0:>6.2f}')
    print(f'{"":>10} eff planes counts distinct planes, out of d/2; '
          f'heads per layer: {heads}')
    for name, a in actual.items():
        short = {l: c['heads'] - c['planar'] for l, c in a['turns'].items() if c['planar'] < c['heads']}
        if short:
            print(f'{"":>10} {name}: heads with no proper rotation plane (turn still '
                  f'reported, plane energy omitted): {short}')


if __name__ == '__main__':
    main()
