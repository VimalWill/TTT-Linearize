"""What rotation does chi_l actually apply, and is it more than noise?

plot_reader_maps.py reports ||Q-I||_F, which has no units and no null. This
script works in degrees and carries a control.

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
        x = data[f'readout_{layer}']
        p = data[f'stretched_{layer}']
        y = data[f'aligned_{layer}']
        basis = data[f'basis_{layer}']
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
        actual[name] = dict(layer=layer, head=meta['head'], readout=x, stretched=p,
            aligned=y, positions=pos, position_fractions=pos_fraction, ids=ids,
            coordinates=coords, unit_coordinates=unit, staged_coordinates=staged,
            metadata=meta, map_angle=meta['rotation_angle_degrees'])
        print(f'{name}: plotting {len(x)} actual readouts from L{layer}, head {meta["head"]}; '
              f'median plane energy {meta["median_readout_energy_in_plane"]:.1%}')

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
                   label='Actual readout x', zorder=2)
        ax.scatter(mapped[:, 0], mapped[:, 1], s=18, color='#147d92', alpha=.72,
                   label='Actual reader output $\\chi x$', zorder=3)
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
        turn = np.degrees(np.arctan2(np.cross(*rays), np.dot(*rays)))
        ax.set_title(f"{name}: actual held-out readouts, L{act['layer']} head {act['head']}\n"
                     f"{act['metadata']['median_readout_energy_in_plane']:.1%} of readout energy "
                     f"in plane; mean direction turns {abs(turn):.0f}$\\degree$")
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
        ax.set_title(f'{name}: checkpoint geometry vs displacement-matched null')
        ax.grid(axis='y', alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], ls='--', color='.5', alpha=.5))
        ax.legend(handles, labels + ['Matched Gaussian null'], fontsize=9)
    fig.suptitle('Reader maps acting on held-out model activations', fontsize=17, y=1.01)
    fig.text(.5, .01, 'Points are real pre-alignment readouts and the corresponding module outputs, '
             'projected onto the selected invariant plane of Q.',
             ha='center', fontsize=9, color='.35')
    fig.tight_layout(rect=(0, .045, 1, .98), h_pad=2.5)
    save_figure(fig, out, 'reader_rotation')
    plt.close(fig)

    fig, axes = plt.subplots(len(runs), 3, figsize=(13, 4.7 * len(runs)), squeeze=False)
    cmap = plt.get_cmap('viridis')
    for row, (name, act) in enumerate(actual.items()):
        coords = act['staged_coordinates']
        colors = cmap(act['position_fractions'])
        extent = float(np.quantile(np.abs(np.concatenate(coords)), .995)) * 1.25
        picks = np.linspace(0, len(coords[0]) - 1, min(8, len(coords[0]))).round().astype(int)
        for col, (ax, points, title) in enumerate(zip(axes[row], coords,
                ['Captured readout x', 'Stretch P x', 'Actual reader output $\\chi x$'])):
            setup_plane(ax, extent)
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=18, alpha=.7, linewidths=0)
            for i in picks:
                if col == 0:
                    continue
                before = coords[col - 1][i]
                ax.annotate('', xy=points[i], xytext=before,
                            arrowprops=dict(arrowstyle='->', color='#d97732', alpha=.4, lw=1))
            ax.set_title(title)
        axes[row, 0].set_ylabel(f"{name} · L{act['layer']} head {act['head']}\nPlane coordinate 2")
        axes[row, 1].set_xlabel(f"Token position color: early → late\n"
            f"Median activation energy in plane: {act['metadata']['median_readout_energy_in_plane']:.1%}")
    fig.suptitle('Original → stretch → rotate on real activations', fontsize=17, y=1.01)
    fig.text(.5, .01, 'Each dot is a captured readout from a held-out prompt. '
             'All panels show projections onto the same selected Q-plane; '
             'the reader output is captured directly from the model.',
             ha='center', fontsize=9, color='.35')
    fig.tight_layout(rect=(0, .045, 1, .98), h_pad=2.5)
    save_figure(fig, out, 'reader_transform_stages')
    plt.close(fig)
    metadata = dict(actual_activations=True, synthetic_vectors=False,
        activation_runs={name: activation_metadata[name] for name in activation_metadata},
        selected_layers={name: act['layer'] for name, act in actual.items()},
        interpretation='Projection onto the strongest invariant plane of Q; '
                       'captured full reader outputs are used in the final stage.')
    (out / 'reader_activation_figure.json').write_text(json.dumps(metadata, indent=2) + '\n')

    print(f'\n{"run":>10} {"layer":>6} {"median activation turn":>24} {"plane energy":>14} {"top-8":>9} {"null":>8} {"effective":>10} {"null":>7} {"ratio":>6}')
    for name, per_layer in runs.items():
        for layer, st in per_layer.items():
            n, n0 = planes(st['ang']), planes(st['ang_null'])
            if layer == actual[name]['layer']:
                # np.load gives ndarrays, which have no .norm; go through torch
                # so the turn is computed the same way as the probe angles.
                x = torch.from_numpy(actual[name]['readout']).double()
                y = torch.from_numpy(actual[name]['aligned']).double()
                cosine = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp(min=1e-30)
                turn = float(cosine.clamp(-1, 1).arccos().rad2deg().median())
                energy = actual[name]['metadata']['median_readout_energy_in_plane']
            else:
                turn, energy = float('nan'), float('nan')
            print(f'{name:>10} {layer:>6} {turn:>22.2f}deg {energy:>13.1%} '
                  f'{st["ang"][:, ::2][:, :8].mean():>8.2f} '
                  f'{st["ang_null"][:, ::2][:, :8].mean():>7.2f} '
                  f'{n:>10.1f} {n0:>7.1f} {n / n0:>6.2f}')
    print(f'{"":>10} eff planes counts distinct planes, out of d/2')


if __name__ == '__main__':
    main()
