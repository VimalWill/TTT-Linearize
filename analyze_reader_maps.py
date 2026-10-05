"""What did the learned reader alignment maps actually do?

chi_l is the per-head [d, d] map a FOLLOWER applies to a readout that arrives
in the LEADER's value space (ReaderOutputAlignment). It is identity-initialised.
Departure from identity establishes a changed map, not its task benefit.

Measurements, CPU only, straight from ttt_params.pt:

  1. orthogonality  ||chi^T chi - I||_F / sqrt(d)   near 0 => close to a rotation
  2. spectrum       singular values of chi          all ~1 => rotation, not rescaling
  3. displacement   ||chi - I||_F                   > 0   => it learned something
  4. displacement   ||chi - I||_F / sqrt(2d)        distance on a Haar reference scale
  5. worst case     ||chi - I||_2 (operator norm)   how much the worst direction moves

For Haar orthogonal matrices, the expected squared distance from identity is
2d. This reference scale is NOT a rotation angle or fraction of rotation.
The right polar decomposition chi = Q P separates orthogonal reorientation
(Q, possibly including reflection) from symmetric positive-semidefinite stretch
(P). For singular chi the orthogonal factor is not unique on the null space.

Reported per head, since four follower layers give only four points but
4 x 32 heads give a distribution.

    python analyze_reader_maps.py --ckpt .../ttt_ar_unified_longalpaca/best_ckpt
"""
import argparse
import json
import re
from pathlib import Path

import torch

PREFIX = re.compile(r'^base_model\.model\.')
KEY = re.compile(r'layers\.(\d+)\..*ttt_reader_alignment\.weight$')


def polar_factors(weight):
    """Return Q, P, singular values for weight = Q @ P, in CPU float64."""
    matrix = weight.detach().to(device='cpu', dtype=torch.float64)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('Reader maps must be square')
    if not torch.isfinite(matrix).all():
        raise ValueError('Reader maps must be finite')
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    q = u @ vh
    p = vh.transpose(-1, -2) @ (singular.unsqueeze(-1) * vh)
    return q, p, singular


def apply_reader_component(modules, component):
    """Replace readers in memory only; leave all other checkpoint weights fixed."""
    if component not in ('orthogonal', 'stretch'):
        raise ValueError('Expected orthogonal or stretch')
    aligned = [m for m in modules if getattr(m, 'ttt_reader_alignment', None) is not None]
    if not aligned:
        raise ValueError('Reader component control requires enabled reader maps')
    with torch.no_grad():
        for module in aligned:
            weight = module.ttt_reader_alignment.weight
            q, p, _ = polar_factors(weight)
            weight.copy_((q if component == 'orthogonal' else p).to(weight))
            module._identity_reader_alignment = False
    return len(aligned)


def stats(tensor):
    return dict(mean=tensor.mean().item(), min=tensor.min().item(), max=tensor.max().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='stage-2 dir holding ttt_params.pt')
    ap.add_argument('--out', default=None, help='write a JSON summary here')
    a = ap.parse_args()

    path = Path(a.ckpt) / 'ttt_params.pt'
    if not path.is_file():
        raise SystemExit(f'no ttt_params.pt in {a.ckpt}')
    sd = torch.load(path, map_location='cpu', weights_only=True)
    maps = {int(m.group(1)): v.float()
            for k, v in sd.items() if (m := KEY.search(PREFIX.sub('', k)))}
    if not maps:
        raise SystemExit('no ttt_reader_alignment weights; this checkpoint has no readers')

    print(f'{len(maps)} reader maps from {path}')
    print(f'\n{"layer":>6} {"heads":>6} {"||chi-I||_F":>22} {"||chiT chi-I||/sqrt d":>24} '
          f'{"sigma":>22}')
    print(f'{"":>6} {"":>6} {"mean [min, max]":>22} {"mean [min, max]":>24} {"mean [min, max]":>22}')
    summary = {}
    for layer in sorted(maps):
        chi = maps[layer]                      # [heads, d, d]
        h, d, d2 = chi.shape
        if d != d2:
            raise SystemExit(f'layer {layer}: chi is {d}x{d2}, not square')
        eye = torch.eye(d).expand(h, d, d)
        delta = chi - eye
        disp = delta.flatten(1).norm(dim=1)
        rot = disp / (2 * d) ** 0.5           # 0 = identity, 1 = Haar rotation
        op = torch.linalg.matrix_norm(delta, ord=2)
        orth = (chi.transpose(1, 2) @ chi - eye).flatten(1).norm(dim=1) / d ** 0.5
        sv = torch.linalg.svdvals(chi)         # [heads, d]
        f = lambda t: f'{t.mean():7.4f} [{t.min():6.4f},{t.max():6.4f}]'
        print(f'{layer:>6} {h:>6} {f(disp):>22} {f(orth):>24} {f(sv):>22}')
        print(f'{"":>6} {"":>6} normalized displacement {rot.mean():6.4f} (Haar reference scale); '
              f'worst-direction ||chi-I||_2 {op.mean():6.4f} [{op.min():.4f},{op.max():.4f}]')
        # Diagonal and off-diagonal face different bf16 resolution: entries
        # start at 1.0 on the diagonal, where the ULP is 2^-7 = 0.0078 and an
        # AdamW step of 2e-4 rounds away, and at 0.0 off it, where bf16 is
        # much finer. Off-diagonal entries can change while the diagonal is
        # pinned, so the two must be reported separately.
        diag = torch.diagonal(delta, dim1=1, dim2=2).norm(dim=1)
        off = (delta.flatten(1).norm(dim=1) ** 2 - diag ** 2).clamp(min=0).sqrt()
        print(f'{"":>6} {"":>6} diagonal {diag.mean():7.4f} [{diag.min():.4f},{diag.max():.4f}]'
              f'   off-diagonal {off.mean():7.4f} [{off.min():.4f},{off.max():.4f}]')
        # mean/min/max hide the shape. "sigma ~ 1" is a claim about the whole
        # spectrum, so measure the whole spectrum: what fraction is actually
        # near 1, how many directions the map effectively uses, and how far the
        # extremes sit. A rotation gives near=1.00, effective rank d, kappa 1.
        near = ((sv > 0.9) & (sv < 1.1)).float().mean(dim=1)
        eff = sv.sum(dim=1) ** 2 / (sv ** 2).sum(dim=1) / d      # participation ratio
        kappa = sv.max(dim=1).values / sv.min(dim=1).values.clamp(min=1e-12)
        q = torch.quantile(sv, torch.tensor([0.05, 0.5, 0.95]), dim=1).mean(dim=1)
        print(f'{"":>6} {"":>6} sigma in [0.9,1.1]: {100*near.mean():5.1f}%   '
              f'effective rank {eff.mean():5.3f} of full   condition {kappa.mean():7.2f}')
        print(f'{"":>6} {"":>6} sigma percentiles  p5 {q[0]:6.4f}  p50 {q[1]:6.4f}  p95 {q[2]:6.4f}')
        summary[layer] = dict(
            heads=h, dim=d,
            displacement=dict(mean=disp.mean().item(), min=disp.min().item(), max=disp.max().item()),
            orthogonality=dict(mean=orth.mean().item(), min=orth.min().item(), max=orth.max().item()),
            singular=dict(mean=sv.mean().item(), min=sv.min().item(), max=sv.max().item(),
                          spread=(sv.max(dim=1).values - sv.min(dim=1).values).mean().item()),
            normalized_displacement=stats(rot),
            operator=dict(mean=op.mean().item(), min=op.min().item(), max=op.max().item()),
            diagonal=dict(mean=diag.mean().item(), min=diag.min().item(), max=diag.max().item()),
            offdiagonal=dict(mean=off.mean().item(), min=off.min().item(), max=off.max().item()),
            spectrum=dict(fraction_near_one=near.mean().item(), effective_rank=eff.mean().item(),
                          condition=kappa.mean().item(),
                          p5=q[0].item(), p50=q[1].item(), p95=q[2].item()))
        q, p, singular = polar_factors(chi)
        q_disp = torch.linalg.matrix_norm(q - eye.double(), ord='fro')
        p_disp = torch.linalg.matrix_norm(p - eye.double(), ord='fro')
        det_q = torch.linalg.det(q)
        reconstruction = torch.linalg.matrix_norm(q @ p - chi.double(), ord='fro')
        near_singular = singular.min(dim=-1).values <= singular.max(dim=-1).values * 1e-8
        print(f'{"":>6} {"":>6} polar ||Q-I||_F {f(q_disp)}; ||P-I||_F {f(p_disp)}')
        print(f'{"":>6} {"":>6} reflection heads: {int((det_q < 0).sum())}/{h}; '
              f'near-singular heads: {int(near_singular.sum())}/{h}')
        summary[layer]['polar'] = dict(
            convention='chi = Q @ P; P acts first on column readouts',
            orthogonal_displacement=stats(q_disp), stretch_displacement=stats(p_disp),
            reflection_heads=int((det_q < 0).sum()),
            near_singular_heads=int(near_singular.sum()),
            reconstruction_error=stats(reconstruction),
            per_head=[dict(head=i, orthogonal_displacement=q_disp[i].item(),
                           stretch_displacement=p_disp[i].item(), determinant_q=det_q[i].item(),
                           singular_min=singular[i].min().item(), singular_max=singular[i].max().item())
                      for i in range(h)])

    moved = max(s['displacement']['max'] for s in summary.values())
    print(f'\nlargest ||chi - I||_F over all heads: {moved:.6f}')
    print('  -> the maps are approximately identity'
          if moved < 1e-6 else '  -> the maps differ from identity; test task benefit with ablations')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(ckpt=a.ckpt, layers=summary), indent=2) + '\n')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
