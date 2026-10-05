"""What did the learned reader alignment maps actually do?

chi_l is the per-head [d, d] map a FOLLOWER applies to a readout that arrives
in the LEADER's value space (ReaderOutputAlignment). It is identity-initialised,
and until 2026-09-28 it could not move: bf16's ULP near 1.0 is 2^-7 = 0.0078
against an AdamW step of 2e-4, so every update rounded away. Measurement 3 is
therefore the empirical check on that fix -- on any pre-fix checkpoint
||chi - I||_F is exactly zero.

Three measurements, CPU only, straight from ttt_params.pt:

  1. orthogonality  ||chi^T chi - I||_F / sqrt(d)   near 0 => close to a rotation
  2. spectrum       singular values of chi          all ~1 => rotation, not rescaling
  3. displacement   ||chi - I||_F                   > 0   => it learned something
  4. rotated        ||chi - I||_F / sqrt(2d)        1.0   => as far as a random rotation
  5. worst case     ||chi - I||_2 (operator norm)   how much the worst direction moves

Raw Frobenius distance is not interpretable alone: in d dimensions a Haar
rotation sits at ||Q - I||_F = sqrt(2d), which is 16 at d=128. Measurement 4
rescales so identity reads 0 and a random rotation reads 1.

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
        print(f'{"":>6} {"":>6} rotated {rot.mean():6.4f} of a Haar rotation; '
              f'worst-direction ||chi-I||_2 {op.mean():6.4f} [{op.min():.4f},{op.max():.4f}]')
        summary[layer] = dict(
            heads=h, dim=d,
            displacement=dict(mean=disp.mean().item(), min=disp.min().item(), max=disp.max().item()),
            orthogonality=dict(mean=orth.mean().item(), min=orth.min().item(), max=orth.max().item()),
            singular=dict(mean=sv.mean().item(), min=sv.min().item(), max=sv.max().item(),
                          spread=(sv.max(dim=1).values - sv.min(dim=1).values).mean().item()),
            rotated_fraction=dict(mean=rot.mean().item(), min=rot.min().item(), max=rot.max().item()),
            operator=dict(mean=op.mean().item(), min=op.min().item(), max=op.max().item()))

    moved = max(s['displacement']['max'] for s in summary.values())
    print(f'\nlargest ||chi - I||_F over all heads: {moved:.6f}')
    print('  -> the maps NEVER MOVED; this is a pre-fix checkpoint (bf16 rounded every update)'
          if moved < 1e-6 else '  -> the maps trained')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(ckpt=a.ckpt, layers=summary), indent=2) + '\n')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
