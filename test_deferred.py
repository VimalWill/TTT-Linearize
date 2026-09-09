"""Is the deferred chunk update equivalent to one operator pass?

Pure operator test -- no model, no checkpoint. Compares
    one call over 2C+1 tokens   (two updates, done internally)
against
    one call over C+1 tokens, then a deferred update on the next C
which is exactly what LinearTTTAttention._apply_deferred_chunk does during
decode. If the states differ, decode cannot match a full-sequence forward.

    python3 test_deferred.py
"""
import torch
from LinearTTT.model.LinearizeLlama.ttt_l2 import block_causal_lact_swiglu_l2

torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
B, D, C = 8, 128, 32          # b*h, head dim, chunk (small C keeps it quick)
N = 2 * C
f32 = dict(device=dev, dtype=torch.float32)

w = [torch.randn(B, D, D, **f32) * 0.1 for _ in range(3)]
k = torch.nn.functional.normalize(torch.randn(B, N + 1, D, **f32), dim=-1)
v = torch.nn.functional.normalize(torch.randn(B, N + 1, D, **f32), dim=-1)
lr = [torch.rand(B, N + 1, 1, **f32) * 0.02 for _ in range(3)]
mom = torch.rand(B, N + 1, 1, **f32)
ret = 0.98 + 0.01 * torch.rand(B, N + 1, 1, **f32)

for use_muon in (False, True):
    for use_mom in (False, True):
        kw = dict(chunk_size=C, use_muon=use_muon,
                  momentum=mom if use_mom else None, retention=ret)

        # A: one pass over 2C+1 -> range(0, C+1, C) = [0, C] -> two updates
        sl = lambda x, a, b: x[:, a:b]
        _, a0, a1, a2, _ = block_causal_lact_swiglu_l2(
            *w, sl(k, 0, N + 1), sl(k, 0, N + 1), sl(v, 0, N + 1),
            *[sl(x, 0, N + 1) for x in lr],
            **{**kw, 'momentum': sl(mom, 0, N + 1) if use_mom else None,
               'retention': sl(ret, 0, N + 1)}, return_momentum=True)

        # B: first chunk, then the deferred update on tokens [C, 2C)
        pad = lambda x: torch.cat([x[:, :C], x[:, C - 1:C]], dim=1)
        _, b0, b1, b2, bm = block_causal_lact_swiglu_l2(
            *w, pad(sl(k, 0, C)), pad(sl(k, 0, C)), pad(sl(v, 0, C)),
            *[pad(sl(x, 0, C)) for x in lr],
            **{**kw, 'momentum': pad(sl(mom, 0, C)) if use_mom else None,
               'retention': pad(sl(ret, 0, C))}, return_momentum=True)
        _, b0, b1, b2, _ = block_causal_lact_swiglu_l2(
            b0, b1, b2, pad(sl(k, C, N)), pad(sl(k, C, N)), pad(sl(v, C, N)),
            *[pad(sl(x, C, N)) for x in lr],
            **{**kw, 'momentum': pad(sl(mom, C, N)) if use_mom else None,
               'retention': pad(sl(ret, C, N))},
            init_momentum=bm, return_momentum=True)

        d = max((x - y).abs().max().item() for x, y in ((a0, b0), (a1, b1), (a2, b2)))
        print(f'muon={use_muon!s:<5} momentum={use_mom!s:<5}  '
              f'max |w_full - w_deferred| = {d:.3e}  '
              f'{"OK" if d < 1e-4 else "MISMATCH"}')
