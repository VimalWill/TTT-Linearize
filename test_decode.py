"""Prefill+decode must equal one full-sequence forward.

    python3 test_decode.py --cfg Configs/ttt_ar_unified.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_unified/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_unified/best_ckpt
"""
import argparse, torch
from omegaconf import OmegaConf
import LinearTTT  # noqa: F401
from Training.train import build_model_config
from diag_common import load_model

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--adapter', default=None)
ap.add_argument('--prefix', type=int, default=600)
ap.add_argument('--gen', type=int, default=16)
a = ap.parse_args()

cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(a.cfg), resolve=True))
cfg.model.pretrained_model_name_or_path = a.ckpt
mc = build_model_config(cfg)
model = load_model(a.ckpt, mc, a.adapter)
for p in model.parameters():
    p.requires_grad_(False)

torch.manual_seed(0)
V = mc.vocab_size
C = int(cfg.model.lact_chunk_size)


def check(prefix, gen):
    ids = torch.randint(0, V, (1, prefix + gen), device='cuda')
    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=False).logits
        out = model(input_ids=ids[:, :prefix], use_cache=True,
                    cache_position=torch.arange(prefix, device='cuda'))
        got = [out.logits[:, -1]]
        for i in range(gen - 1):
            pos = prefix + i
            out = model(input_ids=ids[:, pos:pos + 1], use_cache=True,
                        cache_position=torch.tensor([pos], device='cuda'),
                        position_ids=torch.tensor([[pos]], device='cuda'))
            got.append(out.logits[:, -1])
        got = torch.stack(got, dim=1)
    r = ref[:, prefix - 1:prefix - 1 + gen]
    d = (got.float() - r.float()).abs()
    agree = (got.argmax(-1) == r.argmax(-1)).float().mean().item()
    crosses = (prefix % C) + gen >= C
    print(f'prefix {prefix:>5}  gen {gen:>3}  crosses chunk boundary: {str(crosses):<5} '
          f'max {d.max().item():.4f}  mean {d.mean().item():.4f}  argmax {agree:.1%}')
    return agree


ok = True
# no boundary crossed -- the frozen-weight path only
ok &= check(600, 16) == 1.0
# boundary crossed mid-continuation -- exercises the deferred chunk update
ok &= check(C - 12, 48) == 1.0
# boundary crossed twice
ok &= check(C - 4, 2 * C + 8) == 1.0
# prefix exactly on a boundary: buffer starts empty
ok &= check(2 * C, 24) == 1.0
print('PASS' if ok else 'FAIL -- do not run generation evals')
