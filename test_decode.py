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
ids = torch.randint(0, V, (1, a.prefix + a.gen), device='cuda')

with torch.no_grad():
    # reference: one teacher-forced pass over the whole thing
    ref = model(input_ids=ids, use_cache=False).logits

    # prefill on the prefix, then decode the rest one token at a time
    out = model(input_ids=ids[:, :a.prefix], use_cache=True,
                cache_position=torch.arange(a.prefix, device='cuda'))
    got = [out.logits[:, -1]]
    for i in range(a.gen - 1):
        pos = a.prefix + i
        out = model(input_ids=ids[:, pos:pos + 1], use_cache=True,
                    cache_position=torch.tensor([pos], device='cuda'),
                    position_ids=torch.tensor([[pos]], device='cuda'))
        got.append(out.logits[:, -1])
    got = torch.stack(got, dim=1)

r = ref[:, a.prefix - 1:a.prefix - 1 + a.gen]
d = (got.float() - r.float()).abs()
print(f'prefix {a.prefix}  decoded {a.gen}')
print(f'max abs logit diff  {d.max().item():.6f}')
print(f'mean abs logit diff {d.mean().item():.6f}')
print(f'argmax agreement    {(got.argmax(-1) == r.argmax(-1)).float().mean().item():.1%}')
