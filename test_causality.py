"""Do future tokens change earlier positions' logits?

A causal LM must satisfy: changing token t leaves logits at every position < t
bit-identical. In a shared TTT group each non-leader layer starts from the
previous member's state AFTER the whole sequence, so early positions may read a
memory fitted on future tokens. If so, the LM loss is contaminated.

    python3 test_causality.py --cfg Configs/ttt_ar_unified.yml \
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
ap.add_argument('--seq-len', type=int, default=2048)
a = ap.parse_args()

cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(a.cfg), resolve=True))
cfg.model.pretrained_model_name_or_path = a.ckpt
mc = build_model_config(cfg)
model = load_model(a.ckpt, mc, a.adapter)
for p in model.parameters():
    p.requires_grad_(False)

torch.manual_seed(0)
N = a.seq_len
ids = torch.randint(0, mc.vocab_size, (1, N), device='cuda')

with torch.no_grad():
    base = model(input_ids=ids, use_cache=False).logits.float()
    for flip in (N - 1, N // 2):
        alt = ids.clone()
        alt[0, flip] = (alt[0, flip] + 12345) % mc.vocab_size
        got = model(input_ids=alt, use_cache=False).logits.float()
        # positions strictly before the edit must be untouched
        d = (got[:, :flip] - base[:, :flip]).abs()
        print(f'edited token at {flip:>5}  ->  max |dlogit| at positions < {flip}: '
              f'{d.max().item():.3e}   {"CAUSAL" if d.max().item() < 1e-3 else "*** NON-CAUSAL ***"}')
