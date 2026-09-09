"""Do future tokens change earlier positions' logits?

A causal LM must satisfy: changing token t leaves logits at every position < t
bit-identical. In a shared TTT group each non-leader layer starts from the
previous member's state AFTER the whole sequence, so early positions read a
memory fitted on future tokens and the LM loss is contaminated.

    python3 test_unify_tmp.py --cfg Configs/ttt_ar_unified.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_unified/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_unified/best_ckpt

Run it against a per-layer config too -- that must come back CAUSAL, and is what
tells you the harness is measuring the sharing rather than something else.
"""
import argparse, math, torch
from omegaconf import OmegaConf
import LinearTTT  # noqa: F401
from Training.train import build_model_config
from eval import load_model

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--adapter', default=None)
ap.add_argument('--seq-len', type=int, default=2048)
ap.add_argument('--tol', type=float, default=1e-3)
a = ap.parse_args()

cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(a.cfg), resolve=True))
cfg.model.pretrained_model_name_or_path = a.ckpt
mc = build_model_config(cfg)

N, C = a.seq_len, int(cfg.model.lact_chunk_size)
if N <= 2 * C:
    raise SystemExit(f'--seq-len must exceed 2*lact_chunk_size ({2 * C}) or the '
                     'operator applies at most one chunk and there is nothing to test')

# The operator's loop is range(0, N - C, C), so it applies ceil((N - C) / C)
# chunks and everything from C*n_upd on is tail: read out, never written. A flip
# there cannot reach the state and reports CAUSAL whatever the model does, which
# is why the edit positions are pinned inside the written region.
n_upd = math.ceil((N - C) / C)
written = C * n_upd
flips = [C + 5, N // 2]
assert all(f < written for f in flips)

model = load_model(a.ckpt, mc, a.adapter)
for p in model.parameters():
    p.requires_grad_(False)

groups = getattr(mc, 'ttt_share_groups', None) or []
print(f'seq_len {N}, chunk {C}, {n_upd} chunks applied, positions < {written} written')
print(f'share groups: {groups if groups else "none (per-layer memories)"}\n')

torch.manual_seed(0)
ids = torch.randint(0, mc.vocab_size, (1, N), device='cuda')

with torch.no_grad():
    base = model(input_ids=ids, use_cache=False).logits.float()

    def check(flip, label):
        alt = ids.clone()
        alt[0, flip] = (alt[0, flip] + 12345) % mc.vocab_size
        got = model(input_ids=alt, use_cache=False).logits.float()
        # positions strictly before the edit must be untouched
        d = (got[:, :flip] - base[:, :flip]).abs().max().item()
        verdict = 'CAUSAL' if d < a.tol else '*** NON-CAUSAL ***'
        print(f'edit @ {flip:>5} ({label:<17}) -> max |dlogit| at pos < {flip}: '
              f'{d:.3e}   {verdict}')
        return d

    worst = max(check(f, 'written region') for f in flips)
    # Control: the tail chunk is never folded into the state, so this edit is
    # inert by construction. If it ever fires, the harness itself is wrong.
    check(N - 1, 'tail, inert')

raise SystemExit(0 if worst < a.tol else 1)
