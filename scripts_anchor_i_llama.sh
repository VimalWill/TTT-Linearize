#!/usr/bin/env bash
# Llama Anchor-I: stage 1 -> stage 2 -> commonsense / recall / MMLU.
#
# Anchor-I is the SHARED checkpoint kept at all layers: ttt_share_groups
# [[0,1],[28,29,30,31]] gives 28 memories (176 MB), and ttt_layer_indices stays
# null so nothing is dropped. The same stage-2 checkpoint also produces the
# Anchor row, by re-evaluating it under ttt_deploy_anchor.yml (2 memories,
# 12.6 MB) -- no extra training. Anchor-F is the separate-memory arm and is a
# different pipeline (ttt_{at,ar}_l2_longalpaca).
#
# Writes to llama_v3, NOT the anchor_v2 tree. anchor_v2 backs the current table
# and has to survive until these numbers land. This re-run exists because every
# shared Llama arm so far trained with ReaderOutputAlignment frozen at identity
# (bf16 ULP near 1.0 is 2^-7, an AdamW step at 2e-4 rounds away) and with the
# LR pinned at peak (the trainer's real scheduler was ReduceLROnPlateau).
set -euo pipefail

ROOT=/work/nvme/bgly/vwilliam/ttt-checkpoints
export TTT_CKPT_DIR=$ROOT/llama_v3
BASE=meta-llama/Llama-3.1-8B
S1=$TTT_CKPT_DIR/ttt_at_anchor_gate/best_ckpt
S2=$TTT_CKPT_DIR/ttt_ar_unified_longalpaca/best_ckpt

echo "=== stage 1: attention transfer (~7.5h) ==="
python run.py --cfg Configs/ttt_at_anchor_gate.yml
test -d "$S1" || { echo "stage 1 produced no best_ckpt at $S1"; exit 1; }

echo "=== stage 2: AR finetune (~6h) ==="
python run.py --cfg Configs/ttt_ar_unified_longalpaca.yml
test -d "$S2" || { echo "stage 2 produced no best_ckpt at $S2"; exit 1; }

# Anchor-I: all 28 memories retained (cfg has ttt_layer_indices: null).
for suite in commonsense recall; do
  echo "=== Anchor-I $suite ==="
  python eval.py --cfg Configs/ttt_ar_unified_longalpaca.yml \
    --ckpt "$S1" --adapter "$S2" --base "$BASE" \
    --suite $suite --batch-size auto --out "$ROOT/v3_anchorI_$suite"
done
echo "=== Anchor-I mmlu (5-shot) ==="
python eval.py --cfg Configs/ttt_ar_unified_longalpaca.yml \
  --ckpt "$S1" --adapter "$S2" --base "$BASE" \
  --suite mmlu --num-fewshot 5 --batch-size auto --out "$ROOT/v3_anchorI_mmlu"

echo
echo "Anchor-I done. The Anchor row (12.6 MB) reuses this same checkpoint:"
echo "  python eval.py --cfg Configs/ttt_deploy_anchor.yml \\"
echo "    --ckpt $S1 --adapter $S2 --base $BASE \\"
echo "    --suite commonsense recall --batch-size auto --out $ROOT/v3_anchor_cs_recall"
