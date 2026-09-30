#!/usr/bin/env bash
# Before/after passkey AR fine-tuning, with memory on/off at the same 2k length.
# Run from the repository root. Required: ANCHOR_CKPT, BASE_TOKENIZER,
# TTT_CKPT_DIR (a new directory). Optional: ANCHOR_ADAPTER, RULER_LIMIT.
set -euo pipefail
: "${ANCHOR_CKPT:?Set the trained anchor full/base checkpoint path}"
: "${BASE_TOKENIZER:?Set the matching Llama or Mistral tokenizer name/path}"
: "${TTT_CKPT_DIR:?Set a fresh output directory for this experiment}"

CFG=Configs/ttt_ar_passkey_2048.yml
PYTHON_BIN=${PYTHON_BIN:-python}
export TTT_INIT="$ANCHOR_CKPT"
export TTT_ADAPTER="${ANCHOR_ADAPTER:-}"
export TTT_CKPT_DIR
if [[ -e "$TTT_CKPT_DIR" ]]; then
  echo "Output directory already exists: $TTT_CKPT_DIR" >&2
  exit 1
fi
mkdir -p "$TTT_CKPT_DIR/eval"

common=(--cfg "$CFG" --ckpt "$ANCHOR_CKPT" --base "$BASE_TOKENIZER"
        --tasks niah_single_1 niah_single_2 niah_single_3
        --ruler-lengths 2048 --seq-len 2560 --num-fewshot 0
        --batch-size 1 --seed 0 --log-samples)
if [[ -n "${RULER_LIMIT:-}" ]]; then
  common+=(--limit "$RULER_LIMIT")
fi
before=()
if [[ -n "$TTT_ADAPTER" ]]; then
  before=(--adapter "$TTT_ADAPTER")
fi

"$PYTHON_BIN" eval.py "${common[@]}" "${before[@]}" --out "$TTT_CKPT_DIR/eval/before"
"$PYTHON_BIN" eval.py "${common[@]}" "${before[@]}" --ablate ttt --out "$TTT_CKPT_DIR/eval/before"

"$PYTHON_BIN" run.py --cfg "$CFG"
AFTER="$TTT_CKPT_DIR/ttt_ar_passkey_2048/best_ckpt"
test -f "$AFTER/adapter_config.json"
test -f "$AFTER/ttt_params.pt"

"$PYTHON_BIN" eval.py "${common[@]}" --adapter "$AFTER" --out "$TTT_CKPT_DIR/eval/after"
"$PYTHON_BIN" eval.py "${common[@]}" --adapter "$AFTER" --ablate ttt --out "$TTT_CKPT_DIR/eval/after"
echo "Results and per-example predictions: $TTT_CKPT_DIR/eval"
