#!/usr/bin/env bash
# Reduced run: Anchor-F, 40M tokens at 4K/8K; evaluate at 4K/8K/16K/32K.
set -euo pipefail
: "${ANCHOR_F_CKPT:?Set the full/base Llama Anchor-F checkpoint}"
: "${LONG_CONTEXT_DATA:?Set the prepared-data directory}"
: "${LONG_CONTEXT_OUTPUT:?Set a NEW experiment output directory}"
PYTHON_BIN=${PYTHON_BIN:-python}
BASE_TOKENIZER=${BASE_TOKENIZER:-meta-llama/Llama-3.1-8B}
if [[ -e "$LONG_CONTEXT_OUTPUT" ]]; then
  echo "Output already exists: $LONG_CONTEXT_OUTPUT" >&2
  exit 1
fi
mkdir -p "$LONG_CONTEXT_OUTPUT"
export LONG_CONTEXT_DATA
if [[ ! -f "$LONG_CONTEXT_DATA/manifest.json" ]]; then
  "$PYTHON_BIN" prepare_long_context.py --out-dir "$LONG_CONTEXT_DATA" \
    --tokenizer "$BASE_TOKENIZER" --lengths 4096 8192
fi
# Check reused data before loading a large model. Full-recipe data contain 16K
# validation rows and cannot be used to select the reduced-run checkpoint.
"$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
manifest = json.loads((Path(os.environ['LONG_CONTEXT_DATA']) / 'manifest.json').read_text())
if set(manifest['lengths']) != {4096, 8192}:
    raise ValueError('Reduced run requires 4K/8K data; set LONG_CONTEXT_DATA to a separate directory')
PY
export TTT_INIT="$ANCHOR_F_CKPT"
export TTT_ADAPTER="${ANCHOR_F_ADAPTER:-}"
export TTT_CKPT_DIR="$LONG_CONTEXT_OUTPUT/anchor_f"
"$PYTHON_BIN" run.py --cfg Configs/ttt_ar_llama_long_context_anchor_f_reduced.yml \
  2>&1 | tee "$LONG_CONTEXT_OUTPUT/train_anchor_f.log"
F_AFTER="$LONG_CONTEXT_OUTPUT/anchor_f/ttt_ar_llama_long_context_anchor_f_reduced/best_ckpt"
test -f "$F_AFTER/ttt_params.pt"
"$PYTHON_BIN" prepare_long_context.py --niah-from "$LONG_CONTEXT_DATA" \
  --out-dir "$LONG_CONTEXT_OUTPUT/niah_data" --tokenizer "$BASE_TOKENIZER" \
  --lengths 4096 8192 16384 32768 --depths 0 20 40 60 80 100 \
  --samples-per-cell 25 --seed 271828
evaluation_status=0
"$PYTHON_BIN" evaluate_long_context.py --data-dir "$LONG_CONTEXT_DATA" \
  --niah-data "$LONG_CONTEXT_OUTPUT/niah_data/niah.jsonl" \
  --anchor-f-ckpt "$ANCHOR_F_CKPT" --anchor-f-adapter "$F_AFTER" \
  --anchor-f-cfg Configs/ttt_ar_llama_long_context_anchor_f_reduced.yml \
  --arms baseline anchor_f --ruler-limit "${RULER_LIMIT:-500}" \
  --base "$BASE_TOKENIZER" --out-dir "$LONG_CONTEXT_OUTPUT/evaluation" || evaluation_status=$?
"$PYTHON_BIN" plot_long_context.py --results "$LONG_CONTEXT_OUTPUT/evaluation"

exit "$evaluation_status"
