#!/usr/bin/env bash
# Fill the 11 missing batch-19 cells at settings matched to the 5 already on
# disk: --warmup 2 --reps 10 --decode-tokens 512 --skip-flops.
#
# Matching matters. The sweep ran batch-19 cases with reps 10 and 512 decode
# tokens and skipped the FLOP reference pass (skip_flops = batch != 1). Cells
# measured with fewer reps or a shorter decode are not comparable and must not
# share a table with them.
#
# Already measured, do not rerun:
#   4096  baseline 53.1s  anchor_f 43.9s  anchor_i_full 42.0s  anchor_i_deploy 20.2s
#   8192  baseline 100.3s
#   16384 baseline OOM at batch 19 (probe: fits 15616, OOMs at 15872)
#
# Each case is its own process, so an OOM is recorded and the next one still
# runs. Results land beside the sweep's own cases.
set -uo pipefail

CK=${CK:-/work/nvme/bgly/vwilliam/ttt-checkpoints}
export OUT=${OUT:-$CK/efficiency/sweep_01/cases}
PY=${PYTHON_BIN:-python}
COMMON="--batch 19 --warmup 2 --reps 10 --decode-tokens 512 --skip-flops"
mkdir -p "$OUT"

run () {  # run <arm> <length> <cfg-or-baseline...> 
  local arm=$1 len=$2; shift 2
  local stem="$OUT/${arm}_b19_s${len}"
  if [[ -f "$stem.json" ]]; then echo "SKIP $arm $len (exists)"; return; fi
  echo "START $arm length=$len"
  $PY measure_flops.py $COMMON --seq-len "$len" "$@" \
      --out "$stem.json" > "$stem.log" 2>&1
  python3 -c "
import json,sys
try:
    r=json.load(open('$stem.json')); print(f\"  $arm $len -> {r['status'].upper()}\",
        f\"{r.get('latency_seconds',0)*1e3:.1f} ms\" if r['status']=='ok' else f\"phase={r.get('phase')}\")
except Exception as e: print(f'  $arm $len -> NO REPORT ({e})')"
}

BASE="--baseline --base mistralai/Mistral-7B-v0.1"
AI_S1=$CK/mistral_fixed/ttt_at_mistral_anchor/best_ckpt
AI_S2=$CK/mistral_fixed/ttt_ar_mistral_anchor/best_ckpt
AF_S1=$CK/ttt_at_mistral_l2/best_ckpt
AF_S2=$CK/ttt_ar_mistral_l2/best_ckpt

for len in 8192 16384 32768; do
  run baseline        "$len" $BASE
  run anchor_f        "$len" --cfg Configs/ttt_ar_mistral_l2.yml      --ckpt "$AF_S1" --adapter "$AF_S2"
  run anchor_i_full   "$len" --cfg Configs/ttt_ar_mistral_anchor.yml  --ckpt "$AI_S1" --adapter "$AI_S2"
  run anchor_i_deploy "$len" --cfg Configs/ttt_deploy_mistral_anchor.yml --ckpt "$AI_S1" --adapter "$AI_S2"
done
echo; echo "Collecting:"
python3 - <<'PY'
import glob, json, os, re
rows = {}
for p in sorted(glob.glob(os.environ.get('OUT', '') + '/*_b19_s*.json')):
    m = re.match(r'(.+)_b19_s(\d+)\.json', os.path.basename(p))
    if not m: continue
    r = json.load(open(p))
    rows[(m.group(1), int(m.group(2)))] = (
        f"{r['latency_seconds']*1e3:.1f}" if r['status'] == 'ok' else r['status'].upper())
arms = ['baseline', 'anchor_f', 'anchor_i_full', 'anchor_i_deploy']
print(f"{'length':>8} " + ''.join(f'{a:>18}' for a in arms))
for L in (4096, 8192, 16384, 32768):
    print(f'{L:>8} ' + ''.join(f'{rows.get((a, L), "-"):>18}' for a in arms))
PY
