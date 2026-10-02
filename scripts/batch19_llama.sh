#!/usr/bin/env bash
# Llama batch-19 latency table, 2k-16k. Mirrors scripts/batch19_fill.sh, which
# produced the Mistral table, at the same settings: warmup 2, reps 10, 512
# decode tokens, FLOP reference pass skipped. Cells measured otherwise are not
# comparable to those.
#
# Anchor-F comes from nonunified_v2, the plateau-trained checkpoint, not a
# llama_v3 retrain. Architecture and therefore timing are identical; only the
# quality table cares which weights these are.
#
# Each case is its own process, so an OOM is recorded and the next one still
# runs, and existing reports are skipped -- rerun after a time limit to resume.
set -uo pipefail

CK=${CK:-/work/nvme/bgly/vwilliam/ttt-checkpoints}
export OUT=${OUT:-$CK/efficiency/llama_b19/cases}
PY=${PYTHON_BIN:-python}
COMMON="--batch 19 --warmup 2 --reps 10 --decode-tokens 512 --skip-flops"
mkdir -p "$OUT"

run () {
  local arm=$1 len=$2; shift 2
  local stem="$OUT/${arm}_b19_s${len}"
  if [[ -f "$stem.json" ]]; then echo "SKIP $arm $len (exists)"; return; fi
  echo "START $arm length=$len"
  $PY measure_flops.py $COMMON --seq-len "$len" "$@" --out "$stem.json" > "$stem.log" 2>&1
  python3 -c "
import json
try:
    r=json.load(open('$stem.json')); print(f\"  $arm $len -> {r['status'].upper()}\",
        f\"{r.get('latency_seconds',0)*1e3:.1f} ms\" if r['status']=='ok' else f\"phase={r.get('phase')}\")
except Exception as e: print(f'  $arm $len -> NO REPORT ({e})')"
}

BASE="--baseline --base meta-llama/Llama-3.1-8B"
AI_S1=$CK/llama_v3/ttt_at_anchor_gate/best_ckpt
AI_S2=$CK/llama_v3/ttt_ar_unified_longalpaca/best_ckpt
AF_S1=$CK/nonunified_v2/ttt_at_l2_longalpaca/best_ckpt
AF_S2=$CK/nonunified_v2/ttt_ar_l2_longalpaca/best_ckpt

for len in 2048 4096 8192 16384; do
  run baseline        "$len" $BASE
  run anchor_f        "$len" --cfg Configs/ttt_ar_l2_longalpaca.yml      --ckpt "$AF_S1" --adapter "$AF_S2"
  run anchor_i_full   "$len" --cfg Configs/ttt_ar_unified_longalpaca.yml --ckpt "$AI_S1" --adapter "$AI_S2"
  run anchor_i_deploy "$len" --cfg Configs/ttt_deploy_anchor.yml         --ckpt "$AI_S1" --adapter "$AI_S2"
done

echo; echo "Collecting:"
python3 - <<'PY'
import glob, json, os, re
rows = {}
for p in sorted(glob.glob(os.environ['OUT'] + '/*_b19_s*.json')):
    m = re.match(r'(.+)_b19_s(\d+)\.json', os.path.basename(p))
    if not m: continue
    r = json.load(open(p))
    rows[(m.group(1), int(m.group(2)))] = (
        f"{r['latency_seconds']*1e3:.1f}" if r['status'] == 'ok' else r['status'].upper())
arms = ['baseline', 'anchor_f', 'anchor_i_full', 'anchor_i_deploy']
print(f"{'length':>8} " + ''.join(f'{a:>18}' for a in arms))
for L in (2048, 4096, 8192, 16384):
    print(f'{L:>8} ' + ''.join(f'{rows.get((a, L), "-"):>18}' for a in arms))
PY
