#!/usr/bin/env bash
# Run inside a GPU allocation after activating the project's Python environment.
# TTT_INIT: full stage-1 checkpoint; TTT_ADAPTER: optional existing stage-2 adapter.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TTT_INIT:?Set TTT_INIT to the full stage-1 checkpoint directory}"
export TTT_INIT
export TTT_CKPT_DIR="${TTT_CKPT_DIR:-checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH=".:${PYTHONPATH:-}"

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf

cfg = OmegaConf.load('Configs/ttt_ar_hybrid3_full_passkey_mixed.yml')
OmegaConf.resolve(cfg)
base = Path(cfg.model.pretrained_model_name_or_path)
if not (base / 'config.json').is_file():
    raise SystemExit(f'Full stage-1 checkpoint config not found: {base}')
adapter = cfg.train.get('continue_adapter')
if adapter:
    for name in ('adapter_config.json', 'ttt_params.pt'):
        if not (Path(adapter) / name).is_file():
            raise SystemExit(f'Missing continuation file: {Path(adapter) / name}')
out = Path(cfg.train.output_dir) / 'ttt_ar_hybrid3_full_passkey_mixed'
if out.exists() and any(out.iterdir()):
    raise SystemExit(f'Output already contains files; choose a fresh TTT_CKPT_DIR: {out}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU required. Run this script inside your GPU allocation.')
print('GPU:', torch.cuda.get_device_name(0))
print('Stage-1 base:', base)
print('Stage-2 adapter:', adapter or '(fresh stage 2)')
print('Output:', out)
PY

exec "$PYTHON_BIN" run.py --cfg Configs/ttt_ar_hybrid3_full_passkey_mixed.yml
