"""Forward latency/memory and registered-op FLOPs per arm.

Timing uses the selected runtime backend, after warmup and without a counter.
Counting runs separately, with compilation disabled and chunked window matmuls
enabled for both Llama and Mistral. This reference path exposes TTT updates
(including the vendored Newton-Schulz iterations) and attention to the counter.

The reference window path performs dense matmuls within each chunk and masks
afterwards. Its counted work is not the work of sparse compiled FlexAttention.
PyTorch's SDPA FLOP formula also counts a dense rectangle without adjusting for
causality. Unregistered elementwise ops are excluded. These are registered-op
reference counts, not hardware FLOPs or a backend-independent architectural cost.
Throughput below measures a full forward with logits at every input position;
use_cache=False means neither cached prefill state nor incremental decode is
benchmarked. Peak allocated memory includes the model and inputs.

    python measure_flops.py --cfg Configs/ttt_deploy_anchor.yml --ckpt CKPT --seq-len 8192 --reps 10
    python measure_flops.py --baseline --base meta-llama/Llama-3.1-8B --seq-len 8192

Add --sdpa-window only when intentionally timing the window math fallback.
"""
import argparse
import time

import torch


def set_window_backend(enable):
    from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window
    from LinearTTT.model.LinearizeMistral.LinearizeMistral import (
        use_sdpa_sliding_window as use_mistral_sdpa_sliding_window,
    )
    use_sdpa_sliding_window(enable)
    use_mistral_sdpa_sliding_window(enable)


def load_benchmark_config(path, checkpoint, seq_len):
    from omegaconf import OmegaConf
    config = OmegaConf.load(path)
    # Apply --ckpt before resolving a required ${oc.env:TTT_INIT} interpolation.
    config.model.pretrained_model_name_or_path = checkpoint
    config.model.max_length = max(seq_len, int(config.model.max_length))
    return OmegaConf.create(OmegaConf.to_container(config, resolve=True))


def count_forward(model, ids, depth=0):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(model, depth=depth, display=depth > 0)
    # Compiled kernels need not expose their internal matmuls to a dispatch
    # counter. Use eager arithmetic for this measurement, including nested
    # compiled calls in the TTT update and Newton-Schulz routines.
    with torch.no_grad(), torch._dynamo.config.patch(disable=True), counter:
        model(input_ids=ids, use_cache=False)
    return counter.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--depth', type=int, default=0,
                    help='module-tree depth in the breakdown; 0 = total only')
    ap.add_argument('--sdpa-window', action='store_true',
                    help='time the chunked window math fallback rather than '
                         'compiled FlexAttention; FLOP counting always uses '
                         'the fallback reference path')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--reps', type=int, default=3, help='timed forwards after warmup')
    a = ap.parse_args()
    if min(a.seq_len, a.batch, a.reps, a.warmup) < 1 or a.depth < 0:
        ap.error('seq-len, batch, reps and warmup must be positive; depth must be nonnegative')
    if not a.baseline and (not a.cfg or not a.ckpt):
        ap.error('--cfg and --ckpt are required unless --baseline is supplied')
    if a.baseline and (a.adapter or a.sdpa_window):
        ap.error('--adapter and --sdpa-window apply to the TTT model only')
    if not torch.cuda.is_available():
        ap.error('CUDA is required for the latency and memory benchmark')

    import LinearTTT  # noqa: F401
    from Training.train import build_model_config

    if a.baseline:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, device_map={'': 0}).eval()
        label = f'baseline {a.base}'
        runtime_backend = f'teacher {model.config._attn_implementation}'
        count_backend = 'eager teacher attention (dense SDPA counting convention)'
    else:
        from eval import load_model
        set_window_backend(a.sdpa_window)
        cfg = load_benchmark_config(a.cfg, a.ckpt, a.seq_len)
        mc = build_model_config(cfg)
        model = load_model(a.ckpt, mc, a.adapter).eval()
        label = (f'{a.cfg}  window={mc.window_size} chunk={mc.lact_chunk_size} '
                 f'layers={mc.ttt_layer_indices} share={mc.ttt_share_groups}')
        runtime_backend = 'chunked window math' if a.sdpa_window else 'compiled FlexAttention'
        count_backend = 'eager chunked window math (includes masked-out matmul work)'

    ids = torch.randint(model.config.vocab_size, (a.batch, a.seq_len), device=model.device)
    device = model.device

    with torch.no_grad():
        for _ in range(a.warmup):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize(device)
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(a.reps):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / a.reps
        peak = torch.cuda.max_memory_allocated(device)

    # The reference counter runs AFTER timing and its allocations never enter
    # the recorded runtime peak. Restore the selected backend even on failure.
    try:
        if not a.baseline:
            set_window_backend(True)
        total = count_forward(model, ids, a.depth)
    finally:
        if not a.baseline:
            set_window_backend(a.sdpa_window)

    tok = a.batch * a.seq_len
    print(f'\n{label}')
    print(f'  seq_len {a.seq_len}  batch {a.batch}  ({tok:,} tokens)')
    print('  scope          full forward with all-position logits, use_cache=False')
    print(f'  GPU            {torch.cuda.get_device_name(device)}; torch {torch.__version__}')
    print(f'  runtime        {runtime_backend}; warmup {a.warmup}, reps {a.reps}')
    print(f'  FLOP reference {count_backend}')
    print(f'  counted FLOPs  {total/1e12:8.3f} T total   {total/tok/1e9:7.3f} G/token')
    print(f'  latency        {dt*1e3:8.1f} ms          {tok/dt:,.0f} tok/s')
    print(f'  peak allocated {peak / 2**30:8.2f} GiB (includes model and inputs)')
    print(f'  above resident {(peak - resident) / 2**30:8.2f} GiB')
    print('\n  NOTE: reference FLOPs exclude unregistered operations and are not '
          'compiled-kernel FLOPs. They are not divided by runtime latency to '
          'claim achieved hardware TFLOP/s.')


if __name__ == '__main__':
    main()
