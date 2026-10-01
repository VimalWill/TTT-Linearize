"""Cached inference latency/memory and registered-op FLOPs per arm.

Timing uses the selected runtime backend, after warmup and without a counter.
Counting runs separately, with compilation disabled and chunked window matmuls
enabled for both Llama and Mistral. This reference path exposes TTT updates
(including the vendored Newton-Schulz iterations) and attention to the counter.

The reference window path performs dense matmuls within each chunk and masks
afterwards. Its counted work is not the work of sparse compiled FlexAttention.
PyTorch's SDPA FLOP formula also counts a dense rectangle without adjusting for
causality. Unregistered elementwise ops are excluded. These are registered-op
reference counts, not hardware FLOPs or a backend-independent architectural cost.
The default workload is cached inference: one last-position-logit prefill,
then single-token greedy decode with the returned KV/TTT cache. Each repetition
starts a fresh request. Prefill and decode latency and retained cache storage
are reported separately. Peak allocated memory includes model, inputs, and cache.

    python measure_flops.py --cfg Configs/ttt_deploy_mistral_anchor.yml --ckpt CKPT --seq-len 8192 --reps 10
    python measure_flops.py --baseline --base meta-llama/Llama-3.1-8B --seq-len 8192

Add --sdpa-window only when intentionally timing the window math fallback.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

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


def count_forward(model, ids, depth=0, forward_kwargs=None):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(model, depth=depth, display=depth > 0)
    # Compiled kernels need not expose their internal matmuls to a dispatch
    # counter. Use eager arithmetic for this measurement, including nested
    # compiled calls in the TTT update and Newton-Schulz routines.
    with torch.no_grad(), torch._dynamo.config.patch(disable=True), counter:
        model(input_ids=ids, use_cache=True, **(forward_kwargs or {}))
    return counter.get_total_flops()


def inference_kwargs(model):
    """Support both the pinned transformers 4.45 and newer HF signatures."""
    parameters = inspect.signature(model.forward).parameters
    for name in ('logits_to_keep', 'num_logits_to_keep'):
        if name in parameters:
            return {name: 1, 'return_dict': True}
    raise ValueError('Model must support last-position-only logits for inference benchmarking')


def cache_storage_bytes(cache):
    """Unique backing storage bytes, including storage retained by tensor views."""
    seen, storages = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif hasattr(value, '__dict__'):
            visit(vars(value))
    visit(cache)
    return sum(storages.values())


def checked_cache(output, expected_length):
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError('use_cache=True returned no cache; refusing to benchmark recomputation')
    if output.logits.shape[1] != 1:
        raise RuntimeError('Inference must compute only the last-position logits')
    # TTTCache and HF DynamicCache expose logical length. Rolling HF caches
    # can instead expose the number of retained tokens, so do not equate it
    # with total history here; record the cache class and backing bytes.
    from LinearTTT.model.LinearizeMistral.cache import TTTCache as MistralCache
    from LinearTTT.model.LinearizeLlama.cache import TTTCache as LlamaCache
    if isinstance(cache, (MistralCache, LlamaCache)) and cache.get_seq_length() != expected_length:
        raise RuntimeError('TTT cache did not advance through the request')
    return cache


def cached_request(model, ids, decode_tokens, kwargs, synchronize, on_phase=None):
    """One fresh prefill and a fixed number of cached single-token forwards.

    EOS does not shorten the workload. The prefill's first prediction is fed
    into decode; decode_tokens counts incremental forwards after prefill.
    """
    if on_phase:
        on_phase('prefill')
    synchronize()
    start = time.perf_counter()
    output = model(input_ids=ids, use_cache=True, **kwargs)
    synchronize()
    prefill_seconds = time.perf_counter() - start
    cache = checked_cache(output, ids.shape[1])
    prefill_cache_bytes = cache_storage_bytes(cache)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    del output
    if on_phase:
        on_phase('decode')
    synchronize()
    start = time.perf_counter()
    for step in range(decode_tokens):
        output = model(input_ids=token, past_key_values=cache, use_cache=True, **kwargs)
        cache = checked_cache(output, ids.shape[1] + step + 1)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        del output
    synchronize()
    decode_seconds = time.perf_counter() - start
    return dict(prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                prefill_cache_bytes=prefill_cache_bytes,
                decode_cache_bytes=cache_storage_bytes(cache),
                cache_class=type(cache).__name__)


def count_cached_inference(model, ids, decode_tokens, kwargs, depth=0):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), torch._dynamo.config.patch(disable=True):
        prefill = FlopCounterMode(model, depth=depth, display=depth > 0)
        with prefill:
            output = model(input_ids=ids, use_cache=True, **kwargs)
        cache = checked_cache(output, ids.shape[1])
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        del output
        decode = FlopCounterMode(model, depth=depth, display=depth > 0)
        with decode:
            for step in range(decode_tokens):
                output = model(input_ids=token, past_key_values=cache, use_cache=True, **kwargs)
                cache = checked_cache(output, ids.shape[1] + step + 1)
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                del output
    return prefill.get_total_flops(), decode.get_total_flops()


def is_cuda_oom(error):
    """Recognize allocator OOMs, including errors wrapped by compilation."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        message = str(error).lower()
        if isinstance(error, torch.cuda.OutOfMemoryError) or any(
                marker in message for marker in
                ('cuda out of memory', 'cuda error: out of memory',
                 'cuda_error_out_of_memory')):
            return True
        error = error.__cause__ or error.__context__
    return False


def run_benchmark(a, result):
    """Update a serializable record as each measurement phase completes."""
    import LinearTTT  # noqa: F401
    from Training.train import build_model_config
    result['phase'] = 'model_load'
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
        cfg = load_benchmark_config(a.cfg, a.ckpt, a.seq_len + a.decode_tokens)
        mc = build_model_config(cfg)
        model = load_model(a.ckpt, mc, a.adapter).eval()
        label = (f'{a.cfg}  window={mc.window_size} chunk={mc.lact_chunk_size} '
                 f'layers={mc.ttt_layer_indices} share={mc.ttt_share_groups}')
        runtime_backend = 'chunked window math' if a.sdpa_window else 'compiled FlexAttention'
        count_backend = 'eager chunked window math (includes masked-out matmul work)'

    device = model.device
    result.update(label=label, runtime_backend=runtime_backend,
                  flop_reference=count_backend,
                  decode_backend=(runtime_backend if a.baseline else 'SDPA local attention + TTT memory'),
                  gpu=torch.cuda.get_device_name(device),
                  torch_version=torch.__version__, gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory,
                  window_size=getattr(model.config, 'window_size', None),
                  ttt_layer_indices=getattr(model.config, 'ttt_layer_indices', None),
                  ttt_share_groups=getattr(model.config, 'ttt_share_groups', None))
    result['phase'] = 'inputs'
    generator = torch.Generator(device=device).manual_seed(a.seed)
    ids = torch.randint(model.config.vocab_size, (a.batch, a.seq_len),
                        device=device, generator=generator)

    kwargs = inference_kwargs(model)
    result.update(logits_selection=kwargs, use_cache=True, decode_tokens=a.decode_tokens,
                  baseline_sliding_window=getattr(model.config, 'sliding_window', None))
    sync = lambda: torch.cuda.synchronize(device)
    with torch.no_grad():
        result['phase'] = 'warmup'
        for _ in range(a.warmup):
            cached_request(model, ids, a.decode_tokens, kwargs, sync)
        sync()
        # No request cache survives warmup or a measured repetition.
        resident = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        measurements = []
        def phase(name):
            result['phase'] = name
        for _ in range(a.reps):
            measurements.append(cached_request(model, ids, a.decode_tokens, kwargs, sync, phase))
        peak = torch.cuda.max_memory_allocated(device)
    prefill = sum(m['prefill_seconds'] for m in measurements) / a.reps
    decode = sum(m['decode_seconds'] for m in measurements) / a.reps
    result.update(status='ok', latency_seconds=prefill + decode,
                  prefill_latency_seconds=prefill, decode_latency_seconds=decode,
                  prefill_tokens_per_second=a.batch * a.seq_len / prefill,
                  decode_tokens_per_second=a.batch * a.decode_tokens / decode,
                  decode_seconds_per_step=decode / a.decode_tokens,
                  tokens_per_second=a.batch * (a.seq_len + a.decode_tokens) / (prefill + decode),
                  prefill_cache_bytes=max(m['prefill_cache_bytes'] for m in measurements),
                  decode_cache_bytes=max(m['decode_cache_bytes'] for m in measurements),
                  cache_class=measurements[-1]['cache_class'],
                  peak_allocated_bytes=peak, resident_bytes=resident,
                  above_resident_bytes=peak - resident,
                  counted_flops=None, decode_counted_flops=None, flops_status='skipped')

    # Counting has its own status: a reference-pass OOM must not erase a
    # successful runtime measurement or enter the runtime OOM-boundary search.
    if not a.skip_flops:
        result['phase'] = 'flop_count'
        try:
            if not a.baseline:
                set_window_backend(True)
            result['counted_flops'], result['decode_counted_flops'] = count_cached_inference(
                model, ids, a.decode_tokens, kwargs, a.depth)
            result['flops_status'] = 'ok'
        except Exception as error:
            result['flops_status'] = 'oom' if is_cuda_oom(error) else 'error'
            result['flops_error'] = f'{type(error).__name__}: {error}'
            traceback.print_exc()
        finally:
            if not a.baseline:
                set_window_backend(a.sdpa_window)
    result['phase'] = 'complete'


def print_result(result):
    if result['status'] != 'ok':
        print(f"\n{result['status'].upper()} at {result['phase']}: {result['error']}")
        return
    tok = result['batch'] * result['seq_len']
    print(f"\n{result['label']}")
    print(f"  seq_len {result['seq_len']}  batch {result['batch']}  ({tok:,} tokens)")
    print(f"  scope          cached prefill + {result['decode_tokens']} incremental decode steps, last-position logits")
    print(f"  GPU            {result['gpu']}; torch {result['torch_version']}")
    print(f"  runtime        {result['runtime_backend']}; warmup {result['warmup']}, reps {result['reps']}")
    print(f"  decode backend {result['decode_backend']}")
    print(f"  FLOP reference {result['flop_reference']}")
    total = result['counted_flops']
    if total is not None:
        print(f'  prefill FLOPs  {total/1e12:8.3f} T total   {total/tok/1e9:7.3f} G/token')
        print(f"  decode FLOPs   {result['decode_counted_flops']/1e12:8.3f} T total")
    else:
        print(f"  counted FLOPs  {result['flops_status']}")
    print(f"  prefill        {result['prefill_latency_seconds']*1e3:8.1f} ms          {result['prefill_tokens_per_second']:,.0f} tok/s")
    print(f"  decode         {result['decode_seconds_per_step']*1e3:8.2f} ms/step    {result['decode_tokens_per_second']:,.0f} tok/s")
    print(f"  cache          {result['cache_class']}; prefill {result['prefill_cache_bytes']/2**30:.3f} GiB, decode {result['decode_cache_bytes']/2**30:.3f} GiB")
    print(f"  peak allocated {result['peak_allocated_bytes'] / 2**30:8.2f} GiB (includes model and inputs)")
    print(f"  above resident {result['above_resident_bytes'] / 2**30:8.2f} GiB")
    if result.get('flops_error'):
        print(f"  FLOP error     {result['flops_error']}")
    print('\n  NOTE: reference FLOPs exclude unregistered operations and are not '
          'compiled-kernel FLOPs. They are not divided by runtime latency to '
          'claim achieved hardware TFLOP/s.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--decode-tokens', type=int, default=512,
                    help='cached single-token forwards per request (default: one 512-token TTT chunk)')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--depth', type=int, default=0,
                    help='module-tree depth in the breakdown; 0 = total only')
    ap.add_argument('--sdpa-window', action='store_true',
                    help='time the chunked window math fallback rather than '
                         'compiled FlexAttention; FLOP counting always uses '
                         'the fallback reference path')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--reps', type=int, default=3, help='fresh cached requests after warmup')
    ap.add_argument('--skip-flops', action='store_true',
                    help='measure runtime only; useful for capacity/OOM probes')
    ap.add_argument('--out', help='write a JSON result, including an OOM/error status')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if min(a.seq_len, a.batch, a.reps, a.warmup, a.decode_tokens) < 1 or a.depth < 0:
        ap.error('seq-len, decode-tokens, batch, reps and warmup must be positive; depth must be nonnegative')
    if not a.baseline and (not a.cfg or not a.ckpt):
        ap.error('--cfg and --ckpt are required unless --baseline is supplied')
    if a.baseline and (a.adapter or a.sdpa_window):
        ap.error('--adapter and --sdpa-window apply to the TTT model only')
    if not torch.cuda.is_available():
        ap.error('CUDA is required for the latency and memory benchmark')

    result = dict(status='error', phase='setup', cfg=a.cfg, ckpt=a.ckpt,
                  adapter=a.adapter, baseline=a.baseline, base=a.base,
                  seq_len=a.seq_len, batch=a.batch, warmup=a.warmup,
                  reps=a.reps, seed=a.seed, decode_tokens=a.decode_tokens, use_cache=True,
                  scope='cached_prefill_and_incremental_decode_last_position_logits')
    try:
        run_benchmark(a, result)
    except Exception as error:
        result.update(status='oom' if is_cuda_oom(error) else 'error',
                      error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    print_result(result)
    if a.out:
        path = Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return 0 if result['status'] == 'ok' else 2 if result['status'] == 'oom' else 1


if __name__ == '__main__':
    sys.exit(main())
