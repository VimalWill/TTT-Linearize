"""Measured FLOPs, latency and peak memory per arm -- not a hand count.

The hand estimate for this architecture has one soft spot: Newton-Schulz
dominates the TTT branch (~94 of 178 MMACs per chunk per head by hand) but its
iteration count lives in LaCT's upstream `zeropower_via_newtonschulz5`, which
is not vendored here. FlopCounterMode counts what actually dispatches, so it
settles that without assuming.

Caveat worth reading before quoting the number: FlopCounterMode is a
TorchDispatchMode and sees aten ops. flex_attention is a higher-order op whose
compiled Triton kernel it does NOT see, so the sliding-window branch is
undercounted. Run with --sdpa-window to route attention through the chunked
SDPA path instead, which does dispatch countable matmuls. The TTT branch is
plain bmm either way and is always counted.

    python measure_flops.py --cfg Configs/ttt_deploy_anchor.yml --ckpt CKPT --seq-len 8192 --sdpa-window
    python measure_flops.py --baseline --base meta-llama/Llama-3.1-8B --seq-len 8192
"""
import argparse
import time

import torch


def main():
    from omegaconf import OmegaConf
    from torch.utils.flop_counter import FlopCounterMode
    import LinearTTT  # noqa: F401
    from Training.train import build_model_config

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
                    help='route SWA through chunked SDPA so its matmuls are '
                         'counted; flex_attention is a HOP and is invisible to '
                         'the counter')
    ap.add_argument('--reps', type=int, default=3, help='timed forwards after warmup')
    a = ap.parse_args()

    if a.baseline:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, device_map={'': 0}).eval()
        label = f'baseline {a.base}'
    else:
        from eval import load_model
        from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window
        if a.sdpa_window:
            use_sdpa_sliding_window(True)
        cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(a.cfg), resolve=True))
        cfg.model.pretrained_model_name_or_path = a.ckpt
        cfg.model.max_length = max(a.seq_len, int(cfg.model.max_length))
        mc = build_model_config(cfg)
        model = load_model(a.ckpt, mc, a.adapter).eval()
        label = (f'{a.cfg}  window={mc.window_size} chunk={mc.lact_chunk_size} '
                 f'layers={mc.ttt_layer_indices} share={mc.ttt_share_groups}')

    ids = torch.randint(0, 1000, (a.batch, a.seq_len), device=model.device)

    with torch.no_grad():
        counter = FlopCounterMode(model, depth=a.depth, display=a.depth > 0)
        with counter:
            model(input_ids=ids, use_cache=False)
        total = counter.get_total_flops()

        for _ in range(2):                       # warmup
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / a.reps
        peak = torch.cuda.max_memory_allocated() / 2**30

    tok = a.batch * a.seq_len
    print(f'\n{label}')
    print(f'  seq_len {a.seq_len}  batch {a.batch}  ({tok:,} tokens)')
    print(f'  FLOPs          {total/1e12:8.3f} T total   {total/tok/1e9:7.3f} G/token')
    print(f'  latency        {dt*1e3:8.1f} ms          {tok/dt:,.0f} tok/s')
    print(f'  achieved       {total/dt/1e12:8.2f} TFLOP/s')
    print(f'  peak memory    {peak:8.2f} GiB')
    if not a.baseline and not a.sdpa_window:
        print('\n  NOTE: flex_attention is not counted. Pass --sdpa-window for a '
              'FLOP number that includes the sliding-window branch.')


if __name__ == '__main__':
    main()
