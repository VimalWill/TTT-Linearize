# Context scaling and a measured OOM boundary

Run on the GH200 in the environment used for `measure_flops.py`, from the
repository root. The sweep compares the teacher, Anchor-I full, Anchor-I deploy,
and Anchor-F at **4096, 8192, 16384, and 32768 tokens per sequence**.

```bash
python sweep_efficiency.py \
  --base mistralai/Mistral-7B-v0.1 \
  --anchor-i-ckpt /path/to/anchor_i/full_base_checkpoint \
  --anchor-i-adapter /path/to/anchor_i/stage2/best_ckpt \
  --anchor-f-ckpt /path/to/anchor_f/full_base_checkpoint \
  --anchor-f-adapter /path/to/anchor_f/stage2/best_ckpt \
  --out-dir /work/nvme/bgly/vwilliam/efficiency/context_sweep_01 \
  --reps 10
```

Supply the exact full/base checkpoint used to train each stage-2 adapter. Both
Anchor-I variants load the same checkpoint and adapter; their configs select the
full architecture (28 memories) or deployment architecture (2 memories). Anchor-F
uses its independently trained checkpoint and `Configs/ttt_ar_mistral_l2.yml`
(32 separate memories). If a model was exported as a full merged checkpoint,
omit its adapter argument. All config paths are overrideable in `--help`.
The output directory must be new or empty. The sweep reads existing checkpoints;
it does not export or copy model weights.

## Two comparisons

**Batch 1:** all four arms at all four lengths, retaining the earlier benchmark
scope. These cases measure latency, throughput, allocated memory, and reference
FLOPs. Use `--skip-flops` for a runtime-only sweep.

**Fixed stress batch:** the same four arms and lengths at one shared batch size,
chosen by baseline calibration. The runner probes the teacher at 16K, doubles the
batch up to `--max-batch 128`, then searches for the smallest batch that OOMs at
16K. It verifies that batch fits at 8K. Every arm uses this exact batch across all
lengths; OOM cases are recorded, with no automatic per-case batch reduction.
Stress cases skip the separate FLOP reference pass.

If calibration cannot verify the requested condition, it records
`target_met: false` and explains why. It still runs the matrix at the reported
fallback batch. A stronger GPU, available-memory changes, or the batch-search
limit can prevent finding the bracket; an OOM between 8K and 16K cannot be
guaranteed in advance. Set `--max-batch 256` to widen a search that found no OOM.
Use `--batch N` to test a specific stress batch instead of automatic calibration;
the runner still checks its baseline OOM boundary.

Calibration measures the real selected backend on the actual GPU. It does not
reduce the allocator limit or disable the teacher's efficient attention backend
to produce a failure. Run with the GPU otherwise idle; concurrent jobs can move
the observed boundary. Treat the chosen stress batch as a capacity experiment,
rather than a tuned serving batch or an unbiased prediction of OOM probability.

## Where the teacher runs out of memory

After batch selection, baseline-only probes refine the sequence length between
8192 (fits) and 16384 (OOM). `summary.json` records:

- `max_tested_fit_tokens`: a measured fitting sequence length;
- `min_tested_oom_tokens`: a measured failing sequence length;
- the batch size and the corresponding total batch token counts;
- `interval_width_tokens`, at most `--oom-resolution 256` by default.

This is a measured bracket under a monotonic-capacity assumption. It is not an
exact token threshold unless adjacent token counts were tested. Set
`--oom-resolution 1` for that refinement; this requires more fresh model loads.
GPU allocator/compilation state and available memory can still affect repeated
measurements. The recorded phase identifies whether inputs, warmup, or timed
execution failed. A checkpoint-load OOM does not qualify as a token boundary.

## Failure isolation and results

Each case launches `measure_flops.py` in a **new sequential subprocess**. The
process exits after a CUDA OOM, releasing its CUDA context before the next config
starts. No GPU cases run concurrently. Calibration also uses separate processes,
so model-loading time adds to sweep duration but never enters reported forward
latency. Each probe uses the configured warmup count and one timed forward; matrix
cases use the configured repetition count.

The worker writes structured JSON with `status: ok`, `oom`, or `error`, plus the
phase and message. A FLOP reference-pass OOM has its own `flops_status`; a valid
runtime measurement survives. Unrelated errors and external process termination
remain errors instead of being silently classified as CUDA OOMs. The sweep
continues through subsequent cases, then exits nonzero if an ordinary error was
recorded. Expected OOMs do not fail the completed sweep.

Outputs are saved incrementally:

- `summary.json`: configuration, all calibration probes, OOM bracket, and results;
- `results.csv`: matrix results, memory, runtime status and baseline speedups;
- `cases/*.json` and `cases/*.log`: individual result records and complete logs;
- `probes/*.json` and `probes/*.log`: calibration and boundary-search records.

Speedup uses teacher latency divided by arm latency **at the same batch and
length**, only when both runs succeed. If the teacher OOMs while an anchor fits,
record that capacity advantage; speedup is left empty rather than assigned
infinity. All measured cases use the same random-input seed and selected
checkpoint vocabularies; provide matching Mistral models for this comparison.

## Measurement scope

These remain **full forwards with logits at all input positions, cache disabled**.
Peak allocated bytes include model weights, inputs, activations, logits, and any
live model state. They exclude allocator-reserved unused memory and allocations
outside PyTorch's CUDA allocator. An OOM can be caused by logits or temporary
activations; it should not be described as KV-cache exhaustion without profiling.
This sweep establishes context scaling for this workload. Cached prefill/decode
and concurrent serving need their own comparison, and pruning requires separate
quality measurements.

Reference FLOPs count the eager chunked attention implementation, including its
masked-out matmul work, and exclude unregistered operations. They are not divided
by compiled runtime latency to claim hardware TFLOP/s. Runtime measurements use
the teacher's existing attention implementation and compiled FlexAttention for
the TTT arms.

CPU regression checks:

```bash
python3 -m unittest discover -s tests -v
```

The tests verify subprocess continuation after OOM, structured error reporting,
batch and token-boundary searches, and preservation of runtime results when only
the FLOP counter OOMs. Actual GH200 capacity and timings must be measured there.
