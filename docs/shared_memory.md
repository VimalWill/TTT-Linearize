# Shared TTT memory

Each group has one writer: its lowest layer index. The writer computes
`W[c]` using only chunks strictly before chunk `c`. Every follower reads
`W[c]` with its own queries, normalization, and output gate. Followers do not
run inner-loop updates. The SWA branch remains private to each layer.

`Configs/ttt_at_unified.yml` and `Configs/ttt_ar_unified.yml` already select
L2–L27 as a group. Thus L2 writes and L3–L27 read; L0–L1 and L28–L31 retain
private memories. Set `ttt_share_groups: null` for separate memories, or use
multiple disjoint lists to try smaller groups. Group entries must be valid,
unique layer indices with equal fast-weight dimensions. Reordering a list
does not change its writer. Sharing requires `ttt_inner_loss: l2`.

Trajectories are explicit tensor inputs and outputs of decoder layers, so
gradient checkpointing preserves reader-to-writer gradients without global
mutable state. Each forward owns its trajectories. They contain one weight
snapshot per chunk: prefill/training memory is not constant in context length.
Measure peak memory and backward time before making efficiency claims.
Shared models default to non-reentrant checkpointing; requesting reentrant
checkpointing raises an error because it cannot track nested trajectory outputs.

Cached generation returns a `TTTCache` containing each writer's fast weights,
momentum, pending chunk, and each layer's sliding-window KV. Pass that cache
back on subsequent calls. A writer applies a full pending chunk before reading
the next token; followers use the resulting writer state. Cache objects belong
to requests and can be interleaved. Greedy and sampling generation are supported;
beam search, nonempty legacy KV caches, and multi-token continuation of an
existing cache are explicitly unsupported. Start a new request without a cache.

## Validation

Use `transformers==4.45.0`, `peft==0.13.2`, PyTorch, Accelerate, OmegaConf,
einops, pandas, and safetensors. Small CPU tests need no pretrained weights,
datasets, FLA, or CUDA:

```bash
TORCHDYNAMO_DISABLE=1 python -m unittest discover -s tests -v
```

The tests exercise the real small Llama model: future-token invariance, an
acausal final-state negative control, chunk-boundary decode equivalence,
independent requests, greedy generation, checkpointed gradient equivalence,
both training objectives, PEFT/TTT restoration, and sharded checkpoint loading.
CPU tests use eager FP32; they do not validate compiled CUDA numerics or 8B-scale
memory. Run the same suite on the intended CUDA environment with
`TTT_TEST_DEVICE=cuda` (and without `TORCHDYNAMO_DISABLE`) before large runs.
The regression suite currently uses unpadded batches; padded-batch behavior and
multi-device/offloaded execution still require validation.

Check a trained model separately:

```bash
python test_causality.py --cfg Configs/ttt_ar_unified.yml \
  --ckpt /path/to/stage1/best_ckpt --adapter /path/to/stage2/best_ckpt
```

The checkpoint check changes tokens around chunk boundaries and in the tail,
then asserts that earlier logits remain unchanged within tolerance. All probes
affect the exit status. `test_unify_tmp.py` forwards to the same implementation.
`eval.py` runs this tripwire before evaluating a shared model and records the
probe deltas in its output JSON. There is no student-versus-teacher perplexity
ordering assertion: that ordering is not a causality criterion.

## Training and interpreting results

Retrain both stages using new output directories. Checkpoints trained with the
old final-state handoff were optimized under an acausal architecture. Loading
their weights into the new implementation is useful for diagnostics, but does
not turn their historical metrics into valid results for this architecture.

After the CUDA checks, compare retrained intact per-bucket CE and retrieval
scores for separate memories, one large group, and smaller groups, using
matched data and training budgets. Causality does not guarantee that one writer
has enough capacity to serve 25 readers, nor that long-range deficits disappear.
The existing `--ablate ttt --layers ...` diagnostic suppresses a layer's TTT
readout. Ablating the leader's readout still leaves its writes available to
followers, so this diagnostic alone does not measure the writer's importance.
