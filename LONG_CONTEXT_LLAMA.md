# Llama long-context continuation

This pipeline continues trained Llama-3.1-8B Anchor-I and Anchor-F checkpoints,
trains through 16K, then evaluates at 4K/8K/16K/32K. The fourth evaluation arm,
`anchor`, is the two-memory deployment of the continued Anchor-I checkpoint.
It is evaluated separately; pruning does not imply retained retrieval quality.

## Run on the GPU host

From the repository root, use the same environment as the existing training
and `eval.py` jobs. `lm_eval`, `datasets`, and `matplotlib` are needed for the
final evaluations and figures. Keep the harness installation fixed throughout
an experiment; the evaluation summary records installed package versions.

```bash
export ANCHOR_I_CKPT=/path/to/llama_anchor_i/full_base_checkpoint
export ANCHOR_I_ADAPTER=/path/to/llama_anchor_i/stage2/best_ckpt
export ANCHOR_F_CKPT=/path/to/llama_anchor_f/full_base_checkpoint
export ANCHOR_F_ADAPTER=/path/to/llama_anchor_f/stage2/best_ckpt
export LONG_CONTEXT_DATA=/work/nvme/bgly/vwilliam/long_context_data_v1
export LONG_CONTEXT_OUTPUT=/work/nvme/bgly/vwilliam/long_context_llama_v1
bash scripts/long_context_llama.sh
```

`LONG_CONTEXT_OUTPUT` must be new. The data directory is generated if it does
not contain `manifest.json`; otherwise both training runs reuse it. The full/base
checkpoint and its stage-2 adapter are distinct inputs. Omit an adapter variable
only when its trained adapter has already been merged into that full checkpoint.
Do not use the original pretrained Llama weights as an Anchor checkpoint.

The script prepares data, trains both arms sequentially, constructs the final
NIAH grid, evaluates all arms and ablations, and exports PNG/PDF figures and CSV
results. Evaluation failures do not stop later cases; plots are still generated
from completed cases, and the script exits nonzero if any case failed. Inspect
`evaluation/summary.json` and the individual logs rather than treating missing
cases as zero scores.

## Data and training recipe

The default source is `Yukang/LongAlpaca-12k`. Documents are deduplicated and split
80/10/10 before generating any examples. Synthetic examples use filler only from
their own split. Source instruction answers provide the instruction/QA component.
A local JSONL source can be used instead, with `text`, `prompt`, and `answer`.
There must be at least 30 distinct documents and usable instruction examples.

```bash
python prepare_long_context.py \
  --out-dir "$LONG_CONTEXT_DATA" \
  --tokenizer meta-llama/Llama-3.1-8B \
  --source-jsonl /path/to/source_documents.jsonl
```

The prepared files contain intact prompt/answer records, actual token counts,
task/length metadata, document identifiers, and a tokenizer fingerprint. Overlength
examples raise rather than lose a needle or answer through truncation. The training
loader preserves variable lengths at microbatch 1 and does not pack haystacks together.

Sampling probabilities are per example, not proportions of input tokens:

| Task | Probability |
|---|---:|
| Single numeric value | 15% |
| Single UUID value | 15% |
| Multiple keys / distractors | 10% |
| Multiple queried keys | 10% |
| Multiple values for one key | 10% |
| Four-hop register tracing | 10% |
| Frequency aggregation | 10% |
| Source instruction / QA | 20% |

Multiple-key/query/value examples vary numeric and UUID values. Aggregation varies
the number of requested labels. Synthetic keys/values are generated independently
of RULER, and reserved across the training, validation, and internal test splits.
80% of synthetic examples explicitly require every inserted fact to be at least
768 tokens before the answer prompt; remaining examples sample a broader depth range.
This rules out a direct last-window shortcut, but not relay through stacked local
attention layers. Final memory ablations are still needed.

Both arms use the same sampler seed, data, and token schedule:

- First 20M successful-update input tokens: uniformly sampled eligible 4K/8K
  buckets within each synthetic task.
- Remaining 80M: uniformly sampled eligible 4K/8K/16K buckets.
- Total budget: 100M actual input tokens, including answers. The final optimizer
  update may overshoot by one accumulation window. Token proportions differ from
  example proportions; `training_progress.json` records actual task/length exposure.
- Microbatch 1, accumulation 8; window/chunk 512.
- Outer learning rate warms up from zero to `1e-4` over 3M input tokens, then
  decays to `1e-5` at the 100M-token boundary.
- Continue existing Q/K/V LoRA and train the TTT branch, including reader maps.
  Reader optimizer parameters remain FP32. Each arm retains its own sharing layout.
- Answer-only loss. The LM head computes only the response-prediction span;
  the transformer and TTT still process the complete prompt.
- Training cache is disabled for backpropagation. Generation validation and final
  inference explicitly enable the per-request KV/TTT cache.

The new configs stop on backward errors or nonfinite losses/gradients. They do
not silently finish a nominal matched-budget run after skipping broken updates.
An epoch/step limit reached before the token budget raises. To change the budget,
edit **both** long-context configs consistently, including warmup and transition.

## Checkpoint selection

Every 100 optimizer updates, evaluate held-out answer loss plus greedy generated
answer accuracy on four validation examples per synthetic task/length cell.
`eval/instruction_loss_ce` separately tracks instruction/QA answer loss.
`eval/retrieval_accuracy` is the equal-weight mean across cells, and higher is better.
Per-length accuracies are also recorded. This is exact answer matching after case,
whitespace, and comma-spacing normalization; explanatory text is not accepted.
Final NIAH and RULER test examples never select checkpoints.
The run directory contains `eval_metrics.csv` and `training_progress.json`.

The final update is validated even when it falls between evaluation intervals.
`best_ckpt` is selected by retrieval accuracy; `last_ckpt` retains the final trained
state. `best` is the restored selected model. Initial accuracy zero is eligible
for saving, so failure to improve cannot erase the starting checkpoint.

## Evaluation protocol and outputs

The final NIAH grid uses independent numeric and UUID values and held-out source
documents. Default settings: 4K/8K/16K/32K prompt lengths, 11 insertion depths
(0% through 100%), and 50 examples per cell/type. That is 4,400 examples per arm
or ablation. Prompts are sized to the target with at most 16 tokens of slack,
and the actual length/distance are retained in the samples.

RULER runs all 13 tasks at each length, up to 500 examples per task. Task-provided
prompts, few-shot defaults, generation settings, and scoring are retained. Each
length is evaluated separately with generation headroom. Requested-length metrics
must exist for all 13 tasks, and `-1` sentinel metrics are never included in averages.

Evaluate baseline Llama, continued Anchor-F, continued Anchor-I, and the two-memory
Anchor deployment. Each memory arm also runs with the TTT branch ablated, using
identical prompts. Accuracy uses the same reference window backend as `eval.py`;
efficiency remains a separate measurement using `sweep_efficiency.py`.

Figures and tables under `evaluation/plots/`:

- NIAH heatmaps per arm/type, including memory-off variants, as PNG and PDF.
- `niah_cells.csv`: exact-match accuracy, sample counts, Wilson 95% intervals.
- `paired_memory_gains.csv`: memory-on minus memory-off exact-match accuracy,
  with paired bootstrap 95% intervals, aggregated across depths per type/length.
- `ruler_scores.csv`: individual task scores and equal-task macro averages,
  represented as fractions; the plotted RULER curve uses percentages.
- RULER score-versus-length curve, as PNG and PDF.

OOMs/failed or incomplete lengths are missing data in figures. They are not
scored as wrong answers, or silently included in a partial RULER aggregate.
Full defaults are a substantial evaluation workload (including all ablations).
First check the commands with a small grid and the existing trained checkpoints:

```bash
python prepare_long_context.py --niah-from "$LONG_CONTEXT_DATA" \
  --out-dir /path/to/new_smoke_niah --lengths 4096 \
  --depths 0 50 100 --samples-per-cell 1 --seed 271828
python evaluate_long_context.py --data-dir "$LONG_CONTEXT_DATA" \
  --niah-data /path/to/new_smoke_niah/niah.jsonl \
  --anchor-i-ckpt "$ANCHOR_I_CKPT" --anchor-i-adapter "$ANCHOR_I_ADAPTER" \
  --anchor-f-ckpt "$ANCHOR_F_CKPT" --anchor-f-adapter "$ANCHOR_F_ADAPTER" \
  --out-dir /path/to/new_smoke_results --lengths 4096 \
  --ruler-limit 2 --no-ablation
python plot_long_context.py --results /path/to/new_smoke_results
```

A smoke result is a wiring check, not a final quality result. These files implement
the experiment; they do not establish that exact retrieval or 32K extrapolation
will succeed. Those claims require the generated-answer and ablation results.
