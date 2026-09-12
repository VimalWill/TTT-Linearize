# Low-rank estimation of TTT memory states

Status: full-state, applied-write and accumulated-adaptation spectra are
implemented in `diag_rank.py`. Raw-gradient/momentum spectra and compression
interventions below remain follow-up experiments. Checkpoint weights are
unchanged by observation.

## Running the first pilot

From the repository root on the GPU node, using the fixed fresh checkpoint pair:

```sh
python diag_rank.py \
  --cfg configs/ttt_ar_unified.yml \
  --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_at_unified/best_ckpt \
  --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified/best_ckpt \
  --corpora pg19 wikitext --seq-len 8192 --seqs 8 \
  --chunks 0 1 2 4 8 15 \
  --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_rank_8192
```

Use `Configs/` if that is the directory's capitalization on the cluster.
Copy both `diag_rank.py` and the updated
`LinearTTT/model/LinearizeLlama/LinearizeLlama.py` to the cluster. The model change
adds an optional inference observer, disabled during ordinary training/evaluation.
The probe captures actual operator trajectories, then computes SVD outside the
model forward. It skips the language-model head and never loads the training
dataloader or its optional metrics.

Outputs in the new directory:

- `metadata.json`: checkpoint/config identities, token hashes, observed layers,
  and a `complete` flag written only after the run finishes.
- `tokens.pt`: reusable exact input sequences and source metadata. Use
  `--tokens-file .../tokens.pt` for matched future probes with the same `--seqs`,
  `--seq-len` and tokenizer.
- `spectra.jsonl`: singular values and statistics for every sequence, memory,
  head, matrix and sampled chunk.
- `ranks.csv`: the same measurements without singular-value arrays.
- `summary.json`: descriptive per-layer/chunk means, medians, p90 and maxima.
- `pg19_r95/r99` and `wikitext_r95/r99` PNG/PDF heatmaps. Each has separate
  panels for W0/W1/W2; cells average across heads and sequences. Inspect the
  summary/per-head files for tails hidden by the mean.

Default SVD runs on CUDA in float32; `--svd-device cpu` moves SVD to CPU while
model evaluation still uses CUDA. `--no-plots` skips Matplotlib. The output
directory must be new to prevent mixing different experiments. These validation
prefixes are a spectral pilot, not the full lm-eval WikiText test protocol.

## Running the adaptation/write probe

Sync `diag_rank.py`, `LinearTTT/model/LinearizeLlama/LinearizeLlama.py`, and
`LinearTTT/model/LinearizeLlama/ttt_l2.py` to the cluster. This extension captures
the retention multipliers actually used inside the compiled update operator.
Ordinary training and evaluation leave that extra diagnostic output disabled.

Reuse the exact previous pilot sequences and the same fixed checkpoint pair:

```sh
python diag_rank.py \
  --cfg configs/ttt_ar_unified.yml \
  --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_at_unified/best_ckpt \
  --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified/best_ckpt \
  --tokens-file /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_rank_8192/tokens.pt \
  --corpora pg19 wikitext --seq-len 8192 --seqs 8 \
  --chunks 0 1 2 4 8 15 --components adaptation update \
  --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_adaptation_rank_8192
```

The output schema now includes `component` and `write_chunk`. At state index c:

- `state`: W[c], the weights used to read chunk c, after exactly c updates.
- `adaptation`: W[c] - prod(alpha[:c]) W[0]; identically zero at c=0.
- `update`: W[c] - alpha[c-1] W[c-1], the write from chunk c-1. There is no
  update row at c=0. Even when sampled states skip chunks, subtraction uses the
  immediately preceding state, not the previously sampled state.

The cumulative product and subtractions use float64 on copied CPU states; SVD
uses float32. Tiny differences can still reflect rounding in the original
operator. `relative_to_state_norm` and `state_frobenius_norm` quantify component
size: avoid interpreting rank alone for a negligible component. The update is
reconstructed from the actual state transition and includes transition rounding;
it is not the raw gradient or a separately sampled pre-Muon direction.

Outputs retain the same JSON/CSV names. Heatmaps are separated, for example
`wikitext_adaptation_r95.png` and `wikitext_update_r95.png`. Summaries group by
component so full states, accumulated adaptation, and individual writes are
never averaged together. Existing state-only summaries remain plottable.

## Question and scope

Determine whether the unified and private TTT models have compressible recurrent
states, and whether compression preserves predictions and long-context behavior.
Use the fresh unified stage-1 + stage-2 checkpoint measured at WikiText word PPL
10.8027 and a fixed, matched private-model checkpoint. Record exact paths, config,
checkpoint identifiers, code revision, tokenizer, and dataset/token hashes.

The linear-attention connection motivates measurement, not a low-rank assumption.
A sum of rank-one writes can become full rank; a full-rank initial state can
remain full rank. Our model updates all three SwiGLU matrices using residual MSE,
momentum, Muon and retention, so it is not a fixed-feature linear memory.

Relevant source: Liu et al., *Test-Time Training with KV Binding Is Secretly
Linear Attention*, https://arxiv.org/html/2602.21204v1. Section 5 gives the
generalized attention interpretation; Appendix I distinguishes it from the
simpler parallel reduction when feature weights are also updated. This paper
does not establish low numerical rank for our implementation.

## What to measure

The fast-weight component of the recurrent state is (W0, W1, W2). Momentum
buffers are additional recurrent state; reader trajectories are derived copies
used in full-sequence execution, not additional independent shared memories.
Do not confuse a transformer's token hidden activation with this recurrent state.

For each sequence, memory owner, head, matrix j and chunk c, collect:

1. The learned initial weight W[j,0] and running state W[j,c].
2. The actual write U[j,c] = W[j,c+1] - alpha[c] W[j,c].
3. Accumulated adaptation E[j,c] = W[j,c] - A[c] W[j,0], where
   A[0] = 1 and A[c+1] = alpha[c] A[c]. Alpha is the actual per-head chunk-mean
   retention used by the operator. This separates initialization from writes.
4. The raw gradient-derived write, momentum accumulator, and post-Muon write,
   separately. A compressible raw gradient does not establish a compressible
   applied update. Under finite-step Newton-Schulz, measure rather than assume
   singular-value flattening.

Use actual dimensions from the checkpoint. With 32 heads, head dimension 128 and
expansion 1, each of the three per-head matrices is 128 x 128. Analyze each
matrix separately; never average matrices before SVD or flatten heads together.
Count a unified memory once, regardless of its number of readers.

## First pilot: spectra only

- Eight fixed 8192-token sequences from held-out PG19 and eight from WikiText
  validation; reuse identical tokens for both models. This is exploratory, not
  a final benchmark. Avoid choosing ranks on the final WikiText test set.
- Capture initialization and states used to read chunks 1, 2, 4, 8 and 15
  (zero-based). With chunk size 512, an 8192-token forward reads 16 chunks and
  applies only 15 updates; do not invent a terminal update after the last chunk.
- Unified: all seven independent memories (L0, L1, L2, L28-L31).
- Private: the same seven layers initially. If results justify expansion,
  include L8, L16 and L24 to characterize interior private memories.
- Compute singular values in float32 on detached snapshots outside the compiled
  model operator. Stream summaries to CPU; do not retain all GPU trajectories
  for the whole dataset. Instrumentation must leave model outputs unchanged.

For nonzero singular spectra s, report:

- Energy ranks r90, r95 and r99: smallest r with
  sum(s[:r]^2) / sum(s^2) >= 0.90, 0.95 or 0.99.
- Stable rank: sum(s^2) / max(s)^2.
- Relative best-rank-r Frobenius error:
  sqrt(sum(s[r:]^2) / sum(s^2)).
- Frobenius norms of states, initialization and adaptation. Low rank in a nearly
  zero adaptation component may be unimportant.

Treat zero matrices explicitly (rank zero; normalized quantities undefined).
Use energy criteria rather than default matrix_rank tolerances on BF16 weights.
Report per-head distributions and per-sequence summaries, including high-rank
tails; do not treat all heads/chunks as independent experimental samples.

Deliverables: JSON/CSV with sequence, layer, role, head, chunk, matrix, component,
norms, singular values and rank metrics; spectral-decay plots; rank-versus-chunk
plots; unified/private and anchor/interior comparisons.

## Second pilot: functional sensitivity

Only after inspecting spectra, test ranks 8, 16, 32 and 64 when compatible with
the actual dimensions. Start with W1, the linear readout matrix. Then test W0/W2
and the joint three-matrix approximation if W1 compression is promising.

For recorded query vectors, compare original and truncated-state SwiGLU
readouts. Measure relative error and cosine similarity before and after TTT
normalization/gating, plus contribution error after the layer's output
projection. Near-zero reference outputs need absolute-error reporting too.
For unified memory, evaluate the writer and multiple readers (early, middle,
late); identical shared weights need not have identical query sensitivity.

Compare two approximation targets:

1. Full-state SVD: W[c] -> best_rank_r(W[c]).
2. Adaptation-only SVD: W[c] -> A[c] W[0] + best_rank_r(E[c]).

Keep full precision updates in this offline sensitivity test. It measures
readout sensitivity only, not the trajectory error of recurrent compression.
Full-rank reconstruction must reproduce the unmodified readout within the
appropriate dtype tolerance.

## End-to-end confirmation and accounting

If offline results are promising, implement explicit compression at chunk
boundaries and propagate the compressed state into subsequent updates. Compare
against the uncompressed checkpoint on held-out CE, then full WikiText word PPL,
PG19 and long-context retrieval. Verify prefill/decode agreement and causality.
Test the trained 8192 length first; treat 16K/32K as separate extrapolation tests.
Select the rank on validation data and lock it before final test evaluation.

For an m x n state stored as two factors with singular values absorbed into one
factor, storage is r(m+n), versus mn dense. At 128 x 128, r=32 halves matrix
storage; r=64 does not save storage. Reconstructing a dense matrix after SVD is
not a memory-saving implementation. A factorized implementation must account
for update workspace and factorization cost as well as read cost.

For adaptation compression, the dense learned W[0] remains as shared model
storage, while factors are per-sequence dynamic state. Report both. Account
separately for momentum buffers, pending chunk K/V/LR buffers, attention-window
KV, and full-sequence trajectory storage. A fast-weight compression ratio is
not an end-to-end GPU memory or speedup claim.

Decision rule: proceed only if low-rank factors offer positive net storage
savings and acceptable held-out quality loss under an agreed tolerance.
If full-state spectra are flat, inspect adaptation and query sensitivity before
rejecting compression. If all are unfavorable, retain the working checkpoint
and report the negative result rather than forcing a low-rank architecture.
