# Test whether interior TTT modules retrieve overlapping information

Start with the **non-unified** checkpoint used in the retrieval ablations.
Its independent memories provide the evidence needed to motivate sharing;
similarity in a model already trained to share is a separate result.

The hypothesis is that some interior modules have both small marginal
retrieval contributions and overlapping readouts. Similarity alone cannot
establish causal redundancy, and small ablation effects alone cannot
establish shared information.

## What the probe measures

`diag_similarity.py` runs ordinary evaluation forwards with temporary hooks.
It compares all pairs of TTT layers at the same sampled query positions.
The default is eight 8192-token sequences per corpus, with up to 128 queries
per slice per sequence. It does not train or change checkpoint parameters.

Three sets of readouts are compared after the layer's normalization, gate,
and output projection into residual-stream coordinates:

| Component | Meaning |
| --- | --- |
| `current` | Actual TTT branch contribution |
| `initial` | Initial memory read at the exact same observed query and gate |
| `history_effect` | `current - initial`, the conditional effect of accumulated memory changes |

The initial reference uses the checkpoint's learned initial weights.
It is evaluated at the observed hidden states; it is **not** a full model
run with memory updates disabled. Similarity in `history_effect` is more
relevant to shared historical information than similarity confined to the
static initial MLP, but common query features can still affect it.

Two token slices are kept separate, using the masks from `eval.py`:

- `cross_window`: repeated-bigram targets whose previous occurrence is beyond
  the sliding attention window. This is a retrieval proxy.
- `other`: targets without a previously seen identical bigram.

Target position `t` is evaluated using its prediction query at `t-1`.
The first chunk is excluded because its memory has received no updates.

## Similarity measures and controls

Use **debiased linear CKA** as the primary layer-pair map. Ordinary linear
CKA is also saved: with many features and few samples it can give large
scores for independent representations. The implementation uses U-centered
Gram matrices and reports a shuffled-token control alongside the aligned
score. Negative debiased scores are possible and are retained.

CKA compares representation geometry and can remain high after a rotation.
Signed cosine separately measures whether the actual residual contributions
point in similar directions. High CKA does not imply the readouts can be
substituted without a learned alignment. The reference is
[Kornblith et al., 2019](https://proceedings.mlr.press/v97/kornblith19a.html).

The shuffle is a descriptive control, not a significance test; neighboring
tokens and chunks are correlated. Results are computed per sequence and
then averaged equally. Sequence standard deviations are saved, not treated
as token-level confidence intervals. Small feature magnitudes and BF16
rounding should be checked before interpreting a history-effect map.

## Run

Use the actual non-unified checkpoint pair that produced the retrieval
results. The command below assumes the standard `ttt_at_l2` / `ttt_ar_l2`
directory names; change them if that experiment used different directories.
The local repository spells the config directory `configs`; use `Configs`
if that is its spelling on the cluster.

```text
python diag_similarity.py \
    --cfg configs/ttt_ar_l2.yml \
    --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_at_l2/best_ckpt \
    --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_l2/best_ckpt \
    --tokens-file /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_rank_8192/tokens.pt \
    --corpora pg19 wikitext --seq-len 8192 --seqs 8 --samples 128 \
    --out /work/nvme/bgly/vwilliam/ttt-checkpoints/nonunified_similarity_8192
```

The token file only supplies the same text; its directory name does not
select the model. Without `--tokens-file`, the script loads validation text
directly. For a 32768-token run, set `--seq-len 32768` and supply matching
tokens or omit `--tokens-file`. Use a new output directory for each run.
Fixed checkpoints are required; do not overwrite them during collection.

Outputs include `summary.json`, per-sequence `pairs.csv`, `magnitudes.json`,
`metadata.json`, sampled `selections.json`, reusable `tokens.pt`, and PNG/PDF
heatmaps. Optional `--save-features` stores the sampled vectors. The script
uses `diag_rank.py` only for shared corpus/loading helpers; no spectral
analysis is run. CUDA is needed for the existing checkpoint loader.

## How this connects to the retrieval experiment

Compare the layer-pair maps with ablation contributions measured on the
**same checkpoint, sequence length, corpus tokens, and retrieval masks**.
The 8192-token pilot does not directly establish an explanation for a
32768-token ablation. This probe itself does not recompute ablation losses.

| Observation | Supported interpretation |
| --- | --- |
| Small marginal contribution, high history-effect similarity above shuffle | Candidate overlap worth testing by sharing or group ablation |
| Small marginal contribution, low history-effect similarity | Weak use or distinct weak signals; no evidence for overlap |
| Similarity mainly in `initial` | Common static computation, insufficient evidence for shared historical retrieval |
| High CKA but low/negative cosine | Related geometry with different residual directions; direct substitution is not established |

Include the anchor layers as a comparison, not just the interior. Inspect
PG19 and WikiText separately. If candidate groups emerge, test their joint
ablation against individual effects on the same tokens; nonlinear effects
must be interpreted cautiously. Finally, evaluate sharing the candidate
group after matched training. Those interventions, together with the
similarity measurements, can support the shared-memory design without
claiming that all independent states contain identical information.

## Local validation

```text
TORCHDYNAMO_DISABLE=1 PYTHONPATH=tests:. python -m unittest test_diag_similarity -v
```

Tests cover CKA invariances and high-dimensional bias, undefined features,
target/query alignment, hook cleanup, unchanged model outputs and weights,
zero history effect before the first update, and the measured contribution
against an actual last-layer TTT ablation. These run on small CPU models;
the full checkpoint measurement must run on the GPU cluster.
