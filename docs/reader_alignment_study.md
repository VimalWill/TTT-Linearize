# Reader output alignment study

This controlled continuation starts from the completed unified stage-1 and
stage-2 checkpoint pair. Each reader L3--L27 gets a separate per-head linear
map on its TTT memory output, **before** `ttt_norm`, gating, and `o_proj`:

\[
y_{\ell,h,t}=f_{W_{2,c}}(q_{\ell,h,t}),\qquad
\widetilde y_{\ell,h,t}=A_{\ell,h}y_{\ell,h,t},\qquad A_{\ell,h}=I\text{ initially}.
\]

Here `c` is the current chunk and `W[2,c]` contains only previous chunks.
There are 25 x 32 x 128 x 128 = **13,107,200** added parameters for the
current 8B model. Each reader's map is independent; there are no maps on L2
or the six private-memory layers. There is no cross-head mixing.

Identity gives the original computation (up to numerical roundoff). A
fixed identity is a control, not an alignment intervention. The default
study learns general linear maps; these are **not constrained rotations**.
Their effect can include scaling, shearing, and rotation within each head.
This is an output-side experiment; query alignment is a separate question.

## Training scope

`Configs/ttt_ar_unified_align.yml` enables the maps and resumes both parts
of the existing checkpoint: stage-2 TTT weights are overlaid and the existing
LoRA weights merged into the base model. No new LoRA is created.

Only the reader maps train, using autoregressive CE on PG19. The first run
uses one epoch, learning rate 1e-4, zero weight decay, and the existing batch
and validation settings. The maps use FP32 optimizer parameters while the
readout computation follows the model dtype. All existing model parameters
remain frozen. Inner-loop adaptation still runs; later private memories can
change their trajectories because they receive changed hidden states.

The output is a **full merged model checkpoint**, including the reader maps.
Evaluate it without `--adapter`. The original two checkpoints remain the
baseline. A resolved `study_config.yaml` records the source paths and training
settings; an existing nonempty study directory is rejected.

## Run on the cluster

### Paired identity check on both corpora, without training

`eval_reader_identity.py` loads the model once and evaluates each sequence
with reader maps bypassed and with identity multiplication active. It checks
that all reader maps are exactly identity and requires all saved stage-2 TTT
weights to overlay. Each forward starts a fresh memory trajectory. The
language-model head is evaluated in blocks to avoid retaining full
8192-by-vocabulary logits for both arms at once.

```text
python eval_reader_identity.py \
    --cfg Configs/ttt_ar_unified.yml \
    --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_at_unified/best_ckpt \
    --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified/best_ckpt \
    --corpora pg19 wikitext \
    --seq-len 8192 --seqs 8 \
    --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_reader_identity_8192
```

This performs 32 backbone forwards: two arms for eight sequences per corpus.
No optimizer is created, no parameter changes, and no checkpoint is written.
The config's reader maps are enabled automatically. Validation text is loaded
directly from WikiText and PG19; the runner has no dependency on the rank
probe or its files. Both arms use identical tokens. Use a new output directory
for each run.

The output includes `results.json`, `results.csv`, `sequences.json`, and
`metadata.json`. Perplexity is **token-level on the sampled sequences**,
computed as the exponential of total NLL divided by the number of targets;
it is not comparable directly to the earlier full-test WikiText word PPL
10.8027. The relevant comparison here is baseline versus identity on the
same tokens. Maximum absolute logit differences and argmax disagreements
are also saved. Differences should be zero or within numerical tolerance
(default absolute logit tolerance 1e-3). Larger differences make the process
exit nonzero after writing the measurements for inspection.

The four local runner tests cover scores against a full LM forward, unchanged
parameters, a nonidentity negative control, flag restoration on failure, and
correct aggregation of corpus perplexity. Actual checkpoint validation runs
on the HPC GPU.

### Full WikiText evaluation and subsequent map training

Sync the changed model/config/training/eval files, the new configuration,
`LinearTTT/diagnostics.py`, and `diag_similarity.py`. The latter ensures that
future similarity probes include the new maps in their initial reference.
Use the pinned dependencies from `requirements.txt`, including
Transformers 4.45.0 and PEFT 0.13.2. Commands use the cluster's `Configs`
capitalization.

First evaluate the original checkpoint with identity-initialized maps:

```text
python eval.py \
    --cfg Configs/ttt_ar_unified_align.yml \
    --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_at_unified/best_ckpt \
    --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified/best_ckpt \
    --tasks wikitext --seq-len 8192 --batch-size 1 \
    --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_reader_identity_init
```

The 25 missing `ttt_reader_alignment.weight` keys are expected: the loader
initializes them to identity. Existing stage-2 TTT tensors must all overlay.
Compare with the existing full unified baseline under the same evaluator
(previously reported word PPL 10.8027, provided checkpoint contents and the
evaluation setup are unchanged). Do not substitute an older `_1ep` checkpoint.

Train the maps:

```text
TTT_CKPT_DIR=/work/nvme/bgly/vwilliam/ttt-checkpoints python run.py --cfg Configs/ttt_ar_unified_align.yml
```

Expect `Training only 13,107,200 reader-alignment parameters`. Select on PG19
validation CE; reserve WikiText for evaluation. Training writes under
`ttt_ar_unified_align`, separate from the two source checkpoints. This is
backpropagation through the model and will take longer than the similarity
forward-pass probe; it is not a guaranteed 10--20 minute run.

Evaluate the learned maps:

```text
python eval.py \
    --cfg Configs/ttt_ar_unified_align.yml \
    --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified_align/best_ckpt \
    --tasks wikitext --seq-len 8192 --batch-size 1 \
    --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_reader_learned
```

Then bypass the learned maps to recover the identity control in that same
checkpoint:

```text
python eval.py \
    --cfg Configs/ttt_ar_unified_align.yml \
    --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/ttt_ar_unified_align/best_ckpt \
    --identity-readers \
    --tasks wikitext --seq-len 8192 --batch-size 1 \
    --out /work/nvme/bgly/vwilliam/ttt-checkpoints/unified_reader_control
```

The final output filename includes `_identity_readers`. This control should
recover the original baseline because all pre-existing parameters were
frozen. Report PG19 held-out results alongside WikiText, and use matched
retrieval slices/context lengths when comparing against retrieval ablations.

## Interpretation

- If identity changes results materially before training, stop and inspect
  loading, configuration, precision, and evaluation settings.
- If trained maps improve held-out results and bypassing them restores the
  baseline, this demonstrates a useful reader output transformation.
- Improvement alone does not prove the earlier different-bases hypothesis:
  the extra parameters can provide other corrections. A rotation-constrained
  map and a diagonal-only control can further distinguish alignment from
  scaling/capacity effects. A separately fitted map between independent
  layers on held-out sequences addresses representational correspondence
  more directly.
- No improvement would not rule out all alignment mechanisms: the experiment
  tests per-head output maps at one insertion point, with a particular data
  and optimization budget.

## Validation

The targeted CPU suite passes identity equivalence, batched head mapping,
old-checkpoint loading, learned-map save/reload, stage-2 TTT plus LoRA resume,
map-only gradient updates under checkpointing, future-token causality,
cached-decode equivalence, and similarity-reference consistency. Existing
shared-memory, similarity, and trainer-update checks also pass (28 tests).
GPU checkpoint training/evaluation must be run on the cluster.

The existing evaluation causality check now lives in
`LinearTTT/diagnostics.py`, so evaluation no longer imports the deleted
root-level `test_causality.py` script.
