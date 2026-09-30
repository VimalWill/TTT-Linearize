# Passkey continuation at 2048 tokens

This is a small autoregressive continuation of the **trained anchor**, testing
whether explicit retrieval supervision teaches its memory to preserve arbitrary
digits. It does not rerun attention transfer or change memory sharing/readers.
It is a diagnostic experiment, not a claim that longer-context retrieval is solved.

Run from the repository root in the existing CUDA training/evaluation environment:

```bash
export ANCHOR_CKPT=/path/to/anchor/full_base_checkpoint
export ANCHOR_ADAPTER=/path/to/anchor/stage2/best_ckpt
export BASE_TOKENIZER=/path/to/matching/tokenizer
export TTT_CKPT_DIR=/path/to/new/passkey_2048_run
bash scripts/passkey_2048.sh
```

`ANCHOR_CKPT` must contain the anchor model config, weights, and tokenizer. When
using `ANCHOR_ADAPTER`, supply the exact full/base checkpoint against which that
adapter was trained; the adapter directory must include `ttt_params.pt`. The
existing adapter is continued in place with a fresh optimizer, and the resulting
adapter still loads against that same base. If the trained model was already
exported as a full merged checkpoint, use that as `ANCHOR_CKPT` and leave
`ANCHOR_ADAPTER` unset. A plain pretrained Llama/Mistral checkpoint is not the
trained anchor.

The output directory must be new. `PYTHON_BIN` can select the Python executable.
For a plumbing check, `RULER_LIMIT=20` limits each NIAH task; omit it for the full
comparison. A small-limit run is not evidence of a retrieval improvement.

## What runs

1. Evaluate `niah_single_1`, `niah_single_2`, and `niah_single_3` at 2048, both
   intact and with `--ablate ttt`, using the same seed and per-example logs.
2. Continue training on `nanotron/llama3-16k-passkey-retrieval-finetuning`, with
   answer-only cross entropy and no auxiliary attention-transfer loss. The pilot
   uses outer LR `1e-4`, plateau scheduling, batch size 8 (microbatch 1), and a cap
   of 400 optimizer updates or 10 epochs, whichever comes first. This LR/budget is
   an initial experiment choice, not a tuned optimum.
3. Evaluate the best held-out answer-CE checkpoint with the same four-way
   comparison: before/after training × memory intact/ablated.

Examples stay unpacked and intact. Filtering counts actual tokens, including the
completion and special tokens, with the checkpoint's tokenizer: 1024–2048 tokens
total, with at least 768 tokens after the **last** key occurrence before the answer
prompt. This removes direct access through a 512-token local window; stacked local
layers can still relay information, which is why the ablation remains necessary.
The filter also enforces unique answer strings before the deterministic split:
200 validation examples, at most 10,000 training examples. The source may yield
fewer after filtering; the loader prints actual counts, including malformed rows
it skipped, and fails if the validation split leaves no training examples.
Multiple epochs can repeat training examples.

Only answer/completion tokens and EOS have training labels. Generation inputs
exclude the supplied answer completion. The validation metric is teacher-forced
answer CE, not exact-match generation accuracy. RULER's own scoring is retained.
Numeric passkey training alone may not transfer to NIAH UUIDs or multi-key tasks;
read the three task results separately.

## Reading the result and extending the context

Inspect the `2048` metric in `eval/before.json`, `before_ttt_all.json`, `after.json`,
and `after_ttt_all.json`; adjacent `.samples.json` files contain predictions.
Look for both improved intact retrieval and a positive intact-minus-ablated gap
on the same examples, with uncertainty estimated from paired per-example scores.
A CE decrease alone does not establish memory-mediated retrieval.

If that gap appears at 2k, evaluate the **same checkpoint** at 4k, 8k, and 16k,
one length per invocation, with memory on/off. Use `--ruler-lengths LENGTH` and
`--seq-len LENGTH_PLUS_512` to reserve decoding space. Keep the local window and
TTT chunk at 512. Train at greater lengths only if those tests show a retention
gap; increasing a context-length setting alone does not teach retrieval.

If the 2k ablation gap stays near zero, inspect paired outputs and compare numeric
versus UUID tasks before spending on a longer run. This experiment cannot by
itself establish the cause of a negative result.

## Local validation

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/passkey_2048.sh
```

CPU tests cover masking, generation input leakage, actual-token/distance filtering,
key-disjoint validation, adapter/TTT checkpoint continuation and reload, and an
answer-position gradient reaching earlier writes through both private and shared
memory readouts. They do not replace a CUDA end-to-end training/RULER run.
