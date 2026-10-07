# MAD integration

Add `third_party/mad-lab` to `PYTHONPATH` and install its dependencies in the
training environment. Call `register_ttt()` before building registry models:

```python
from LinearTTT.mad.register import register_ttt
register_ttt()
from mad.configs import MADModelConfig
```

Registration provides `ttt` (memory only) and `ttt-hybrid` (TTT + local SWA),
and protects intentional TTT initialization in both MAD LanguageModel and
AutoEncoder. Gate/retention biases remain -3/+4; their weight matrices remain
zero. Shared readers alias the writer's `w0/w1/w2` Parameters, so optimizers
and `model.parameters()` count each set once. Build/link the same topology
before loading a state dictionary; keep its configuration with checkpoints.

## Three hybrid layers followed by one SWA layer

For stage-1 attention transfer, first train an all-attention MAD language model
on the chosen synthetic task. Then convert a copy of that trained model:

```python
import copy
from LinearTTT.mad.hybrid import hybridize_mad_model

# trained_attention_model uses SDPAAttention (registered by register_ttt).
# It has a multiple of four attention slots; MLP slots are counted separately.
student = hybridize_mad_model(
    copy.deepcopy(trained_attention_model),
    hybrid_window=512, swa_window=4096, chunk_size=64,
)
```

Each group contains three hybrid mixers sharing one writer's causal trajectory
and a fourth SWA-only mixer. Hybrid mixers project q/k/v once, add gated TTT to
SWA, then use a shared output projection. Conversion preserves trained
projections, rotary frequencies, embeddings and MLPs. This is a MAD adaptation:
it retains the primitive's per-feature output gate, has no HF decode/cache
path, and is not a checkpoint-compatible replacement for the Llama model.

For a standalone registry stack, `link_shared_memories(model, [[0,1,2], ...])`
indexes TTT mixers only. It must be called before optimizer construction.
The generic MAD CLI does not call this helper automatically.

## Distillation versus task cross-entropy

Use the new wrapper with Lightning and MAD-generated train/validation loaders:

```python
from LinearTTT.mad.training import MADObjectiveWrap

wrapped = MADObjectiveWrap(student, mad_config, mode="distill",
                           mse_factor=1000., lm_loss_weight=1.)
trainer.fit(wrapped, train_dl, validation_dl)
```

The stage-1 objective freezes the inherited backbone/projections and trains
the TTT parameters. At each hybrid mixer its teacher is full causal attention
on the same hidden states, using the frozen trained projections. The objective
is `lm_loss_weight * task_CE + mse_factor * mean_attention_MSE`; the mean is
over hybrid mixers. The teacher target has no gradients. This uses local
attention-output transfer, rather than teacher-logit KL distillation.

MAD labels are already aligned to their prediction positions, so they are
not shifted. The wrapper logs task CE, raw MSE and weighted MSE separately,
alongside MAD's total loss, accuracy and perplexity. Select checkpoints using
task CE or accuracy when comparing objectives with different MSE weights.

For direct task training, use `mode="task"` on a fresh model with the same
hybrid topology. This trains all parameters using CE. For continuation, create
a new task wrapper around the distilled model and a fresh optimizer. Keep
datasets, initialization, seeds and compute budgets controlled; teacher
pretraining compute belongs in the total distillation cost. Task CE on these
synthetic tasks is not the source Llama model's pretraining loss.

`mad_loss(...)` and `configure_objective(...)` in `objective.py` expose the
same behavior for a custom trainer without Lightning. The upstream MAD
`train.py` still uses its standard CE-only wrapper; use `MADObjectiveWrap`
explicitly to enable the new objective.

## Accounting and validation

`total_state_dim(model)` counts only active fast-weight elements. It does not
normalize to 4096 or include momentum buffers, KV caches, activations or saved
trajectories. The default 128-wide, 4-head writer has 12,288 fast-weight
elements. Report this count and unique model parameters separately; measure
actual peak memory/FLOPs on the training host for scalability plots.

A 4096-token window covers all positions in MAD's usual short sequences.
Use longer sequences or an explicitly documented scaled-window experiment
before drawing conclusions about long-range retrieval.

Run `python3 -m unittest discover -s tests -v`. Local checks cover initialization
using the actual upstream reset routines, causal sharing, parameter counting,
checkpoint reload, fused attention, and separate objective gradients. Full
MAD/Lightning registration and CUDA execution require the training dependencies;
the current local environment lacks `lightning_utilities`.
