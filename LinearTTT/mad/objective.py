"""MAD task CE and stage-1 attention transfer with separately reported losses."""
import math
import torch
import torch.nn.functional as F

from .hybrid import TTTHybridBlock


def configure_objective(model, mode='task'):
    if mode not in ('task', 'distill'):
        raise ValueError('mode must be task or distill')
    blocks = [m for m in model.modules() if isinstance(m, TTTHybridBlock)]
    if mode == 'distill' and not blocks:
        raise ValueError('Attention distillation requires TTTHybridBlock layers')
    for p in model.parameters():
        p.requires_grad_(mode == 'task')
    if mode == 'distill':
        frozen_projections = {id(p) for block in blocks
                              for module in (block.qkv, block.out)
                              for p in module.parameters()}
        for block in blocks:
            for p in block.parameters():
                if id(p) not in frozen_projections:
                    p.requires_grad_(True)
    for block in blocks:
        block.collect_distillation = mode == 'distill'
    return model


def mad_loss(model, inputs, targets, mode='task', mse_factor=1000.,
             lm_loss_weight=1., ignore_index=-100):
    """MAD targets are already aligned: do not shift labels as in HF LM loss."""
    if mode not in ('task', 'distill'):
        raise ValueError('mode must be task or distill')
    if any(not math.isfinite(w) or w < 0 for w in (mse_factor, lm_loss_weight)):
        raise ValueError('Loss weights must be finite and nonnegative')
    if not (targets != ignore_index).any():
        raise ValueError('Batch has no supervised task targets')
    blocks = [m for m in model.modules() if isinstance(m, TTTHybridBlock)]
    if mode == 'distill' and (not blocks or any(not b.collect_distillation for b in blocks)):
        raise ValueError('Call configure_objective(model, "distill") before computing distillation loss')
    logits = model(inputs)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                         targets.to(logits.device).reshape(-1), ignore_index=ignore_index)
    mse = ce.new_zeros(())
    if mode == 'distill':
        if any(b.distillation_loss is None for b in blocks):
            raise ValueError('Every hybrid block must execute in the distillation forward')
        mse = torch.stack([b.distillation_loss for b in blocks]).mean()
    # Release graph references after collecting the scalars used in the loss.
    for block in blocks:
        block.distillation_loss = None
    loss = lm_loss_weight * ce + (mse_factor * mse if mode == 'distill' else 0.)
    return loss, logits, {'loss_ce': ce.detach(), 'loss_mse': mse.detach(),
                          'loss_mse_weighted': (mse_factor * mse).detach()}
