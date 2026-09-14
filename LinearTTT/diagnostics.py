"""Runtime checks used by evaluation and small model tests."""
import math

import torch


@torch.no_grad()
def assert_causal(model, input_ids, chunk_size, atol=1e-3):
    """Future-token tripwire at chunk boundaries and in the tail."""
    length = input_ids.shape[1]
    if length <= 2 * chunk_size:
        raise ValueError('Use more than two chunks to exercise shared-memory updates')
    if model.training:
        raise ValueError('Causality checks require model.eval()')
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    baseline = model(input_ids=input_ids, use_cache=False).logits
    if not torch.isfinite(baseline).all():
        raise AssertionError('Nonfinite baseline logits')
    positions = sorted({1, chunk_size - 1, chunk_size, chunk_size + 1,
                        2 * chunk_size, length - 1} - {0})
    results = {}
    for position in positions:
        edited = input_ids.clone()
        edited[:, position] = (edited[:, position] + 1) % model.config.vocab_size
        logits = model(input_ids=edited, use_cache=False).logits
        delta = (logits[:, :position].float() - baseline[:, :position].float()).abs().max().item()
        if not math.isfinite(delta) or delta > atol:
            raise AssertionError(f'Future edit at {position} changed earlier logits by {delta:g} (tol {atol:g})')
        results[position] = delta
    return results
