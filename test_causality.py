"""Future-token tripwire for a trained checkpoint (no perplexity ordering assumption).

python test_causality.py --cfg Configs/ttt_ar_unified.yml --ckpt PATH --adapter PATH
For small CPU regression tests: TORCHDYNAMO_DISABLE=1 python -m unittest discover -s tests -v
"""
import argparse
import math

import torch


@torch.no_grad()
def assert_causal(model, input_ids, chunk_size, atol=1e-3):
    """Check earlier logits at boundaries, within written chunks, and in the tail."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    from Training.train import build_model_config
    from eval import load_model

    cfg = OmegaConf.load(args.cfg)
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model = load_model(args.ckpt, build_model_config(cfg), args.adapter)
    torch.manual_seed(args.seed)
    ids = torch.randint(model.config.vocab_size, (1, args.seq_len), device=model.device)
    for position, delta in assert_causal(model, ids, model.config.lact_chunk_size, args.tol).items():
        print(f'edit @ {position}: earlier max |delta logit| = {delta:g}')
    print('PASS: all future-token probes, including the tail')


if __name__ == '__main__':
    main()
