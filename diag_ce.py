"""Isolate whether the stage-1 checkpoint broke language modelling.

Measures next-token cross-entropy on the same PG19 validation chunks for three
configurations. ln(128256) = 11.76 is the uniform-random baseline.

    A. base Llama + window 8192  -> ~full attention. Validates the harness.
    B. base Llama + window 512   -> cost of the window alone (TTT gate is
                                    near-closed at init, so it contributes little).
    C. stage-1 checkpoint        -> what attention transfer actually produced.

    python3 diag_ce.py --ckpt $TTT_CKPT_DIR/ttt_at/best --batches 20
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from Training.dataloader import load_data


def ce_of(path, config, window, chunk, loader, n_batches):
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = path
    cfg.model.window_size = window
    cfg.model.lact_chunk_size = chunk
    model_config = build_model_config(cfg)

    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16).eval()

    total, count = 0.0, 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= n_batches:
                break
            ids = data['input_ids'].cuda()
            logits = model(input_ids=ids, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                ids[:, 1:].reshape(-1),
            )
            total += loss.item()
            count += 1

    del model
    torch.cuda.empty_cache()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_ar.yml')
    ap.add_argument('--ckpt', required=True, help='stage-1 checkpoint dir')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--batches', type=int, default=20)
    ap.add_argument('--windows', type=int, nargs='+', default=None,
                    help='sweep base-model CE over these window sizes instead of A/B/C')
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    loader = load_data(config)['validation']

    vocab = 128256
    print(f'\nuniform-random baseline: ln({vocab}) = {math.log(vocab):.3f}\n')

    if args.windows:
        runs = [(f'base + window {w}', args.base, w, min(512, w))
                for w in args.windows]
    else:
        runs = [
            ('A base + window 8192 (~full attn)', args.base, 8192, 512),
            ('B base + window 512', args.base, 512, 512),
            ('C stage-1 checkpoint + window 512', args.ckpt, 512, 512),
        ]
    for label, path, window, chunk in runs:
        ce = ce_of(path, config, window, chunk, loader, args.batches)
        print(f'{label:36s} CE = {ce:7.3f}   ppl = {math.exp(min(ce, 20)):.3g}')


if __name__ == '__main__':
    main()
