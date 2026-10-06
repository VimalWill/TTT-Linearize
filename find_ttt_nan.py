"""Find where the TTT path first goes non-finite as context grows.

diagnose_retrieval.py reports '!!!!!!' for every full-context generation with
the memory on, and '!' is token 0 in Llama's vocabulary -- what argmax returns
when logits are NaN. The same prompts with the memory on but a short
facts-only context score 1.000 exact, so the memory is not the problem; its
behaviour at length is.

This walks the context length up and reports, per layer, the first tensor in
the TTT path that stops being finite: the fitted fast weights w0/w1/w2, the
readout, and the logits. Where it first breaks says which part diverges.

    python find_ttt_nan.py --cfg Configs/....yml --ckpt meta-llama/Llama-3.1-8B \
      --adapter /path/to/last_ckpt
"""
import argparse

import torch
from omegaconf import OmegaConf

from Training.train import build_model_config
from eval import load_model, ttt_layers
from measure_flops import set_window_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[512, 1024, 2048, 3072, 4096, 6144, 8192])
    ap.add_argument('--max-length', type=int, default=8192)
    a = ap.parse_args()

    config = OmegaConf.load(a.cfg)
    config.model.pretrained_model_name_or_path = a.ckpt
    config.model.max_length = max(int(config.model.max_length), a.max_length)
    # Inspecting the model needs no training data, but data.path is an env
    # interpolation in the long-context configs and resolving the whole tree
    # fails on it. build_model_config reads config.model only.
    container = OmegaConf.to_container(config, resolve=False)
    container.pop('data', None)
    config = OmegaConf.create(OmegaConf.to_container(
        OmegaConf.create(container), resolve=True))
    set_window_backend(True)
    model = load_model(a.ckpt, build_model_config(config), a.adapter, strict_ttt=True)
    model.eval()
    mods = ttt_layers(model)
    print(f'{len(mods)} TTT layers\n')

    seen = {}

    def watch(index):
        def hook(module, args, output):
            # The readout leaves as [b*h, n, d] before the gate and norm, so a
            # blowup here is the inner loop, not the merge.
            out = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(out) and not torch.isfinite(out).all():
                seen.setdefault(index, out.abs().max().item())
        return hook

    handles = [m.register_forward_hook(watch(i)) for i, m in enumerate(mods)]
    try:
        for length in a.lengths:
            seen.clear()
            ids = torch.randint(10, 20000, (1, length), device=model.device)
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits
            ok = torch.isfinite(logits).all().item()
            bad = sorted(seen)
            print(f'len {length:>5}  logits {"finite" if ok else "NON-FINITE"}  '
                  f'logit absmax {logits[torch.isfinite(logits)].abs().max():.3e}  '
                  f'first bad layer {bad[0] if bad else "-"}  '
                  f'bad layers {len(bad)}/{len(mods)}')
            if bad:
                print(f'        layers: {bad}')
    finally:
        for h in handles:
            h.remove()


if __name__ == '__main__':
    main()
