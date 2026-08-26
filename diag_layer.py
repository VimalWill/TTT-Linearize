"""Break one LinearTTTAttention layer into pieces and report magnitudes.

diag_inf.py localises the blowup to a layer; this says which tensor inside it.
Hooks the submodules so we see q/k/v, the raw TTT operator output (the input to
ttt_norm), the gate, and the merged student (the input to o_proj).

    python3 diag_layer.py --cfg Configs/ttt_at_smoke.yml --layers 0 1 2
"""

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config


def mag(t):
    if t is None:
        return 'None'
    t = t.detach().float()
    finite = torch.isfinite(t)
    if not finite.all():
        n_inf = (~finite).sum().item()
        return f'NONFINITE({n_inf}/{t.numel()})'
    return f'absmax={t.abs().max():.3e} rms={t.pow(2).mean().sqrt():.3e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at_smoke.yml')
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    model_config = build_model_config(config)
    path = config.model.pretrained_model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(path)

    text = ('The history of the decline and fall of the Roman Empire begins in the '
            'age of the Antonines, when the empire comprised the fairest part of '
            'the earth and the most civilised portion of mankind. ') * 400
    ids = tok(text, return_tensors='pt').input_ids[:, :args.seq_len].cuda()

    log = []

    def watch(layer_idx, name, use_input=False):
        def fn(module, inp, out):
            t = inp[0] if use_input else (out[0] if isinstance(out, tuple) else out)
            log.append((layer_idx, name, mag(t)))
        return fn

    handles = []
    for i in args.layers:
        attn = model.model.layers[i].self_attn
        for name, use_input in [('q_proj', False), ('k_proj', False), ('v_proj', False),
                                ('lr_proj', False), ('ttt_scale_proj', False),
                                ('ttt_norm <- fw_x', True), ('ttt_norm -> out', False),
                                ('o_proj <- student', True)]:
            mod = getattr(attn, name.split()[0])
            handles.append(mod.register_forward_hook(watch(i, name, use_input)))

        # the fast weights are parameters, not module outputs
        for wn in ('w0', 'w1', 'w2'):
            w = getattr(attn, wn)
            log.append((i, f'{wn} (init)', mag(w)))

    with torch.no_grad():
        model(input_ids=ids, output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()

    cur = None
    for idx, name, m in sorted(log, key=lambda r: r[0]):
        if idx != cur:
            print(f'\n=== layer {idx} ===')
            cur = idx
        print(f'  {name:22s} {m}')


if __name__ == '__main__':
    main()
