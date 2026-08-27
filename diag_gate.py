"""How much of each layer's output actually comes from the TTT branch?

The student output is `attn_out + ttt_out`, where
`ttt_out = ttt_norm(fast_weight_out) * silu(ttt_scale_proj(h))`. A good
cross-entropy could in principle come from the sliding-window branch while the
TTT branch shrinks toward nothing -- which would undercut the whole point. This
reports, per layer, the gate value and the share of the merged output that the
TTT branch contributes.

At init the gate is silu(0.1) = 0.0525, so a fresh model is the "before".

    python3 diag_gate.py --ckpt $TTT_CKPT_DIR/ttt_at/best
"""

import argparse

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config


def load_model(path, model_config, adapter=None):
    """Full checkpoint, optionally with PEFT adapters and saved TTT params on top."""
    import os
    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16)
    if adapter:
        from peft import PeftModel
        ttt = os.path.join(adapter, 'ttt_params.pt')
        if os.path.exists(ttt):
            model.load_state_dict(torch.load(ttt, map_location='cpu'), strict=False)
            print('  overlaid saved TTT params')
        else:
            print('  NOTE: no ttt_params.pt -- stage-2 TTT weights were never saved')
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval()


def measure(path, config, tok, ids, adapter=None):
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = path
    model_config = build_model_config(cfg)
    model = load_model(path, model_config, adapter)

    cap = {}

    def grab(idx, key):
        def fn(module, inp, out):
            cap.setdefault(idx, {})[key] = (inp[0] if key == 'merged' else out).detach()
        return fn

    handles, layers = [], model.model.layers
    for i, layer in enumerate(layers):
        a = layer.self_attn
        handles.append(a.ttt_norm.register_forward_hook(grab(i, 'ttt')))
        handles.append(a.ttt_scale_proj.register_forward_hook(grab(i, 'gate')))
        handles.append(a.o_proj.register_forward_hook(grab(i, 'merged')))

    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    rows = []
    for i, layer in enumerate(layers):
        nh = layer.self_attn.num_ttt_heads
        c = cap[i]
        gate = F.silu(c['gate'].float())                       # [b, n, nh]
        ttt = c['ttt'].float()                                 # [(b nh), n, d]
        g = rearrange(gate, 'b n (h d) -> (b h) n d', h=nh)
        ttt_out = rearrange(ttt * g, '(b h) n d -> b n (h d)', h=nh)
        merged = c['merged'].float()                           # [b, n, inner]
        attn_out = merged - ttt_out
        rows.append((i, gate.mean().item(),
                     ttt_out.norm().item() / merged.norm().item(),
                     attn_out.norm().item() / merged.norm().item()))

    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--adapter', default=None,
                    help='PEFT adapter dir to overlay on --ckpt (stage-2 output)')
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    tok = AutoTokenizer.from_pretrained(args.base)
    text = ('The history of the decline and fall of the Roman Empire begins in the '
            'age of the Antonines, when the empire comprised the fairest part of '
            'the earth and the most civilised portion of mankind. ') * 1600
    ids = tok(text, return_tensors='pt').input_ids[:, :args.seq_len].cuda()

    print('\ngate at init = silu(0.1) = 0.0525\n')
    runs = [('BEFORE (fresh init)', args.base, None), ('AFTER (stage 1)', args.ckpt, None)]
    if args.adapter:
        runs.append(('AFTER (stage 2 adapters)', args.ckpt, args.adapter))
    for label, path, adapter in runs:
        rows = measure(path, config, tok, ids, adapter)
        gates = [r[1] for r in rows]
        ttt_share = [r[2] for r in rows]
        print(f'--- {label} ---')
        print(f'{"layer":>5} {"gate":>9} {"|ttt|/|out|":>12} {"|attn|/|out|":>13}')
        for i, g, t, a in rows:
            if i % 4 == 0 or i == len(rows) - 1:
                print(f'{i:>5} {g:>9.4f} {t:>12.4f} {a:>13.4f}')
        print(f'  mean gate {sum(gates)/len(gates):.4f} | '
              f'mean TTT share {sum(ttt_share)/len(ttt_share):.4f}\n')


if __name__ == '__main__':
    main()
