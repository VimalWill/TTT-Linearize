"""Locate where the attention-transfer loss becomes non-finite.

Runs one forward pass with output_attentions=True and reports, per layer, the
magnitude of the residual stream and of the (student, teacher) pair the MSE is
computed over. The first layer whose numbers stop being finite is the culprit.

    python3 diag_inf.py --cfg Configs/ttt_at_smoke.yml --seq-len 2048
"""

import argparse

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import LinearTTT  # noqa: F401  (registers LigerGLAConfig with AutoModel)
from Training.train import build_model_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_at_smoke.yml')
    ap.add_argument('--seq-len', type=int, default=2048)
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    model_config = build_model_config(config)
    path = config.model.pretrained_model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(path)

    print(f'window_size={getattr(model_config, "window_size", None)} '
          f'lact_chunk_size={getattr(model_config, "lact_chunk_size", None)} '
          f'seq_len={args.seq_len}')

    text = ('The history of the decline and fall of the Roman Empire begins in the '
            'age of the Antonines, when the empire comprised the fairest part of '
            'the earth and the most civilised portion of mankind. ') * 400
    ids = tok(text, return_tensors='pt').input_ids[:, :args.seq_len].cuda()

    stats = {}

    def hook(idx):
        def fn(module, inp, out):
            h = out[0]
            stats[idx] = {'h_absmax': h.abs().max().float().item(),
                          'h_finite': bool(torch.isfinite(h).all())}
        return fn

    handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(model.model.layers)]

    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True, use_cache=False)

    for h in handles:
        h.remove()

    print(f'\n{"layer":>5}  {"|h|max":>12} {"h ok":>5}  '
          f'{"|student|":>12} {"|teacher|":>12} {"mse":>12}  {"ok":>4}')
    first_bad = None
    for i, aux in enumerate(out.attentions):
        s = stats.get(i, {})
        if aux is None:
            print(f'{i:>5}  {s.get("h_absmax", float("nan")):>12.3e} '
                  f'{str(s.get("h_finite")):>5}  {"-":>12} {"-":>12} {"-":>12}')
            continue
        student, teacher = aux[0].float(), aux[1].float()
        mse = torch.nn.functional.mse_loss(student, teacher).item()
        ok = bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()
                  and torch.isfinite(torch.tensor(mse)))
        print(f'{i:>5}  {s.get("h_absmax", float("nan")):>12.3e} '
              f'{str(s.get("h_finite")):>5}  '
              f'{student.abs().max().item():>12.3e} '
              f'{teacher.abs().max().item():>12.3e} {mse:>12.3e}  {str(ok):>4}')
        if first_bad is None and not ok:
            first_bad = i

    print()
    if first_bad is None:
        print('all layers finite -- the overflow is in the loss reduction, not the forward')
    else:
        print(f'first non-finite layer: {first_bad}')


if __name__ == '__main__':
    main()
