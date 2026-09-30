"""Did the memory learn retrieval? Exact match by needle position, memory on vs off.

A global NIAH score cannot answer this. Measured on anchor_v2 (2026-09-28),
RULER niah_single at 2048 scored 0.2500 -- exactly 512/2048 -- and ablating all
32 TTT memories moved it by noise. The whole score was the sliding window
catching needles that happened to land inside it. Any new global number has the
same ambiguity: "the memory learned retrieval" and "the window got luckier" are
indistinguishable.

So bucket by depth and ablate. With window 512 at context 2048, a needle at
depth d sits (1-d) * ctx tokens from the query, so everything below depth 75%
is OUTSIDE the window and reachable only through the memory. The result that
matters is one cell:

    depth < 75%, memory ON minus memory ABLATED

A gain there is memory-borne retrieval. A flat delta means the memory still
contributes nothing, whatever the headline moved to.

`eval/loss` and AR perplexity are not substitutes -- CE and extraction rank in
opposite order across arms on this project.

    python eval_passkey_position.py \
      --cfg Configs/ttt_pk_anchor.yml \
      --ckpt  .../llama_v3/ttt_at_anchor_gate/best_ckpt \
      --adapter .../passkey/ttt_pk_anchor/best_ckpt \
      --skip 10200 --n 400 --ctx 2048
"""
import argparse
import collections
import json

import torch


def bucket(depth_percent, ctx, window):
    """Needle sits (1 - d) * ctx tokens before the query."""
    inside = (1.0 - depth_percent / 100.0) * ctx < window
    return 'inside_window' if inside else 'beyond_window'


@torch.no_grad()
def answer(model, tokenizer, prompt, max_new):
    ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    import itertools
    import LinearTTT  # noqa: F401
    from Training.train import build_model_config
    from Training.dataloader import PASSKEY_MARKER
    from eval import load_model, ttt_layers, ablate

    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True, help='stage-1 base')
    ap.add_argument('--adapter', required=True, help='the passkey stage-3 adapter')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--data', default='nanotron/llama3-16k-passkey-retrieval-finetuning')
    ap.add_argument('--skip', type=int, default=10200,
                    help='rows consumed by training; these must be UNSEEN keys')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--ctx', type=int, default=2048,
                    help='context length to test; 4096/8192 probe extrapolation '
                         'with no further training')
    ap.add_argument('--tol', type=int, default=128,
                    help='accept rows within +/- tol tokens of --ctx')
    ap.add_argument('--max-new', type=int, default=16)
    ap.add_argument('--out', default='passkey_position')
    a = ap.parse_args()

    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(a.cfg), resolve=True))
    cfg.model.pretrained_model_name_or_path = a.ckpt
    cfg.model.max_length = max(a.ctx + 256, int(cfg.model.max_length))
    model_config = build_model_config(cfg)
    window = int(model_config.window_size)
    model = load_model(a.ckpt, model_config, a.adapter)
    model.eval()
    mods = ttt_layers(model)
    tok = AutoTokenizer.from_pretrained(a.base)

    rows = []
    for row in itertools.islice(load_dataset(a.data, split='train', streaming=True), a.skip, None):
        if abs(row['num_tokens'] - a.ctx) > a.tol:
            continue
        rows.append(row)
        if len(rows) >= a.n:
            break
    if not rows:
        raise SystemExit(f'no rows within {a.tol} tokens of {a.ctx}')
    print(f'{len(rows)} unseen examples at ~{a.ctx} tokens, window {window}')
    print(f'needles below depth {100 * (1 - window / a.ctx):.0f}% are beyond the window\n')

    results = {}
    for label, branch in (('memory_on', None), ('memory_ablated', 'ttt')):
        hits = collections.defaultdict(lambda: [0, 0])
        with ablate(mods if branch else [], branch):
            for row in rows:
                prompt = row['prompt'].rsplit(PASSKEY_MARKER, 1)[0] + PASSKEY_MARKER
                got = answer(model, tok, prompt, a.max_new)
                ok = str(row['answer']) in got
                b = bucket(row['depth_percent'], a.ctx, window)
                hits[b][0] += ok
                hits[b][1] += 1
        results[label] = {k: (v[0], v[1]) for k, v in hits.items()}
        print(f'{label}')
        for b in ('beyond_window', 'inside_window'):
            if b in hits:
                h, n = hits[b]
                print(f'  {b:>14}  {h}/{n} = {h / n:.3f}')

    print('\n=== the cell that matters ===')
    for b in ('beyond_window', 'inside_window'):
        on = results['memory_on'].get(b)
        off = results['memory_ablated'].get(b)
        if not (on and off):
            continue
        d = on[0] / on[1] - off[0] / off[1]
        # binomial SE on the difference of two independent proportions
        import math
        p1, n1 = on[0] / on[1], on[1]
        p0, n0 = off[0] / off[1], off[1]
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        flag = 'MEMORY-BORNE' if d > 2 * se else 'within noise'
        print(f'  {b:>14}  on {p1:.3f}  ablated {p0:.3f}  delta {d:+.3f} '
              f'+/- {se:.3f}  -> {flag}')

    path = f'{a.out}_{a.ctx}.json'
    with open(path, 'w') as f:
        json.dump({'ctx': a.ctx, 'window': window, 'n': len(rows),
                   'ckpt': a.ckpt, 'adapter': a.adapter, 'results': results}, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
