"""Layer-wise MMLU probe, in the Gather-and-Aggregate design.

Bick, Xing & Gu (arXiv:2504.18574) use MMLU as a retrieval probe rather than a
knowledge benchmark. Its format asks the model to emit a LETTER, so it must
retrieve which of A-D goes with the option it believes -- knowledge plus
in-context retrieval. Disabling one Gather or Aggregate head in Llama-3.1-8B
took MMLU from 66% to 25% (chance) while knowledge benchmarks held.

That contrast is the whole method, so this scores both halves:

  letter  question + lettered options + "Answer:" -> argmax over " A".." D"
          needs knowledge AND in-context letter retrieval
  cloze   question + "Answer:" -> argmax over the four option TEXTS,
          length-normalised. Needs the same knowledge, NO letter retrieval.

A layer whose ablation costs letter accuracy but not cloze accuracy is doing
retrieval. One that costs both is doing knowledge or general computation. The
AR-slice sweep in diag_retrieval.py cannot separate those, because a 0.006-nat
CE delta can hide a task-level cliff -- which is exactly what the G&A paper
demonstrates.

    python3 diag_mmlu.py --cfg Configs/ttt_ar_l2.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_l2/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_l2/best_ckpt --n 300

Prompts are RIGHT-padded, which is safe here and lets the sweep batch: attention
is causal so trailing pads cannot influence earlier positions, and the TTT
operator is apply-then-update, so the chunk holding the last real token is read
before any pad-polluted update lands.
"""

import argparse
import random

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from diag_common import load_model
from diag_retrieval import ablate, ttt_layers

LETTERS = ['A', 'B', 'C', 'D']


def build(tok, n, shots, seed):
    from datasets import load_dataset
    test = load_dataset('cais/mmlu', 'all', split='test')
    dev = load_dataset('cais/mmlu', 'all', split='dev')

    by_subj = {}
    for r in dev:
        by_subj.setdefault(r['subject'], []).append(r)

    rng = random.Random(seed)
    idx = list(range(len(test)))
    rng.shuffle(idx)

    def block(r, with_answer):
        s = r['question'].strip() + '\n'
        for L, c in zip(LETTERS, r['choices']):
            s += f'{L}. {c}\n'
        s += 'Answer:'
        if with_answer:
            s += f' {LETTERS[r["answer"]]}\n\n'
        return s

    items = []
    for i in idx:
        r = test[i]
        if len(r['choices']) != 4:
            continue
        head = (f'The following are multiple choice questions (with answers) '
                f'about {r["subject"].replace("_", " ")}.\n\n')
        fs = ''.join(block(d, True) for d in by_subj.get(r['subject'], [])[:shots])
        letter_prompt = head + fs + block(r, False)

        # cloze: same knowledge, no letters anywhere
        cloze_head = (f'The following are questions (with answers) about '
                      f'{r["subject"].replace("_", " ")}.\n\n')
        cloze_fs = ''.join(
            d['question'].strip() + '\nAnswer: ' + d['choices'][d['answer']] + '\n\n'
            for d in by_subj.get(r['subject'], [])[:shots])
        cloze_prompt = cloze_head + cloze_fs + r['question'].strip() + '\nAnswer:'

        items.append({
            'letter_ids': tok.encode(letter_prompt, add_special_tokens=True),
            'letter_cands': [tok.encode(f' {L}', add_special_tokens=False)[0]
                             for L in LETTERS],
            'cloze_ids': tok.encode(cloze_prompt, add_special_tokens=True),
            'cloze_cands': [tok.encode(f' {c.strip()}', add_special_tokens=False)
                            for c in r['choices']],
            'gold': r['answer'],
        })
        if len(items) >= n:
            break
    return items


def _pad(rows, pad_id):
    T = max(len(r) for r in rows)
    out = torch.full((len(rows), T), pad_id, dtype=torch.long)
    last = []
    for i, r in enumerate(rows):
        out[i, :len(r)] = torch.tensor(r)
        last.append(len(r) - 1)
    return out, torch.tensor(last)


@torch.no_grad()
def score_letter(model, items, batch, pad_id):
    hits = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids, last = _pad([c['letter_ids'] for c in chunk], pad_id)
        logits = model(input_ids=ids.cuda(), use_cache=False).logits
        for j, c in enumerate(chunk):
            lg = logits[j, last[j]].float()
            pick = max(range(4), key=lambda k: lg[c['letter_cands'][k]].item())
            hits += (pick == c['gold'])
        del logits
    return hits / len(items)


@torch.no_grad()
def score_cloze(model, items, batch, pad_id):
    """argmax over option texts, mean logprob per token (length-normalised)."""
    hits = 0
    flat = [(i, k, it['cloze_ids'] + it['cloze_cands'][k], len(it['cloze_cands'][k]))
            for i, it in enumerate(items) for k in range(4)]
    scores = {}
    for a in range(0, len(flat), batch):
        chunk = flat[a:a + batch]
        ids, last = _pad([c[2] for c in chunk], pad_id)
        logits = model(input_ids=ids.cuda(), use_cache=False).logits
        for j, (qi, k, seq, ln) in enumerate(chunk):
            # tokens [len-ln .. len-1] are the option; predicted from one earlier
            lo = last[j] - ln + 1
            lp = F.log_softmax(logits[j, lo - 1:last[j]].float(), dim=-1)
            tgt = torch.tensor(seq[lo:last[j] + 1])
            scores[(qi, k)] = lp.gather(-1, tgt.cuda().unsqueeze(-1)).mean().item()
        del logits
    for i, it in enumerate(items):
        pick = max(range(4), key=lambda k: scores[(i, k)])
        hits += (pick == it['gold'])
    return hits / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_ar_l2.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--n', type=int, default=300, help='test questions')
    ap.add_argument('--shots', type=int, default=5)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--branch', choices=['ttt', 'attn', 'both'], default='ttt')
    ap.add_argument('--layers', type=int, nargs='+', default=None)
    ap.add_argument('--group', type=int, nargs='+', default=None,
                    help='ablate these layers together as one condition')
    ap.add_argument('--cloze', action='store_true',
                    help='also run the cloze control per layer (4x slower)')
    ap.add_argument('--out', default='mmlu_anchor')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model_config = build_model_config(cfg)

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = build(tok, args.n, args.shots, args.seed)
    L = max(len(i['letter_ids']) for i in items)
    print(f'{len(items)} questions, {args.shots}-shot, longest prompt {L} tokens, '
          f'window {model_config.window_size}')

    model = load_model(args.ckpt, model_config, args.adapter)
    mods = ttt_layers(model)
    branches = ['ttt', 'attn'] if args.branch == 'both' else [args.branch]

    b_letter = score_letter(model, items, args.batch, pad_id)
    b_cloze = score_cloze(model, items, args.batch, pad_id) if args.cloze else None
    print(f'\nbaseline  letter {b_letter:6.2%}' +
          (f'   cloze {b_cloze:6.2%}' if args.cloze else '') +
          f'   (chance 25.00%)')
    if b_letter < 0.30:
        print('  WARNING: letter accuracy is at chance -- ablation cannot show '
              'anything. This probe needs a model that can do the task.')

    if args.group:
        for br in branches:
            with ablate([mods[i] for i in args.group], br):
                a = score_letter(model, items, args.batch, pad_id)
                c = score_cloze(model, items, args.batch, pad_id) if args.cloze else None
            line = f'  -{br:<4} layers {args.group}: letter {a:6.2%} ({a-b_letter:+.2%})'
            if args.cloze:
                line += f'   cloze {c:6.2%} ({c-b_cloze:+.2%})'
            print(line)
        return

    targets = args.layers if args.layers is not None else list(range(len(mods)))
    rows = {}
    for br in branches:
        print(f'\nper-layer, ablating {br}')
        hdr = f'{"layer":>5}{"letter":>9}{"d letter":>10}'
        if args.cloze:
            hdr += f'{"cloze":>9}{"d cloze":>10}{"retrieval":>11}'
        print(hdr)
        out = []
        for li in targets:
            with ablate([mods[li]], br):
                a = score_letter(model, items, args.batch, pad_id)
                c = score_cloze(model, items, args.batch, pad_id) if args.cloze else None
            line = f'{li:>5}{a:>9.2%}{a-b_letter:>+10.2%}'
            # retrieval score: letter degrades MORE than cloze -> retrieval layer
            rs = ((b_letter - a) - (b_cloze - c)) if args.cloze else None
            if args.cloze:
                line += f'{c:>9.2%}{c-b_cloze:>+10.2%}{rs:>+11.2%}'
            print(line)
            out.append((li, a, c, rs))
        rows[br] = out
        rank = sorted(out, key=lambda r: (r[3] if args.cloze else b_letter - r[1]),
                      reverse=True)
        print(f'  most retrieval-critical: ' +
              ', '.join(f'L{li}' for li, _, _, _ in rank[:6]))

    with open(f'{args.out}.csv', 'w') as f:
        f.write('branch,layer,letter,cloze,retrieval\n')
        # layer -1 is the unablated baseline; without it the file holds only
        # absolute accuracies and the deltas cannot be recovered downstream.
        f.write(f'baseline,-1,{b_letter:.5f},'
                f'{"" if b_cloze is None else f"{b_cloze:.5f}"},\n')
        for br, out in rows.items():
            for li, a, c, rs in out:
                f.write(f'{br},{li},{a:.5f},'
                        f'{"" if c is None else f"{c:.5f}"},'
                        f'{"" if rs is None else f"{rs:.5f}"}\n')
    print(f'\nwrote {args.out}.csv')


if __name__ == '__main__':
    main()
