"""Rescore retrieval validation from the saved samples, with no GPU.

`retrieval_validation` scores exact equality of the whole normalized string.
That is right for a single answer and wrong for an answer that is a SET: at
step 100 the model returned all five labels of an aggregation item in a
different order and scored 0.0. The capability was there and the metric could
not see it.

Every eval writes prediction and answer to validation_retrieval_samples.jsonl,
so the whole run can be rescored after the fact:

  exact        the original metric, kept so the change is visible
  set          comma-separated answers compared as sets, order-insensitive
  item recall  fraction of the answer's items found anywhere in the output
  degenerate   fraction of predictions that collapsed into a repeated token,
               which is a decoding failure rather than a retrieval one

    python rescore_retrieval.py --samples .../validation_retrieval_samples.jsonl
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text):
    text = re.sub(r'\s+', ' ', text.strip()).casefold()
    return re.sub(r'\s*,\s*', ',', text)


def items(text):
    return {piece for piece in normalize(text).split(',') if piece}


def closest(target, prediction):
    """Best similarity between an answer item and any token in the output.

    Exact-match scoring cannot tell "never retrieved" from "retrieved with one
    character wrong", and both appear as 0.000. At step 100 the model emitted
    label_2869797 for label_2869794 -- the right item, one digit off. Whether
    the twelve zero cells are full of such near misses or of unrelated text
    decides whether the memory is failing or only its precision is.
    """
    tokens = [t for t in re.split(r'[\s,]+', normalize(prediction)) if t]
    if not tokens:
        return 0.0
    return max(SequenceMatcher(None, target, t).ratio() for t in tokens)


def degenerate(prediction, diversity=0.2, dominance=0.6):
    """True when the decode collapsed, as in 'label_label_label...'.

    A collapsed decode says nothing about whether the memory holds the answer,
    so these are counted separately rather than scored as retrieval failures.

    Word frequency alone misses the observed case: the real collapse runs the
    prefix together with no separator, so the whole output splits into one
    token. Character n-gram diversity catches it -- a string cycling with
    period 6 has six distinct 8-grams however long it runs -- and the word
    test still covers the spaced variant.
    """
    text = normalize(prediction).replace(' ', '')
    if len(text) >= 32:
        grams = [text[i:i + 8] for i in range(len(text) - 7)]
        if len(set(grams)) / len(grams) < diversity:
            return True
    words = normalize(prediction).replace(',', ' ').split()
    return len(words) >= 8 and Counter(words).most_common(1)[0][1] / len(words) >= dominance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--step', type=int, default=None, help='default: the last step present')
    ap.add_argument('--show', type=int, default=0, help='print this many near-miss predictions')
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.samples).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit('no samples')
    steps = sorted({r['step'] for r in rows if r.get('step') is not None})
    step = a.step if a.step is not None else (steps[-1] if steps else None)
    rows = [r for r in rows if r.get('step') == step]
    if not rows:
        raise SystemExit(
            f'no samples at step {step}; this file holds {steps or "no steps"}. '
            'Runs before the append fix truncated the file at every eval, so '
            'only the newest step is present.')
    print(f'step {step} of {steps}; {len(rows)} samples\n')

    cells = defaultdict(list)
    for r in rows:
        cells[(r['task'], r['length_bucket'])].append(r)

    print(f'{"task":>14} {"len":>6} {"n":>3} {"exact":>7} {"set":>7} '
          f'{"item recall":>12} {"degenerate":>11} {"near miss":>10}')
    print(f'{"":>14} {"":>6} {"":>3} {"":>7} {"":>7} {"":>12} {"":>11} '
          f'{"(0-1, missed items only)":>10}')
    totals = defaultdict(list)
    for (task, length), group in sorted(cells.items()):
        exact = [float(normalize(r['prediction']) == normalize(r['answer'])) for r in group]
        seteq, recall, bad, near = [], [], [], []
        for r in group:
            want = items(r['answer'])
            got = items(r['prediction'])
            seteq.append(float(want == got))
            recall.append(len(want & got) / len(want) if want else 0.0)
            bad.append(float(degenerate(r['prediction'])))
            # Only items NOT recovered exactly: how close did the output get?
            missed = want - got
            near.append(sum(closest(w, r['prediction']) for w in missed) / len(missed)
                        if missed else float('nan'))
        m = lambda v: sum(v) / len(v)
        mn = [v for v in near if v == v]
        print(f'{task:>14} {length:>6} {len(group):>3} {m(exact):>7.3f} {m(seteq):>7.3f} '
              f'{m(recall):>12.3f} {m(bad):>11.0%} {(m(mn) if mn else float("nan")):>10.2f}')
        for key, value in (('exact', exact), ('set', seteq), ('recall', recall),
                           ('bad', bad), ('near', mn)):
            totals[key].extend(value)
    m = lambda v: sum(v) / len(v)
    print(f'\n{"OVERALL":>14} {"":>6} {len(totals["exact"]):>3} {m(totals["exact"]):>7.3f} '
          f'{m(totals["set"]):>7.3f} {m(totals["recall"]):>12.3f} {m(totals["bad"]):>11.0%} '
          f'{m(totals["near"]):>10.2f}')
    print('\n  near miss near 1.0: the item was retrieved and corrupted, so the memory '
          'holds it\n  near miss near 0.3: unrelated text, so the memory does not')

    if a.show:
        print('\nnear misses: the answer was partly recovered but scored zero exact')
        shown = 0
        for r in rows:
            want, got = items(r['answer']), items(r['prediction'])
            if not want or want == got or not (want & got):
                continue
            print(f"\n  {r['task']} {r['length_bucket']}  {len(want & got)}/{len(want)} items")
            print(f"    want: {r['answer'][:150]}")
            print(f"    got:  {r['prediction'][:150]}")
            shown += 1
            if shown >= a.show:
                break


if __name__ == '__main__':
    main()
