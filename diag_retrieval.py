"""Which layers anchor in-context retrieval in the TTT memory?

Ablation protocol from Bick, Xing & Gu (arXiv:2504.18574, "Gather-and-Aggregate"):
ablate a component, measure the drop in retrieval, read the drop as that
component's share of the mechanism. They find G&A concentrates in a few heads of
Llama-3.1-8B at layers 16-17 (Gather L16H22, Aggregate L17H24); this scans the
linearized model at layer granularity, which is the granularity a fused memory
would share at.

Two tasks:

`--task ar-slice` (default) is the standard recall metric from Zoology / Based
(Arora et al., arXiv:2402.18668, Table 1): split real text into *associative
recall* tokens -- those completing a bigram that already occurred earlier in the
same context -- and *other* tokens, and report perplexity on each slice. It is
the right instrument here for three reasons: it is graded and non-generative, so
a 32-layer sweep is affordable; the distance to the earlier occurrence gives the
distance axis for free on real text; and the non-AR slice is a control that
separates "this layer does retrieval" from "this layer matters for everything" --
the analogue of the G&A paper holding knowledge benchmarks fixed while MMLU
collapses.

`--task kv` is a bespoke synthetic key-value retrieval with exact distance
control. Useful as a sanity check on the harness -- retrieval inside the window
must survive TTT ablation and retrieval beyond it must not -- but it is not a
standard benchmark, so it should not carry a headline number.

The axis this architecture uniquely has is the branch split. Each layer is
`attn_out + ttt_out`, so ablating one branch at a time says whether a layer's
retrieval rides on the 512-token window or on the memory. DISTANCE IS THE POINT:
retrieval inside the window is a window job and says nothing about the memory,
so anything at distance <= window_size is a control, not a measurement.

    python3 diag_retrieval.py --cfg Configs/ttt_ar_l2.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_l2/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_l2/best_ckpt

Read the output as: a large +dCE at distance >> window with TTT ablated means
that layer anchors long-range retrieval in the memory. A layer with ~0 there is
a fusion candidate -- but check its gate first: a nearly-closed gate shows no
ablation effect because the branch is already off, not because the layer has no
demand for it.
"""

import argparse
import contextlib
import math
import random

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from Training.dataloader import load_data
from diag_common import load_model


# ---------------------------------------------------------------- task

def build_pool(tok, n=4000, seed=0):
    """4-digit keys and values whose token length is fixed.

    Equal-length lines make every distance an exact token count and let samples
    batch together, so the sweep is one forward per condition rather than one
    per sample.
    """
    rng = random.Random(seed)
    cand = [f'{i:04d}' for i in range(1000, 9999)]
    rng.shuffle(cand)
    key_len, val_len = {}, {}
    for c in cand:
        key_len.setdefault(len(tok.encode(f'\n{c}:', add_special_tokens=False)), []).append(c)
        val_len.setdefault(len(tok.encode(f' {c}', add_special_tokens=False)), []).append(c)
    lk = max(key_len, key=lambda k: len(key_len[k]))
    lv = max(val_len, key=lambda k: len(val_len[k]))
    keys = [c for c in key_len[lk] if c in set(val_len[lv])]
    if len(keys) < n:
        n = len(keys)
    return keys[:n], lk, lv


def build_batch(tok, pool, lk, lv, seq_len, distance, n_samples, seed):
    """-> input_ids [n, T], answer_ids [n, lv], realised distance."""
    rng = random.Random(seed)
    per_line = lk + lv
    n_lines = seq_len // per_line
    # target line index counted from the end; distance is measured from the
    # start of the target line to the end of the prompt
    idx_from_end = max(1, min(n_lines, round(distance / per_line)))
    realised = idx_from_end * per_line

    rows, answers = [], []
    for s in range(n_samples):
        ks = rng.sample(pool, n_lines)
        vs = [rng.choice(pool) for _ in range(n_lines)]
        tgt = n_lines - idx_from_end
        ids = []
        for k, v in zip(ks, vs):
            ids += tok.encode(f'\n{k}:', add_special_tokens=False)
            ids += tok.encode(f' {v}', add_special_tokens=False)
        ans = tok.encode(f' {vs[tgt]}', add_special_tokens=False)
        ids += tok.encode(f'\n{ks[tgt]}:', add_special_tokens=False)
        rows.append(ids + ans)
        answers.append(ans)
    T = len(rows[0])
    assert all(len(r) == T for r in rows), 'ragged batch -- token lengths drifted'
    return (torch.tensor(rows), torch.tensor(answers), realised)


# ------------------------------------------------------- associative recall

def ar_masks(ids, edges, window):
    """Split a sequence into AR-hit slices by distance, plus a non-AR control.

    A token at position t is an associative-recall hit if the bigram
    (ids[t-1], ids[t]) occurred earlier in the same sequence; its distance is
    how far back that earlier occurrence was. This is the Zoology / Based
    AR-slice split, bucketed by distance so the window and the memory can be
    told apart -- hits closer than window_size are servable by attention alone.

    ids: [T] on cpu. -> dict name -> bool mask [T], aligned to ids (position t
    means "the loss on predicting ids[t]").
    """
    T = ids.shape[0]
    last = {}
    dist = torch.zeros(T, dtype=torch.long)
    for t in range(1, T):
        bg = (int(ids[t - 1]), int(ids[t]))
        prev = last.get(bg)
        if prev is not None:
            dist[t] = t - prev
        last[bg] = t

    masks = {}
    hit = dist > 0
    lo = 0
    for hi in edges:
        name = f'ar_{lo}_{hi}' if hi != math.inf else f'ar_{lo}+'
        masks[name] = hit & (dist > lo) & (dist <= hi)
        lo = hi
    masks['other'] = ~hit
    masks['other'][0] = False
    return masks, window


@torch.no_grad()
def score_ar(model, seqs, masks, batch):
    """Mean CE per slice over a list of sequences. -> dict name -> ce"""
    names = list(masks[0].keys())
    tot = {n: 0.0 for n in names}
    cnt = {n: 0 for n in names}
    for i in range(0, len(seqs), batch):
        chunk = torch.stack(seqs[i:i + batch]).cuda()
        logits = model(input_ids=chunk, use_cache=False).logits.float()
        # loss on predicting token t comes from logits at t-1
        lp = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = chunk[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)     # [b, T-1]
        for j in range(chunk.shape[0]):
            m = masks[i + j]
            for n in names:
                sel = m[n][1:].cuda()
                k = int(sel.sum())
                if k:
                    tot[n] += float(nll[j][sel].sum())
                    cnt[n] += k
    return {n: (tot[n] / cnt[n] if cnt[n] else float('nan')) for n in names}, cnt


# ---------------------------------------------------------------- scoring

def ttt_layers(model):
    base = getattr(model, 'model', model)
    base = getattr(base, 'model', base)
    layers = getattr(base, 'layers', None)
    if layers is not None:
        out = [l.self_attn for l in layers]
        if all(type(m).__name__ == 'LinearTTTAttention' for m in out):
            return out
    return [m for m in model.modules() if type(m).__name__ == 'LinearTTTAttention']


@contextlib.contextmanager
def ablate(mods, branch):
    """branch: 'ttt' | 'attn' | None"""
    if branch is None:
        yield
        return
    attr = '_ablate_ttt' if branch == 'ttt' else '_ablate_attn'
    for m in mods:
        setattr(m, attr, True)
    try:
        yield
    finally:
        for m in mods:
            setattr(m, attr, False)


@torch.no_grad()
def score(model, ids, answers, batch):
    """CE over the answer tokens, and exact-match accuracy. -> (ce, acc)"""
    n, T = ids.shape
    la = answers.shape[1]
    tot_ce, hits = 0.0, 0
    for i in range(0, n, batch):
        chunk = ids[i:i + batch].cuda()
        ans = answers[i:i + batch].cuda()
        logits = model(input_ids=chunk, use_cache=False).logits.float()
        # answer token t is predicted from position T-la+t-1
        pred = logits[:, T - la - 1:T - 1, :]
        tot_ce += F.cross_entropy(
            pred.reshape(-1, pred.shape[-1]), ans.reshape(-1), reduction='sum'
        ).item()
        hits += (pred.argmax(-1) == ans).all(dim=1).sum().item()
    return tot_ce / (n * la), hits / n


def gate_by_layer(model, mods, ids, batch):
    """Mean silu(ttt_scale_proj(h)) per layer -- is the branch even open?"""
    acc = [0.0] * len(mods)
    hooks = []

    def mk(i):
        def fn(_m, _inp, out):
            acc[i] += F.silu(out.detach().float()).mean().item()
        return fn

    for i, m in enumerate(mods):
        hooks.append(m.ttt_scale_proj.register_forward_hook(mk(i)))
    n_calls = 0
    with torch.no_grad():
        for i in range(0, min(batch, ids.shape[0]), batch):
            model(input_ids=ids[i:i + batch].cuda(), use_cache=False)
            n_calls += 1
    for h in hooks:
        h.remove()
    return [a / max(n_calls, 1) for a in acc]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_ar_l2.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', default=None, help='stage-2 output dir')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--task', choices=['ar-slice', 'kv'], default='ar-slice')
    ap.add_argument('--seqs', type=int, default=8, help='ar-slice: validation sequences')
    ap.add_argument('--edges', type=int, nargs='+', default=[512, 2048],
                    help='ar-slice: distance bucket edges; a final open bucket is added')
    ap.add_argument('--seq-len', type=int, default=4096, help='kv only')
    ap.add_argument('--distances', type=int, nargs='+', default=[128, 1024, 3072],
                    help='kv only')
    ap.add_argument('--samples', type=int, default=16, help='kv only')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--branch', choices=['ttt', 'attn', 'both'], default='ttt')
    ap.add_argument('--layers', type=int, nargs='+', default=None)
    ap.add_argument('--out', default='retrieval_anchor')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model_config = build_model_config(cfg)
    window = model_config.window_size
    print(f'window_size = {window}, chunk = {model_config.lact_chunk_size}')

    model = load_model(args.ckpt, model_config, args.adapter)
    mods = ttt_layers(model)
    print(f'{len(mods)} TTT layers found')
    targets = args.layers if args.layers is not None else list(range(len(mods)))
    branches = ['ttt', 'attn'] if args.branch == 'both' else [args.branch]

    # ------------------------------------------------ assemble the task
    if args.task == 'ar-slice':
        loader = load_data(config)['validation']
        seqs, masks = [], []
        edges = list(args.edges) + [math.inf]
        for b in loader:
            # take every row, not just the first: the loader's batch size is the
            # config's micro_batch_size and may be > 1
            for row in b['input_ids']:
                if len(seqs) >= args.seqs:
                    break
                ids = row.cpu()
                seqs.append(ids)
                masks.append(ar_masks(ids, edges, window)[0])
            if len(seqs) >= args.seqs:
                break
        if not seqs:
            raise RuntimeError('validation loader yielded nothing')
        if len({s.shape[0] for s in seqs}) != 1:
            raise RuntimeError(
                f'ragged validation sequences {sorted({s.shape[0] for s in seqs})}; '
                'score_ar stacks them, so they must be equal length'
            )
        names = list(masks[0].keys())
        far = [n for n in names if n != 'other'][-1]
        print(f'{len(seqs)} sequences of {seqs[0].shape[0]} tokens; '
              f'slices {names}, far slice = {far}')

        def run():
            ce, cnt = score_ar(model, seqs, masks, args.batch)
            return ce, cnt

        base, counts = run()
        print('\nbaseline, no ablation')
        for n in names:
            tag = ' (in-window, control)' if n.startswith('ar_0_') else ''
            print(f'  {n:>14}  CE {base[n]:7.4f}  ppl {math.exp(base[n]):9.2f}  '
                  f'{counts[n]:>9,} tokens{tag}')
    else:
        tok = AutoTokenizer.from_pretrained(args.base)
        pool, lk, lv = build_pool(tok, seed=args.seed)
        print(f'key {lk} tok, value {lv} tok, {len(pool)} usable pairs')
        data = {}
        for d in args.distances:
            ids, ans, realised = build_batch(
                tok, pool, lk, lv, args.seq_len, d, args.samples, args.seed)
            data[d] = (ids, ans, realised)
            tag = 'inside window (control)' if realised <= window else 'beyond window'
            print(f'  distance {d:>5} -> {realised:>5} tokens  [{tag}]')
        names = [f'd{d}' for d in args.distances]
        far = names[-1]

        def run():
            ce, cnt = {}, {}
            for d in args.distances:
                ids, ans, _ = data[d]
                c, acc = score(model, ids, ans, args.batch)
                ce[f'd{d}'] = c
                cnt[f'd{d}'] = acc
            return ce, cnt

        base, accs = run()
        print('\nbaseline, no ablation')
        for d in args.distances:
            print(f'  distance {d:>5}   CE {base[f"d{d}"]:7.4f}   '
                  f'exact-match {accs[f"d{d}"]:5.1%}')

    gate_ids = (torch.stack(seqs[:args.batch]) if args.task == 'ar-slice'
                else data[args.distances[0]][0])
    gates = gate_by_layer(model, mods, gate_ids, args.batch)

    # ------------------------------------------------ whole-branch control
    print('\nwhole-branch ablation (all layers at once)')
    for br in branches:
        with ablate(mods, br):
            ce, _ = run()
        print(f'  -{br:<4} ' + '  '.join(
            f'{n} {ce[n]:6.3f} ({ce[n] - base[n]:+6.3f})' for n in names))

    # ------------------------------------------------ per-layer sweep
    results = {}
    ctrl = 'other' if args.task == 'ar-slice' else None
    for br in branches:
        print(f'\nper-layer, ablating {br}   (CE delta vs baseline)')
        hdr = ''.join(f'{n:>13}' for n in names)
        anch = f'{"anchor":>9}' if ctrl else ''
        print(f'{"layer":>5} {"gate":>6}{hdr}{anch}')
        rows = []
        for li in targets:
            with ablate([mods[li]], br):
                ce, _ = run()
            d = {n: ce[n] - base[n] for n in names}
            # retrieval-specific damage: how much MORE the far recall slice
            # degrades than the non-recall control. The G&A analogue of MMLU
            # collapsing while knowledge benchmarks hold.
            a = (d[far] - d[ctrl]) if ctrl else d[far]
            rows.append((li, gates[li], d, a))
            print(f'{li:>5} {gates[li]:>6.3f}' + ''.join(f'{d[n]:>+13.4f}' for n in names)
                  + (f'{a:>+9.4f}' if ctrl else ''))
        results[br] = rows

        ranked = sorted(rows, key=lambda r: -r[3])
        lbl = 'anchor score' if ctrl else f'dCE @ {far}'
        print(f'\n  most anchored ({lbl}):  ' +
              ', '.join(f'L{li}({a:+.3f})' for li, _, _, a in ranked[:6]))
        print(f'  least anchored:          ' +
              ', '.join(f'L{li}({a:+.3f})' for li, _, _, a in ranked[-6:]))
        openg = [r for r in ranked if r[1] > 0.05]
        closed = [r for r in ranked if r[1] <= 0.05]
        if openg:
            print(f'  fusion candidates (gate open, lowest anchor): ' +
                  ', '.join(f'L{li}' for li, _, _, _ in openg[-6:]))
        if closed:
            print(f'  gate <= 0.05, ablation says nothing about demand: ' +
                  ', '.join(f'L{li}' for li, _, _, _ in closed))

    with open(f'{args.out}.csv', 'w') as f:
        f.write('branch,layer,gate,' + ','.join(f'dCE_{n}' for n in names) + ',anchor\n')
        for br, rows in results.items():
            for li, g, d, a in rows:
                f.write(f'{br},{li},{g:.5f},'
                        + ','.join(f'{d[n]:.5f}' for n in names) + f',{a:.5f}\n')
    print(f'\nwrote {args.out}.csv')


if __name__ == '__main__':
    main()
