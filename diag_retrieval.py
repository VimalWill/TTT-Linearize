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

def ar_masks(ids, edges, window, chunk=None):
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

    # Chunk-crossing split. Apply-then-update means a query in chunk i reads
    # weights fitted on chunks 0..i-1, so a source in the query's OWN chunk is
    # invisible to the TTT branch by construction. That makes same-chunk hits a
    # structural control: any TTT ablation effect there cannot be retrieval, it
    # is the branch acting as a learned transform of q through the state.
    #
    # Raw distance does not separate these -- a hit at distance < 512 is
    # same-chunk or previous-chunk depending on the query's offset within its
    # chunk, so the 0-512 bucket is a mixture of both, roughly half each.
    if chunk:
        pos = torch.arange(T)
        src = pos - dist
        same = hit & ((pos // chunk) == (src.clamp(min=0) // chunk))
        masks['ar_same_chunk'] = same
        cross = hit & ~same
        pfx = 'ar_x'
    else:
        cross = hit
        pfx = 'ar_'

    lo = 0
    for hi in edges:
        name = f'{pfx}{lo}_{hi}' if hi != math.inf else f'{pfx}{lo}+'
        masks[name] = cross & (dist > lo) & (dist <= hi)
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
        logits = model(input_ids=chunk, use_cache=False).logits
        # loss on predicting token t comes from logits at t-1.
        # Chunked over positions: at 32k the full [b, T, 128256] in fp32 plus its
        # log_softmax is ~42 GB before the model's own 16 GB, which does not fit.
        # This caps the fp32 working set at STEP * vocab.
        STEP = 2048
        tgt = chunk[:, 1:]
        parts = []
        for a in range(0, tgt.shape[1], STEP):
            b_ = min(a + STEP, tgt.shape[1])
            lg = logits[:, a:b_, :].float()
            lp = F.log_softmax(lg, dim=-1)
            parts.append(-lp.gather(-1, tgt[:, a:b_].unsqueeze(-1)).squeeze(-1))
            del lg, lp
        nll = torch.cat(parts, dim=1)                            # [b, T-1]
        del logits, parts
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
    """branch: 'ttt' | 'attn' | 'both' | None, or a list of those."""
    if branch is None or not mods:
        yield
        return
    which = ['ttt', 'attn'] if branch == 'both' else (
        [branch] if isinstance(branch, str) else list(branch))
    attrs = ['_ablate_ttt' if b == 'ttt' else '_ablate_attn' for b in which]
    for m in mods:
        for a in attrs:
            setattr(m, a, True)
    try:
        yield
    finally:
        for m in mods:
            for a in attrs:
                setattr(m, a, False)


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
    """Mean |silu(ttt_scale_proj(h))| per layer -- is the branch even open?

    ABSOLUTE value, deliberately. These gates train negative, and silu is
    negative on (-inf, 0), so a signed mean cancels to ~0 and every layer looks
    shut regardless of how hard the branch is driving. Magnitude is what says
    whether the branch is on.
    """
    acc = [0.0] * len(mods)
    hooks = []

    def mk(i):
        def fn(_m, _inp, out):
            acc[i] += F.silu(out.detach().float()).abs().mean().item()
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
    ap.add_argument('--data-path', default=None,
                    help='ar-slice: override data.path, e.g. a Pile mirror such as '
                         'monology/pile-uncopyrighted, to match the corpus Based '
                         'Table 1 reports AR-slice perplexity on. The dataset must '
                         'expose a validation split.')
    ap.add_argument('--data-name', default=None,
                    help="ar-slice: override data.name. Must contain 'pg19', 'books' "
                         "or 'longtext' to select the long-form formatter in "
                         'Training/dataloader.py; use longtext for a Pile mirror.')
    ap.add_argument('--edges', type=int, nargs='+', default=[512, 2048],
                    help='ar-slice: distance bucket edges; a final open bucket is added')
    ap.add_argument('--seq-len', type=int, default=None,
                    help='context length. ar-slice: overrides data.max_length, so '
                         'the sweep can run beyond the 8192 it was trained at -- '
                         'nothing in this model sees a position past window_size, '
                         'so there is no RoPE extrapolation to worry about. '
                         'kv: prompt length (default 4096).')
    ap.add_argument('--distances', type=int, nargs='+', default=[128, 1024, 3072],
                    help='kv only')
    ap.add_argument('--samples', type=int, default=16, help='kv only')
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--branch', choices=['ttt', 'attn', 'both'], default='ttt')
    ap.add_argument('--layers', type=int, nargs='+', default=None)
    ap.add_argument('--joint', action='store_true',
                    help='per layer, run the 2x2 of {baseline, -ttt, -attn, -both} '
                         'and report the interaction term. Marginal single-branch '
                         'ablations do not add up when the branches interact, and '
                         'they measurably do -- at L25-27 removing SWA GAINS 0.12 '
                         'nats at >=8192 while removing TTT costs 0.11.')
    ap.add_argument('--group', type=int, nargs='+', default=None,
                    help='ablate these layers TOGETHER as one condition instead of '
                         'sweeping them one at a time. Single-layer ablation leaves '
                         'a depth-composed relay intact -- 32 stacked 512-token '
                         'windows reach ~16k -- so breaking the chain needs a '
                         'contiguous block, e.g. --group $(seq -s\' \' 8 15).')
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
        if args.data_path or args.data_name or args.seq_len:
            config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            if args.seq_len:
                config.model.max_length = args.seq_len
                print(f'context override: {args.seq_len} tokens')
            if args.data_path:
                config.data.path = args.data_path
            if args.data_name:
                config.data.name = args.data_name
            print(f'corpus override: {config.data.path} ({config.data.name})')
        else:
            print(f'corpus: {config.data.path} -- NOTE this is the training corpus; '
                  'pass --data-path for a number comparable to published AR-slice')
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
                masks.append(ar_masks(ids, edges, window,
                                                  model_config.lact_chunk_size)[0])
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
        print(f'{len(seqs)} sequences of {seqs[0].shape[0]} tokens; slices {names}')

        def run():
            ce, cnt = score_ar(model, seqs, masks, args.batch)
            return ce, cnt

        base, counts = run()
        print('\nbaseline, no ablation')
        for n in names:
            tag = ' (in-window, control)' if n.startswith('ar_0_') else ''
            if n == 'ar_same_chunk':
                tag = ' (TTT-blind by construction)'
            if not counts[n]:
                print(f'  {n:>16}  {"EMPTY -- no such tokens at this length":>40}')
                continue
            print(f'  {n:>16}  CE {base[n]:7.4f}  ppl {math.exp(base[n]):9.2f}  '
                  f'{counts[n]:>9,} tokens{tag}')
        # Drop empty slices: an unpopulated bucket makes every delta nan and
        # nan-poisons the anchor score and the whole ranking. ar_8192+ is empty
        # at an 8192 context, which is exactly when --seq-len was not applied.
        empty = [n for n in names if not counts[n]]
        if empty:
            print(f'  dropping empty slices: {empty}')
            names = [n for n in names if counts[n]]
        cross = [n for n in names if n not in ('other', 'ar_same_chunk')]
        if not cross:
            raise SystemExit('no populated cross-chunk AR slice -- nothing to measure')
        far = cross[-1]
        print(f'  far slice = {far}')
    else:
        tok = AutoTokenizer.from_pretrained(args.base)
        pool, lk, lv = build_pool(tok, seed=args.seed)
        print(f'key {lk} tok, value {lv} tok, {len(pool)} usable pairs')
        data = {}
        for d in args.distances:
            ids, ans, realised = build_batch(
                tok, pool, lk, lv, args.seq_len or 4096, d, args.samples, args.seed)
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

    # ------------------------------------------------ grouped ablation
    if args.group:
        print(f'\ngrouped ablation, layers {args.group} together')
        for br in branches:
            with ablate([mods[i] for i in args.group], br):
                ce, _ = run()
            print(f'  -{br:<4} ' + '  '.join(
                f'{n} {ce[n]:6.3f} ({ce[n] - base[n]:+6.3f})' for n in names))
        return

    # ------------------------------------------------ joint 2x2 per layer
    if args.joint:
        # The slice's unconditional entropy: an upper bound on how bad CE can
        # get by destroying context, used to detect saturation below.
        ceiling = max(v for k, v in base.items()) + 2.7
        print(f'\nper-layer 2x2 interaction at {far}   '
              f'(baseline {base[far]:.3f}, saturation guard {0.75 * ceiling:.2f})')
        print('  interaction = d(-both) - d(-ttt) - d(-attn)')
        print('  < 0 redundant (each substitutes) | > 0 complementary (both needed)')
        print(f'\n{"layer":>5}{"d(-ttt)":>10}{"d(-attn)":>10}{"d(-both)":>10}'
              f'{"interact":>11}  verdict')
        jrows = []
        for li in targets:
            d = {}
            for tag, br in (('ttt', 'ttt'), ('attn', 'attn'), ('both', 'both')):
                with ablate([mods[li]], br):
                    ce, _ = run()
                d[tag] = ce[far] - base[far]
            inter = d['both'] - d['ttt'] - d['attn']
            scale = max(abs(d['ttt']), abs(d['attn']), 1e-9)
            # Two guards, both learned from the first run.
            #
            # SATURATED: once a single-branch ablation pushes the slice near its
            # unconditional entropy, d(-both) physically cannot reach the sum of
            # the marginals and the interaction is forced negative regardless of
            # whether the branches substitute for each other. Measured at L0:
            # -ttt 5.95, -attn 6.15, -both 6.40 against a baseline of 3.39, and
            # an interaction of -2.30 that means nothing.
            #
            # NOISE: the verdict was relative to the largest marginal, so layers
            # whose marginals are all ~0.005 got labelled redundant or
            # complementary off pure noise. Require an absolute floor too.
            if base[far] + min(d.values()) > 0.75 * ceiling:
                v = 'SATURATED - interaction uninterpretable'
            elif abs(inter) < 0.004:
                v = 'independent'
            elif inter < -0.15 * scale:
                v = 'redundant'
            elif inter > 0.15 * scale:
                v = 'complementary'
            else:
                v = 'independent'
            print(f'{li:>5}{d["ttt"]:>+10.4f}{d["attn"]:>+10.4f}{d["both"]:>+10.4f}'
                  f'{inter:>+11.4f}  {v}')
            jrows.append((li, d, inter, v))
        with open(f'{args.out}_joint.csv', 'w') as f:
            f.write('layer,d_ttt,d_attn,d_both,interaction,verdict\n')
            for li, d, inter, v in jrows:
                f.write(f'{li},{d["ttt"]:.5f},{d["attn"]:.5f},{d["both"]:.5f},'
                        f'{inter:.5f},{v}\n')
        red = [li for li, _, _, v in jrows if v == 'redundant']
        comp = [li for li, _, _, v in jrows if v == 'complementary']
        print(f'\n  redundant (safe to collapse):     {red}')
        print(f'  complementary (keep both paths):  {comp}')
        print(f'\nwrote {args.out}_joint.csv')
        return

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
        # Rank fusion candidates by the FAR-slice delta, not by the gate.
        # Gate magnitude does not predict importance and filtering on it is
        # actively wrong: measured at 32k, L0 has the smallest gate in the model
        # (0.009) and the largest ablation cost (+2.55 at 8192+), while L31 has
        # the largest gate (0.078) and almost none (+0.016). What matters is
        # where the contribution lands in the residual stream, not how loud it
        # is -- an early layer's output propagates through every layer above it.
        by_far = sorted(rows, key=lambda r: r[2][far])
        print(f'  fusion candidates (smallest dCE at {far}): ' +
              ', '.join(f'L{li}({d[far]:+.3f})' for li, _, d, _ in by_far[:8]))
        print(f'  keep (largest dCE at {far}):                ' +
              ', '.join(f'L{li}({d[far]:+.3f})' for li, _, d, _ in by_far[-4:]))
        print('  gate is reported for information only -- it does not predict '
              'importance (L0: gate 0.009, dCE +2.55)')

    with open(f'{args.out}.csv', 'w') as f:
        f.write('branch,layer,gate,' + ','.join(f'dCE_{n}' for n in names) + ',anchor\n')
        for br, rows in results.items():
            for li, g, d, a in rows:
                f.write(f'{br},{li},{g:.5f},'
                        + ','.join(f'{d[n]:.5f}' for n in names) + f',{a:.5f}\n')
    print(f'\nwrote {args.out}.csv')


if __name__ == '__main__':
    main()
