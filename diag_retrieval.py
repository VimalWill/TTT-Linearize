"""Which layers anchor in-context retrieval in the TTT memory?

Synthetic KV-retrieval, following the identification protocol in Bick, Xing & Gu
(arXiv:2504.18574, "Gather-and-Aggregate"): ablate a component, measure the drop
in retrieval, and read the drop as that component's share of the mechanism.
Synthetic is the point -- the task needs no world knowledge, so it isolates
retrieval from knowledge degradation. They find G&A concentrates in a few heads
of Llama-3.1-8B, at layers 16-17 (Gather L16H22, Aggregate L17H24); this scans
the linearized model at layer granularity, which is the granularity a fused
memory would share at.

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
    ap.add_argument('--seq-len', type=int, default=4096)
    ap.add_argument('--distances', type=int, nargs='+', default=[128, 1024, 3072])
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--batch', type=int, default=4)
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

    tok = AutoTokenizer.from_pretrained(args.base)
    pool, lk, lv = build_pool(tok, seed=args.seed)
    print(f'key {lk} tok, value {lv} tok, {len(pool)} usable pairs; '
          f'window_size = {window}')

    model = load_model(args.ckpt, model_config, args.adapter)
    mods = ttt_layers(model)
    print(f'{len(mods)} TTT layers found')
    targets = args.layers if args.layers is not None else list(range(len(mods)))

    branches = ['ttt', 'attn'] if args.branch == 'both' else [args.branch]

    # ---- build one batch per distance, shared across every ablation ----
    data = {}
    for d in args.distances:
        ids, ans, realised = build_batch(
            tok, pool, lk, lv, args.seq_len, d, args.samples, args.seed)
        data[d] = (ids, ans, realised)
        tag = 'inside window (control)' if realised <= window else 'beyond window'
        print(f'  distance {d:>5} -> {realised:>5} tokens, seq {ids.shape[1]}  [{tag}]')

    # ---- baseline ----
    print('\nbaseline, no ablation')
    base = {}
    for d in args.distances:
        ids, ans, _ = data[d]
        ce, acc = score(model, ids, ans, args.batch)
        base[d] = ce
        print(f'  distance {d:>5}   CE {ce:7.4f}   exact-match {acc:5.1%}')

    gates = gate_by_layer(model, mods, data[args.distances[0]][0], args.batch)

    # ---- whole-branch controls ----
    print('\nwhole-branch ablation (all layers at once)')
    for br in branches:
        with ablate(mods, br):
            row = []
            for d in args.distances:
                ids, ans, _ = data[d]
                ce, acc = score(model, ids, ans, args.batch)
                row.append(f'd{d}: CE {ce:7.3f} ({ce - base[d]:+6.3f}) acc {acc:4.0%}')
        print(f'  −{br:<4} ' + '   '.join(row))

    # ---- per-layer sweep ----
    results = {}
    for br in branches:
        print(f'\nper-layer, ablating {br} (CE delta vs baseline)')
        hdr = ''.join(f'{"d" + str(d):>10}' for d in args.distances)
        print(f'{"layer":>5} {"gate":>7}{hdr}')
        rows = []
        for li in targets:
            with ablate([mods[li]], br):
                deltas = []
                for d in args.distances:
                    ids, ans, _ = data[d]
                    ce, _ = score(model, ids, ans, args.batch)
                    deltas.append(ce - base[d])
            rows.append((li, gates[li], deltas))
            print(f'{li:>5} {gates[li]:>7.3f}' + ''.join(f'{x:>+10.3f}' for x in deltas))
        results[br] = rows

        far = args.distances[-1]
        ranked = sorted(rows, key=lambda r: -r[2][-1])
        print(f'\n  most anchored at d{far}:  ' +
              ', '.join(f'L{li}({d[-1]:+.2f})' for li, _, d in ranked[:6]))
        print(f'  least anchored at d{far}: ' +
              ', '.join(f'L{li}({d[-1]:+.2f})' for li, _, d in ranked[-6:]))
        open_gate = [r for r in ranked if r[1] > 0.05]
        if open_gate:
            print(f'  fusion candidates (gate > 0.05, smallest delta): ' +
                  ', '.join(f'L{li}' for li, _, _ in open_gate[-6:]))

    with open(f'{args.out}.csv', 'w') as f:
        f.write('branch,layer,gate,' + ','.join(f'dCE_{d}' for d in args.distances) + '\n')
        for br, rows in results.items():
            for li, g, ds in rows:
                f.write(f'{br},{li},{g:.5f},' + ','.join(f'{x:.5f}' for x in ds) + '\n')
    print(f'\nwrote {args.out}.csv')


if __name__ == '__main__':
    main()
