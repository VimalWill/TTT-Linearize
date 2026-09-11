
import argparse
import contextlib
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window

_PEFT_PREFIX = re.compile(r'^base_model\.model\.')


def load_ttt_params(model, adapter, verbose=True):
    """Overlay `adapter/ttt_params.pt` onto an *unwrapped* model.

    ttt_params.pt is written from a PeftModel, so its keys carry peft's wrapper
    prefix and load into an unwrapped model matching NOTHING, which strict=False
    hides. Any stage-2 measurement was then really stage-1 TTT weights with
    stage-2 adapters. Hence the prefix strip and the raise on zero matches --
    both are load-bearing, do not drop them.
    """
    ttt = os.path.join(adapter, 'ttt_params.pt')
    if not os.path.exists(ttt):
        if verbose:
            print('  NOTE: no ttt_params.pt -- stage-2 TTT weights were never saved')
        return 0
    sd = torch.load(ttt, map_location='cpu')
    sd = {_PEFT_PREFIX.sub('', k): v for k, v in sd.items()}
    unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
    matched = len(sd) - len(unexpected)
    if matched == 0:
        raise RuntimeError(
            f'{ttt} holds {len(sd)} tensors but none match this model. '
            f'First saved key after prefix strip: {next(iter(sd))}'
        )
    if verbose:
        print(f'  overlaid {matched}/{len(sd)} saved TTT tensors')
        if unexpected:
            print(f'  WARNING: {len(unexpected)} unmatched, e.g. {unexpected[0]}')
    return matched


def load_model(path, model_config, adapter=None, verbose=True):
    """Full checkpoint, optionally with PEFT adapters and saved TTT params."""
    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16)
    if adapter:
        from peft import PeftModel
        load_ttt_params(model, adapter, verbose=verbose)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval()


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


# ------------------------------------------------------------------ AR slices

def ar_masks(ids, edges, chunk=None):
    """Split a sequence into associative-recall slices by retrieval distance.

    A token at position t is an AR hit if the bigram (ids[t-1], ids[t]) occurred
    earlier in the same sequence; its distance is how far back. This is the
    Zoology / Based AR split, bucketed so the window and the memory can be told
    apart -- hits closer than window_size are servable by attention alone.

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
    # invisible to the TTT branch by construction -- same-chunk hits are a
    # structural control. Raw distance does not separate these: a hit at
    # distance < chunk is same- or previous-chunk depending on the query's
    # offset, so that bucket is a mixture of both.
    if chunk:
        # The mask is aligned to the target token t, but its CE comes from
        # the logit at query position t-1.  Use that position for both sides
        # of the comparison; using t misclassifies targets at chunk starts.
        query = torch.arange(T) - 1
        src = query - dist
        same = hit & ((query // chunk) == (src.clamp(min=0) // chunk))
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
    return masks


@torch.no_grad()
def score_ar(model, seqs, masks, batch):
    """Mean CE per slice. -> (means, counts, per_seq).

    per_seq[name][i] is sequence i's own mean CE on that slice, or nan if that
    sequence has no token there. Slice membership does not depend on ablation,
    so these line up across runs and callers can form PAIRED deltas -- which
    cancel per-sequence difficulty and give a real standard error instead of a
    pooled point estimate. At ~0.02 nats per layer the pairing is what makes
    the effect resolvable at all.
    """
    names = list(masks[0].keys())
    tot = {n: 0.0 for n in names}
    cnt = {n: 0 for n in names}
    ps = {n: [float('nan')] * len(seqs) for n in names}
    dev = model.device
    for i in range(0, len(seqs), batch):
        ids = torch.stack(seqs[i:i + batch]).to(dev)
        logits = model(input_ids=ids, use_cache=False).logits
        # Loss on predicting token t comes from logits at t-1. Chunked over
        # positions: at 32k the full [b, T, 128256] in fp32 plus its log_softmax
        # is ~42 GB on top of the model's own 16 GB, which does not fit.
        STEP = 2048
        tgt = ids[:, 1:]
        parts = []
        for a in range(0, tgt.shape[1], STEP):
            b_ = min(a + STEP, tgt.shape[1])
            lp = F.log_softmax(logits[:, a:b_, :].float(), dim=-1)
            parts.append(-lp.gather(-1, tgt[:, a:b_].unsqueeze(-1)).squeeze(-1))
            del lp
        nll = torch.cat(parts, dim=1)                            # [b, T-1]
        del logits, parts
        for j in range(ids.shape[0]):
            m = masks[i + j]
            for n in names:
                sel = m[n][1:].to(dev)
                k = int(sel.sum())
                if k:
                    v = float(nll[j][sel].sum())
                    tot[n] += v
                    cnt[n] += k
                    ps[n][i + j] = v / k
    means = {n: (tot[n] / cnt[n] if cnt[n] else float('nan')) for n in names}
    return means, cnt, ps


def paired(a, b):
    """Mean and standard error of the paired difference a - b, nan-skipping."""
    d = [x - y for x, y in zip(a, b)
         if x == x and y == y]                                # drop nan pairs
    n = len(d)
    if n == 0:
        return float('nan'), float('nan'), 0
    m = sum(d) / n
    if n < 2:
        return m, float('nan'), n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, (var / n) ** 0.5, n


@torch.no_grad()
def gate_by_layer(model, mods, ids, batch):
    """Mean |silu(ttt_scale_proj(h))| per layer -- is the branch even open?

    ABSOLUTE value, deliberately. These gates train negative and silu is
    negative on (-inf, 0), so a signed mean cancels to ~0 and every layer looks
    shut regardless of how hard the branch is driving.
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
    for i in range(0, min(batch, ids.shape[0]), batch):
        model(input_ids=ids[i:i + batch].to(model.device), use_cache=False)
        n_calls += 1
    for h in hooks:
        h.remove()
    return [a / max(n_calls, 1) for a in acc]


def run_retrieval(args, model, model_config, config, mods):
    """Per-layer x per-slice ablation sweep -> one tidy CSV, ready to plot."""
    from Training.dataloader import load_data

    chunk = model_config.lact_chunk_size
    print(f'window_size = {model_config.window_size}, chunk = {chunk}')

    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if args.seq_len:
        cfg.model.max_length = args.seq_len
        print(f'context override: {args.seq_len} tokens')
    if args.data_path:
        cfg.data.path = args.data_path
    if args.data_name:
        cfg.data.name = args.data_name
    # 2x headroom: the loader budgets characters, not tokens
    cfg.data.num_val_seqs = 2 * args.seqs
    if not (args.data_path or args.data_name):
        print(f'corpus: {cfg.data.path} -- NOTE this is the training corpus; '
              'pass --data-path for a number comparable to published AR-slice')

    edges = list(args.edges) + [math.inf]
    seqs, masks = [], []
    for b in load_data(cfg)['validation']:
        # every row, not just the first: the loader's batch is micro_batch_size
        for row in b['input_ids']:
            if len(seqs) >= args.seqs:
                break
            ids = row.cpu()
            seqs.append(ids)
            masks.append(ar_masks(ids, edges, chunk))
        if len(seqs) >= args.seqs:
            break
    if not seqs:
        raise RuntimeError('validation loader yielded nothing')
    lens = {s.shape[0] for s in seqs}
    if len(lens) != 1:
        raise RuntimeError(f'ragged validation sequences {sorted(lens)}; '
                           'score_ar stacks them, so they must be equal length')
    names = list(masks[0].keys())
    print(f'{len(seqs)} sequences of {seqs[0].shape[0]} tokens; slices {names}')

    run = lambda: score_ar(model, seqs, masks, args.ar_batch)

    base, counts, base_ps = run()
    print('\nbaseline, no ablation')
    for n in names:
        if not counts[n]:
            print(f'  {n:>16}  EMPTY -- no such tokens at this length')
            continue
        print(f'  {n:>16}  CE {base[n]:7.4f}  ppl {math.exp(base[n]):9.2f}  '
              f'{counts[n]:>9,} tokens')
    # An unpopulated bucket makes every delta nan and nan-poisons the sweep.
    # The far bucket is empty whenever --seq-len was not applied.
    empty = [n for n in names if not counts[n]]
    if empty:
        print(f'  dropping empty slices: {empty}')
        names = [n for n in names if counts[n]]

    gates = gate_by_layer(model, mods, torch.stack(seqs[:args.ar_batch]),
                          args.ar_batch)

    branches = ['ttt', 'attn'] if args.branch == 'both' else [args.branch]
    targets = args.layers if args.layers is not None else list(range(len(mods)))

    print('\nwhole-branch ablation (all layers at once)')
    whole = {}
    for br in branches:
        with ablate(mods, br):
            whole[br] = run()[0]
        print(f'  -{br:<4} ' + '  '.join(
            f'{n} {whole[br][n]:6.3f} ({whole[br][n] - base[n]:+6.3f})' for n in names))

    group = {}
    if args.group:
        bad = [i for i in args.group if not 0 <= i < len(mods)]
        if bad:
            raise SystemExit(f'--group layer(s) out of range for {len(mods)} layers: {bad}')
        print(f'\ngrouped ablation, layers {args.group[0]}-{args.group[-1]} '
              f'({len(args.group)} layers) together')
        for br in branches:
            with ablate([mods[i] for i in args.group], br):
                group[br] = run()[0]
            print(f'  -{br:<4} ' + '  '.join(
                f'{n} {group[br][n]:6.3f} ({group[br][n] - base[n]:+6.3f})' for n in names))

    rows = []
    for br in branches:
        print(f'\nper-layer, ablating {br}   (paired dCE +- SE over {len(seqs)} sequences)')
        print(f'{"layer":>5} {"gate":>6}' + ''.join(f'{n:>20}' for n in names))
        for li in targets:
            with ablate([mods[li]], br):
                _, _, ps = run()
            d, se = {}, {}
            for n in names:
                d[n], se[n], _ = paired(ps[n], base_ps[n])
            rows.append((br, li, gates[li], d, se))
            print(f'{li:>5} {gates[li]:>6.3f}'
                  + ''.join(f'{d[n]:>+13.4f}+-{se[n]:5.4f}' for n in names))

    path = f'{args.out}_retrieval.csv'
    n_pair = {n: paired(base_ps[n], base_ps[n])[2] for n in names}
    with open(path, 'w') as f:
        f.write('branch,layer,gate,'
                + ','.join(f'dCE_{n},se_{n}' for n in names) + '\n')
        # layer -1 is the unablated baseline in ABSOLUTE CE. Without it the file
        # holds only deltas, and deltas from two models are not comparable -- a
        # smaller delta can mean "this branch matters less" or "this model is
        # stronger and more redundant", and only the baseline separates them.
        # the se_ column on these rows carries n, not a standard error
        f.write('baseline,-1,,'
                + ','.join(f'{base[n]:.5f},{n_pair[n]}' for n in names) + '\n')
        f.write('count,-3,,'
                + ','.join(f'{counts[n]},{n_pair[n]}' for n in names) + '\n')
        # whole-branch rows as ABSOLUTE CE, not deltas: on a base checkpoint the
        # unablated baseline is polluted by the randomly initialised memory, so
        # `whole_-ttt` at a full-length window IS the clean attention reference.
        for br, ce in whole.items():
            f.write(f'whole_-{br},-2,,'
                    + ','.join(f'{ce[n]:.5f},' for n in names) + '\n')
        # layer -4, ABSOLUTE CE like the whole-branch rows. This is the only
        # row that measures a shared memory rather than one layer's readout.
        for br, ce in group.items():
            f.write(f'group_-{br},-4,,'
                    + ','.join(f'{ce[n]:.5f},' for n in names) + '\n')
        for br, li, g, d, se in rows:
            f.write(f'{br},{li},{g:.5f},'
                    + ','.join(f'{d[n]:.5f},{se[n]:.5f}' for n in names) + '\n')
    print(f'\nwrote {path}')


SUITES = {
    'recall': ['swde', 'fda', 'squad_completion'],
    'commonsense': ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag',
                    'winogrande', 'lambada_openai'],
    'mmlu': ['mmlu'],
}


def resolve(tasks):
    """Check task names against the installed registry before spending a load."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    have = set(getattr(tm, 'all_tasks', None) or tm.task_index.keys())
    missing = [t for t in tasks if t not in have]
    if missing:
        hint = sorted(h for h in have
                      if any(m.split('_')[0] in h for m in missing))[:25]
        raise SystemExit(
            f'lm_eval does not register: {missing}\n'
            f'closest available: {hint}\n'
            'Task ids move between lm_eval versions -- pass --tasks explicitly.'
        )
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='Configs/ttt_ar_l2.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--suite', nargs='+', default=['recall'],
                    choices=sorted(SUITES), help='which predefined suites to run')
    ap.add_argument('--tasks', nargs='+', default=None,
                    help='explicit lm_eval task ids, overrides --suite')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap examples per task -- required for a layer sweep')
    ap.add_argument('--num-fewshot', type=int, default=None)
    ap.add_argument('--batch-size', default='4')
    ap.add_argument('--ablate', choices=['ttt', 'attn'], default=None)
    ap.add_argument('--layers', type=int, nargs='+', default=None,
                    help='layers to ablate; omit with --ablate for whole-branch')
    ap.add_argument('--out', default='eval')
    # ---- AR-slice retrieval sweep (does not use lm_eval) ----
    ap.add_argument('--retrieval', action='store_true',
                    help='per-layer x retrieval-distance ablation sweep -> CSV')
    ap.add_argument('--seqs', type=int, default=32,
                    help='retrieval: validation sequences to score')
    ap.add_argument('--edges', type=int, nargs='+',
                    default=[512, 2048, 8192, 16384],
                    help='retrieval: distance bucket upper edges; +inf appended')
    ap.add_argument('--seq-len', type=int, default=None,
                    help='override eval context length, both paths. REQUIRED for '
                         'a long-context suite: config max_length is 8192, so '
                         'without this RULER at 32k silently truncates to 8k. '
                         'Retrieval: run at the TRAINED length -- past it both '
                         'shared and per-layer decay, so a longer sweep measures '
                         'extrapolation, not retrieval')
    ap.add_argument('--data-path', default=None,
                    help='retrieval: override corpus (default is the TRAINING corpus)')
    ap.add_argument('--data-name', default=None)
    ap.add_argument('--ar-batch', type=int, default=2,
                    help='retrieval: sequences per forward')
    ap.add_argument('--branch', choices=['ttt', 'attn', 'both'], default='both',
                    help='retrieval: which branch(es) to ablate per layer')
    ap.add_argument('--group', type=int, nargs='+', default=None,
                    help='retrieval: ALSO ablate these layers together, as one '
                         'extra row. Required to measure a shared memory: '
                         'ablating one reader leaves the memory intact -- the '
                         'writer still writes and every other reader still '
                         'reads -- so per-layer rows measure a single readout, '
                         'never the group. Only with every member off is the '
                         "group's contribution to the residual stream zero.")
    args = ap.parse_args()

    # Fail before a multi-minute checkpoint load, not after.
    if args.retrieval and args.ablate:
        raise SystemExit('--ablate is meaningless with --retrieval: the sweep '
                         'ablates each layer itself. Use --layers to restrict '
                         'which layers it sweeps, and --branch to pick the branch.')

    tasks, tm = None, None
    if not args.retrieval:
        tasks = args.tasks or [t for s in args.suite for t in SUITES[s]]
        tm = resolve(tasks)

    # lm-eval sends a different sequence length for nearly every request, while
    # sliding_window_attention compiles flex_attention with dynamic=False. Past
    # dynamo's default cache_size_limit of 8 distinct shapes it stops recompiling
    # and falls back to eager, where flex_attention decomposes to math_attention
    # and materialises the whole [B, H, Q, KV] score matrix -- 34 GiB at 8192
    # tokens and batch 4, which OOMs a 96 GB GH200. Training never hit this
    # because it has one fixed shape. Raise the limits so every length gets its
    # own compiled kernel and the window is actually exploited.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 1024

    config = OmegaConf.load(args.cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.model.pretrained_model_name_or_path = args.ckpt
    model_config = build_model_config(cfg)
    model = load_model(args.ckpt, model_config, args.adapter)

    mods = ttt_layers(model)
    if args.ablate:
        sel = mods if args.layers is None else [mods[i] for i in args.layers]
        which = 'all layers' if args.layers is None else f'layers {args.layers}'
        print(f'ablating {args.ablate} on {which} ({len(sel)} of {len(mods)})')
    else:
        sel = []

    for prm in model.parameters():
        prm.requires_grad_(False)
    use_sdpa_sliding_window(True)

    causality = None
    if getattr(model.config, 'ttt_share_groups', None):
        from test_causality import assert_causal
        chunk = model.config.lact_chunk_size
        generator = torch.Generator(device=model.device).manual_seed(0)
        probe = torch.randint(model.config.vocab_size, (1, 3 * chunk),
                              device=model.device, generator=generator)
        causality = assert_causal(model, probe, chunk)
        print('Shared-memory causality tripwire: passed')

    if args.retrieval:
        with torch.no_grad():
            run_retrieval(args, model, model_config, config, mods)
        return

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    eval_len = int(args.seq_len or config.model.max_length)
    if eval_len > model_config.max_position_embeddings:
        raise SystemExit(f'--seq-len {eval_len} exceeds the model\'s '
                         f'max_position_embeddings {model_config.max_position_embeddings}')
    print(f'eval context length {eval_len}'
          + (' (--seq-len override)' if args.seq_len else ' (from config.model.max_length)')
          + f'; model max_position_embeddings {model_config.max_position_embeddings}')
    lm = HFLM(pretrained=model, tokenizer=args.base,
              batch_size=args.batch_size, max_length=eval_len)

    kwargs = dict(model=lm, tasks=tasks, task_manager=tm, limit=args.limit)
    if args.num_fewshot is not None:
        kwargs['num_fewshot'] = args.num_fewshot

    model.config.use_cache = True
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.use_cache = True

    with torch.no_grad(), ablate(sel, args.ablate if sel else None):
        res = lm_eval.simple_evaluate(**kwargs)

    print(f'\n{"task":<22}{"metric":<18}{"value":>9}')
    flat = {}
    for task, metrics in sorted(res['results'].items()):
        for k, v in metrics.items():
            if k.endswith('_stderr') or k == 'alias' or not isinstance(v, (int, float)):
                continue
            print(f'{task:<22}{k:<18}{v:>9.4f}')
            flat[f'{task}/{k}'] = v

    tag = f'_{args.ablate}' + ('_all' if args.layers is None else
                               '_L' + '-'.join(map(str, args.layers))) if args.ablate else ''
    path = f'{args.out}{tag}.json'
    with open(path, 'w') as f:
        json.dump({'ckpt': args.ckpt, 'adapter': args.adapter, 'tasks': tasks,
                   'limit': args.limit, 'ablate': args.ablate,
                   'layers': args.layers, 'causality': causality,
                   'results': flat}, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
