
import argparse
import contextlib
import json
import os
import re

import torch
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
    args = ap.parse_args()

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

    import lm_eval
    from lm_eval.models.huggingface import HFLM

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


    eval_len = int(config.model.max_length)
    print(f'eval context length {eval_len} '
          f'(config max_position_embeddings {model_config.max_position_embeddings})')
    lm = HFLM(pretrained=model, tokenizer=args.base,
              batch_size=args.batch_size, max_length=eval_len)

    kwargs = dict(model=lm, tasks=tasks, task_manager=tm, limit=args.limit)
    if args.num_fewshot is not None:
        kwargs['num_fewshot'] = args.num_fewshot

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
