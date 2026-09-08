"""Standard benchmarks for a linearized checkpoint, via lm-eval-harness.

Two suites, matching Based (Arora et al., arXiv:2402.18668, Table 1):

  recall      swde, fda, squad_completion -- the recall-intensive tasks
  commonsense piqa, arc_easy, arc_challenge, hellaswag, winogrande,
              lambada_openai -- the control set, which Based notes does NOT
              need recall capacity because the inputs are short

Can also drive the layer sweep on standard tasks rather than the AR-slice
metric in diag_retrieval.py: pass --ablate ttt --layers N. That is the same
measurement on a reportable benchmark, but it costs a full harness pass per
layer, so use --limit.

    # headline numbers
    python3 eval_harness.py --cfg Configs/ttt_ar_l2.yml \
        --ckpt $TTT_CKPT_DIR/ttt_at_l2/best_ckpt \
        --adapter $TTT_CKPT_DIR/ttt_ar_l2/best_ckpt --suite recall commonsense

    # one layer ablated, cheap
    python3 eval_harness.py ... --suite recall --limit 200 --ablate ttt --layers 16
"""

import argparse
import json

import torch
from omegaconf import OmegaConf

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from diag_common import load_model
from diag_retrieval import ablate, ttt_layers
from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window

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
    ap.add_argument('--out', default='eval_harness')
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

    # NOT model_config.max_position_embeddings. build_model_config sets that to
    # max(checkpoint value, cfg max_length), and Llama-3.1-8B ships 131072, so it
    # stays 131072 however short the training context was. Handing that to lm-eval
    # makes it roll wikitext in 128k-token windows on a model trained at 8k --
    # wrong to report, and it OOMs a 96 GB GH200 before the first request lands.
    # The evaluation context must be the context the model was trained at.
    eval_len = int(config.model.max_length)
    print(f'eval context length {eval_len} '
          f'(config max_position_embeddings {model_config.max_position_embeddings})')
    lm = HFLM(pretrained=model, tokenizer=args.base,
              batch_size=args.batch_size, max_length=eval_len)

    kwargs = dict(model=lm, tasks=tasks, task_manager=tm, limit=args.limit)
    if args.num_fewshot is not None:
        kwargs['num_fewshot'] = args.num_fewshot

    # diag_common.load_model calls model.eval() but leaves requires_grad set --
    # the TTT parameters and the LoRA adapters come back from training still
    # trainable, so without this every forward builds an autograd graph and
    # retains 32 layers of activations. diag_retrieval gets away with it because
    # score_ar is decorated @torch.no_grad(); this script had no such guard, and
    # the retained graph is what put 79 GB on the card before a single
    # loglikelihood_rolling window was scored.
    for prm in model.parameters():
        prm.requires_grad_(False)

    # lm-eval sends a different sequence length for almost every request, so
    # flex_attention never settles on a compiled kernel and falls back to the
    # dense [B, H, Q, KV] path. Use the chunked SDPA window instead: same
    # numerics, memory bounded by the window rather than the sequence.
    use_sdpa_sliding_window(True)

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
                   'layers': args.layers, 'results': flat}, f, indent=2)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
