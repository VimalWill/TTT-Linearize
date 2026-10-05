"""Evaluate selected Llama arms on a fixed NIAH grid and the full RULER suite.

Each arm/length/ablation runs in a fresh process. OOMs and errors are recorded
and do not prevent subsequent cases. Final test data never select checkpoints.
"""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

RULER_TASKS = ('niah_single_1', 'niah_single_2', 'niah_single_3',
               'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
               'niah_multiquery', 'niah_multivalue', 'ruler_vt', 'ruler_cwe',
               'ruler_fwe', 'ruler_qa_hotpot', 'ruler_qa_squad')
ARM_NAMES = ('baseline', 'anchor_f', 'anchor_i', 'anchor')


def selected_arms(args):
    available = dict(baseline=('baseline', None, None, None),
        anchor_f=('anchor_f', args.anchor_f_cfg, args.anchor_f_ckpt, args.anchor_f_adapter),
        anchor_i=('anchor_i', args.anchor_i_cfg, args.anchor_i_ckpt, args.anchor_i_adapter),
        anchor=('anchor', args.anchor_cfg, args.anchor_i_ckpt, args.anchor_i_adapter))
    names = getattr(args, 'arms', ARM_NAMES)
    if not names or len(set(names)) != len(names) or any(n not in available for n in names):
        raise ValueError('Select distinct supported evaluation arms')
    arms = [available[n] for n in names]
    for name, cfg, ckpt, adapter in arms:
        if name != 'baseline' and not ckpt:
            raise ValueError(f'{name} requires its source checkpoint')
    return arms


def ruler_scores(record, length):
    scores = {}
    for task in RULER_TASKS:
        candidates = [v for k, v in record['results'].items()
                      if k.split('/')[0] == task and 'stderr' not in k
                      and k.split('/', 1)[1].split(',')[0] == str(length)]
        if len(candidates) != 1 or not 0 <= candidates[0] <= 1:
            raise ValueError(f'Missing/invalid RULER metric {task} at {length}; never average -1 sentinels')
        scores[task] = candidates[0]
    return scores


def niah_worker(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from Training.long_context import generate_answer, normalize_answer, tokenizer_fingerprint
    from measure_flops import is_cuda_oom, load_benchmark_config, set_window_backend
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    status = dict(status='error', phase='setup', length=args.length, use_cache=True,
                  baseline=args.baseline, ablate=args.ablate, completed_examples=0)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for checkpoint evaluation')
        manifest = json.loads(Path(args.niah_data).with_name('manifest.json').read_text())
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if manifest['tokenizer_fingerprint'] != tokenizer_fingerprint(tokenizer):
            raise ValueError('NIAH data tokenizer differs from evaluation tokenizer')
        torch.manual_seed(args.seed)
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 1024
        status['phase'] = 'model_load'
        if args.baseline:
            model = AutoModelForCausalLM.from_pretrained(
                args.base, torch_dtype=torch.bfloat16, device_map={'': 0}).eval()
            mods = []
        else:
            from Training.train import build_model_config
            from eval import load_model, ttt_layers
            set_window_backend(True)  # same reference window backend as eval.py
            cfg = load_benchmark_config(args.cfg, args.ckpt, args.length + args.max_new_tokens)
            model = load_model(args.ckpt, build_model_config(cfg), args.adapter)
            mods = ttt_layers(model)
        from eval import ablate
        status['phase'] = 'generation'
        with output.with_suffix('.samples.jsonl').open('w') as stream, \
                torch.no_grad(), ablate(mods, args.ablate):
            with Path(args.niah_data).open() as source:
                for line in source:
                    row = json.loads(line)
                    if row['length_bucket'] != args.length:
                        continue
                    actual = len(tokenizer.encode(row['prompt'], add_special_tokens=True))
                    if actual != row['prompt_tokens'] or actual > args.length:
                        raise ValueError('NIAH prompt length changed; refusing truncation')
                    prediction = generate_answer(model, tokenizer, row, args.max_new_tokens)
                    result = {k: row[k] for k in ('id', 'task', 'length_bucket', 'depth_percent',
                                                  'answer', 'answers', 'prompt_tokens', 'needle_distances')}
                    result.update(prediction=prediction,
                                  exact_match=int(normalize_answer(prediction) == normalize_answer(row['answer'])),
                                  answer_recall=sum(a.casefold() in prediction.casefold()
                                                    for a in row['answers']) / len(row['answers']))
                    stream.write(json.dumps(result) + '\n')
                    stream.flush()
                    status['completed_examples'] += 1
                    if status['completed_examples'] % 25 == 0:
                        print(f"evaluated {status['completed_examples']} examples", flush=True)
        expected = 2 * len(manifest['depths']) * manifest['samples_per_cell']
        if status['completed_examples'] != expected:
            raise ValueError(f"Incomplete NIAH length: {status['completed_examples']}/{expected}")
        status.update(status='ok', phase='complete', expected_examples=expected)
    except Exception as error:
        status.update(status='oom' if is_cuda_oom(error) else 'error',
                      error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
    output.write_text(json.dumps(status, indent=2) + '\n')
    return 0 if status['status'] == 'ok' else 2 if status['status'] == 'oom' else 1


def run_matrix(args):
    arms = selected_arms(args)
    directory = Path(args.out_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('--out-dir must be new or empty')
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ('torch', 'transformers', 'lm_eval'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    report = dict(config=vars(args), versions=versions, cases=[])
    environment = dict(os.environ, LONG_CONTEXT_DATA=str(Path(args.data_dir).resolve()))
    repository = Path(__file__).parent
    for benchmark in args.benchmarks:
        for length in args.lengths:
            for name, cfg, ckpt, adapter in arms:
                for branch in ([None] if name == 'baseline' or args.no_ablation else [None, 'ttt']):
                    stem = f'{benchmark}_{name}_{length}' + ('_ttt_off' if branch else '')
                    path, log = directory / f'{stem}.json', directory / f'{stem}.log'
                    if benchmark == 'niah':
                        command = [sys.executable, str(repository / 'evaluate_long_context.py'),
                                   '--worker', '--niah-data', args.niah_data,
                                   '--length', str(length), '--out', str(path),
                                   '--base', args.base, '--seed', str(args.seed),
                                   '--max-new-tokens', str(args.max_new_tokens)]
                    else:
                        # eval.py adds an ablation suffix to --out. Keep the
                        # actual result path in the report for downstream plots.
                        command = [sys.executable, str(repository / 'eval.py'), '--base', args.base,
                                   '--cfg', cfg or args.anchor_i_cfg,
                                   '--tasks', *RULER_TASKS, '--ruler-lengths', str(length),
                                   '--seq-len', str(length + 512), '--batch-size', '1',
                                   '--seed', str(args.seed), '--log-samples',
                                   '--out', str(path.with_suffix(''))]
                        if args.ruler_limit:
                            command += ['--limit', str(args.ruler_limit)]
                        if branch:
                            path = directory / f'{stem}_ttt_all.json'
                    if cfg:
                        if benchmark == 'niah':
                            command += ['--cfg', cfg]
                        command += ['--ckpt', ckpt]
                        if adapter:
                            command += ['--adapter', adapter]
                    else:
                        command += ['--baseline']
                    if branch:
                        command += ['--ablate', branch]
                    print(f'START {stem}; log={log}', flush=True)
                    with log.open('w') as stream:
                        process = subprocess.run(command, env=environment, stdout=stream,
                                                 stderr=subprocess.STDOUT)
                    case = dict(benchmark=benchmark, arm=name, length=length,
                                ablate=branch, returncode=process.returncode, command=command,
                                result_path=str(path), log=str(log), status='error')
                    try:
                        record = json.loads(path.read_text())
                        if benchmark == 'niah':
                            case.update(status=record['status'], error=record.get('error'))
                        else:
                            if process.returncode:
                                raise ValueError(f'RULER worker exited {process.returncode}')
                            case['task_scores'] = ruler_scores(record, length)
                            case.update(status='ok', ruler_macro=sum(case['task_scores'].values()) / 13)
                        if process.returncode and case['status'] == 'ok':
                            raise ValueError('Worker reported success but exited nonzero')
                    except (OSError, KeyError, ValueError) as error:
                        case.update(status='error', error=str(error))
                    report['cases'].append(case)
                    (directory / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(f"{stem}: {case['status'].upper()}", flush=True)
    return 1 if any(c['status'] != 'ok' for c in report['cases']) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--niah-data', required=True, help='prepared niah.jsonl')
    parser.add_argument('--data-dir', help='prepared training data directory')
    parser.add_argument('--out-dir')
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument('--benchmarks', nargs='+', choices=['niah', 'ruler'], default=['niah', 'ruler'])
    parser.add_argument('--arms', nargs='+', choices=ARM_NAMES, default=list(ARM_NAMES),
                        help='only selected arms need checkpoints; memory arms also run ablated')
    parser.add_argument('--anchor-i-ckpt')
    parser.add_argument('--anchor-i-adapter')
    parser.add_argument('--anchor-f-ckpt')
    parser.add_argument('--anchor-f-adapter')
    parser.add_argument('--anchor-i-cfg', default='Configs/ttt_ar_llama_long_context_anchor_i.yml')
    parser.add_argument('--anchor-f-cfg', default='Configs/ttt_ar_llama_long_context_anchor_f.yml')
    parser.add_argument('--anchor-cfg', default='Configs/ttt_deploy_llama_long_context_anchor.yml')
    parser.add_argument('--ruler-limit', type=int, default=500, help='examples per task and length')
    parser.add_argument('--no-ablation', action='store_true')
    parser.add_argument('--seed', type=int, default=314159)
    parser.add_argument('--length', type=int)
    parser.add_argument('--out')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--cfg')
    parser.add_argument('--ckpt')
    parser.add_argument('--adapter')
    parser.add_argument('--ablate', choices=['ttt'])
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()
    if min(args.lengths + [args.max_new_tokens]) < 1 or args.ruler_limit < 1:
        parser.error('Lengths, generation limit, and RULER limit must be positive')
    if args.worker:
        if not args.length or not args.out or (not args.baseline and (not args.cfg or not args.ckpt)):
            parser.error('Worker needs length/output and a baseline or config/checkpoint')
        return niah_worker(args)
    if not all((args.data_dir, args.out_dir)):
        parser.error('Matrix requires --data-dir and --out-dir')
    try:
        selected_arms(args)
    except ValueError as error:
        parser.error(str(error))
    return run_matrix(args)


if __name__ == '__main__':
    sys.exit(main())
