"""Replay distant validation needles with short controls and TTT on/off.

Uses validation.jsonl only; these are debugging results, not final benchmark
numbers. --ckpt is the original full checkpoint; --adapter is the continuation's
last_ckpt directory (containing both adapters and ttt_params.pt).
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re


TASKS = ('single_number', 'single_uuid')


def select_rows(rows, lengths, per_cell, min_distance):
    groups = defaultdict(list)
    for row in rows:
        distances = row.get('needle_distances', [])
        if (row['task'] in TASKS and row['length_bucket'] in lengths
                and distances and min(distances) >= min_distance):
            groups[(row['task'], row['length_bucket'])].append(row)
    selected = []
    for task in TASKS:
        for length in lengths:
            cell = groups[(task, length)]
            if len(cell) < per_cell:
                raise ValueError(f'Need {per_cell} distant validation rows for {task}/{length}; found {len(cell)}')
            selected.extend(cell[:per_cell])
    return selected


def short_control(row):
    """Keep the actual fact and question, removing only the document filler."""
    if row['task'] not in TASKS:
        raise ValueError('Short controls support single-number/UUID tasks only')
    context, separator, question = row['prompt'].rpartition('\n\nQuestion: ')
    if not separator or not question.endswith('\nAnswer: '):
        raise ValueError(f"Unexpected prompt format: {row['id']}")
    facts = re.findall(r'^Register record_[0-9a-f]+ has value [0-9a-f-]+\.$', context, re.MULTILINE)
    if len(facts) != 1 or not facts[0].endswith(' has value ' + row['answer'] + '.'):
        raise ValueError(f"Expected one intact needle: {row['id']}")
    return 'Read the following document.\n\n' + facts[0] + separator + question


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--cfg', default='Configs/ttt_ar_llama_long_context_anchor_f_reduced.yml')
    parser.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096, 8192])
    parser.add_argument('--per-cell', type=int, default=2)
    parser.add_argument('--min-distance', type=int, default=768)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    for name in ('data_dir', 'ckpt', 'adapter', 'cfg', 'base', 'out_dir'):
        if not getattr(args, name).strip():
            hint = (' Set LONG_CONTEXT_DATA to the prepared-data directory used by training.'
                    if name == 'data_dir' else '')
            parser.error(f'--{name.replace("_", "-")} is empty; check your shell variables.' + hint)
    data = Path(args.data_dir).expanduser()
    for filename in ('validation.jsonl', 'manifest.json'):
        if not (data / filename).is_file():
            parser.error(f'Missing {data / filename}; --data-dir must be the prepared-data '
                         'directory used by training, containing validation.jsonl and manifest.json.')
    if min(args.lengths + [args.per_cell, args.min_distance, args.max_new_tokens]) < 1:
        parser.error('Lengths, counts, and distances must be positive')
    args.lengths = sorted(set(args.lengths))
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('--out-dir must be new or empty')
    rows = [json.loads(line) for line in (data / 'validation.jsonl').read_text().splitlines() if line.strip()]
    selected = select_rows(rows, args.lengths, args.per_cell, args.min_distance)
    if any(row.get('split') != 'validation' for row in selected):
        raise ValueError('Diagnostic requires validation examples')
    if not (Path(args.adapter) / 'ttt_params.pt').is_file():
        raise ValueError('Continuation must contain ttt_params.pt')

    import torch
    from transformers import AutoTokenizer
    from Training.long_context import generate_answer, normalize_answer, tokenizer_fingerprint
    from Training.train import build_model_config
    from measure_flops import load_benchmark_config, set_window_backend
    from eval import ablate, load_model, ttt_layers

    if not torch.cuda.is_available():
        raise RuntimeError('Run this diagnostic on a CUDA node')
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    manifest = json.loads((data / 'manifest.json').read_text())
    if manifest['tokenizer_fingerprint'] != tokenizer_fingerprint(tokenizer):
        raise ValueError('Tokenizer differs from prepared validation data')
    os.environ['LONG_CONTEXT_DATA'] = str(data.resolve())
    cfg = load_benchmark_config(args.cfg, args.ckpt, max(args.lengths) + args.max_new_tokens)
    window = int(cfg.model.window_size)
    if args.min_distance <= window:
        raise ValueError('--min-distance must exceed the configured attention window')
    prompts = []
    for row in selected:
        count = len(tokenizer.encode(row['prompt'], add_special_tokens=True))
        if count != row['prompt_tokens'] or count > row['length_bucket']:
            raise ValueError('Prepared prompt length changed; refusing truncation')
        short = short_control(row)
        if len(tokenizer.encode(short, add_special_tokens=True)) + args.max_new_tokens > window:
            raise ValueError('Short control plus generation must fit inside the attention window')
        prompts.append((row, short))
    output.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.adapter) / 'training_progress.json'
    report = dict(config=vars(args), purpose='validation diagnostic', window_size=window,
                  checkpoint_progress=json.loads(progress_path.read_text()) if progress_path.exists() else None)
    (output / 'config.json').write_text(json.dumps(report, indent=2) + '\n')
    torch.manual_seed(314159)
    set_window_backend(True)
    model = load_model(args.ckpt, build_model_config(cfg), args.adapter, strict_ttt=True).eval()
    mods = ttt_layers(model)
    if not mods:
        raise ValueError('No TTT layers found; cannot perform memory ablation')
    scores = defaultdict(list)
    with (output / 'samples.jsonl').open('w') as stream:
        for row, short in prompts:
            for context, prompt in (('full', row['prompt']), ('facts_only', short)):
                for memory in ('on', 'off'):
                    with ablate(mods, 'ttt' if memory == 'off' else None):
                        prediction = generate_answer(model, tokenizer, dict(prompt=prompt), args.max_new_tokens)
                    exact = int(normalize_answer(prediction) == normalize_answer(row['answer']))
                    recall = int(normalize_answer(row['answer']) in normalize_answer(prediction))
                    result = dict(id=row['id'], task=row['task'], length_bucket=row['length_bucket'],
                                  context=context, memory=memory, answer=row['answer'], prediction=prediction,
                                  exact_match=exact, substring_recall=recall,
                                  prompt_tokens=len(tokenizer.encode(prompt, add_special_tokens=True)),
                                  original_needle_distances=row['needle_distances'])
                    stream.write(json.dumps(result) + '\n')
                    stream.flush()
                    key = f"{row['task']}/{row['length_bucket']}/{context}/memory_{memory}"
                    scores[key].append((exact, recall))
                    print(f"{key} | want={row['answer']!r} | got={prediction!r}", flush=True)
    summary = {key: dict(n=len(values), exact=sum(v[0] for v in values)/len(values),
                         substring_recall=sum(v[1] for v in values)/len(values))
               for key, values in scores.items()}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
