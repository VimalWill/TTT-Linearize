"""Prepare independent Llama training splits or a held-out NIAH heatmap grid."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from Training.long_context import (SYNTHETIC_TASKS, SyntheticGenerator, encode_record,
                                   read_jsonl, tokenizer_fingerprint, write_jsonl)


def partition_sources(rows, seed):
    """Deduplicate documents before splitting; no filler-document split leakage."""
    sources, seen = [], set()
    for row in rows:
        text = str(row.get('text') or row.get('input') or row.get('instruction') or '').strip()
        if not text:
            continue
        identity = hashlib.sha256(text.encode()).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        prompt = row.get('prompt')
        answer = row.get('answer')
        if prompt is None and row.get('instruction') and row.get('output'):
            prompt = f"Instruction: {row['instruction']}\n\n"
            if row.get('input'):
                prompt += f"Document:\n{row['input']}\n\n"
            prompt += "Answer: "
            answer = str(row['output'])
        sources.append(dict(id=identity, text=text, prompt=prompt, answer=answer))
    if len(sources) < 30:
        raise ValueError('Need at least 30 distinct source documents for independent splits')
    random.Random(seed).shuffle(sources)
    held_out = max(3, len(sources) // 10)
    return dict(train=sources[2*held_out:], validation=sources[:held_out],
                test=sources[held_out:2*held_out])


def build_training(directory, tokenizer, sources, lengths, counts, seed, min_distance=768):
    directory = Path(directory)
    used_values = set()
    summary = {}
    for split_index, split in enumerate(('train', 'validation', 'test')):
        documents = sources[split]
        write_jsonl(directory / f'documents_{split}.jsonl', documents)
        generator = SyntheticGenerator(tokenizer, documents, seed + split_index * 100003, used_values)
        rows = []
        for task in SYNTHETIC_TASKS:
            for length in lengths:
                for index in range(counts[split]):
                    identifier = f'{split}-{task}-{length}-{index}'
                    rows.append(generator.example(task, length, split, identifier,
                                                  min_distance=min_distance))
                print(f'prepared {split} {task} length={length}: {counts[split]}', flush=True)
        # The sampler gives instruction examples 20% probability regardless of
        # corpus counts. Keep enough distinct instruction rows to avoid relying
        # on a handful of repeated answers.
        instruction_target = max(1, len(rows) // 4)
        for source in documents:
            if not source.get('prompt') or not source.get('answer'):
                continue
            row = dict(id=f'{split}-instruction-{source["id"]}', split=split,
                       task='instruction', prompt=source['prompt'], answer=source['answer'],
                       answers=[source['answer']], document_ids=[source['id']],
                       needle_depths=[], needle_distances=[], identifiers=[])
            try:
                encoded = encode_record(row, tokenizer, max(lengths))
            except ValueError:
                continue
            row['input_tokens'] = len(encoded['input_ids'])
            row['prompt_tokens'] = len(tokenizer.encode(row['prompt'], add_special_tokens=True))
            row['length_bucket'] = next(n for n in sorted(lengths) if row['input_tokens'] <= n)
            rows.append(row)
            instruction_target -= 1
            if not instruction_target:
                break
        if not any(r['task'] == 'instruction' and r['length_bucket'] <= 8192 for r in rows):
            raise ValueError(f'{split}: no instruction examples fitting the initial phase')
        random.Random(seed + split_index).shuffle(rows)
        write_jsonl(directory / f'{split}.jsonl', rows)
        used_values.update(generator.values)
        summary[split] = dict(examples=len(rows), input_tokens=sum(r['input_tokens'] for r in rows),
                              cells=dict(Counter(f"{r['task']}/{r['length_bucket']}" for r in rows)))
    write_jsonl(directory / 'forbidden_values.jsonl', ({'value': v} for v in sorted(used_values)))
    manifest = dict(format_version=1, tokenizer=getattr(tokenizer, 'name_or_path', ''),
                    tokenizer_fingerprint=tokenizer_fingerprint(tokenizer), seed=seed,
                    lengths=lengths, min_distance=min_distance, splits=summary,
                    source_document_ids={k: [d['id'] for d in v] for k, v in sources.items()})
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def build_niah(directory, tokenizer, data_directory, lengths, depths, samples, seed):
    data_directory, directory = Path(data_directory), Path(directory)
    source_manifest = json.loads((data_directory / 'manifest.json').read_text())
    if source_manifest['tokenizer_fingerprint'] != tokenizer_fingerprint(tokenizer):
        raise ValueError('NIAH tokenizer differs from the training-data tokenizer')
    documents = read_jsonl(data_directory / 'documents_test.jsonl')
    forbidden = [r['value'] for r in read_jsonl(data_directory / 'forbidden_values.jsonl')]
    generator = SyntheticGenerator(tokenizer, documents, seed, forbidden)
    path = directory / 'niah.jsonl'
    with path.open('w') as stream:
        for task in ('single_number', 'single_uuid'):
            for length in lengths:
                for depth in depths:
                    for index in range(samples):
                        row = generator.example(task, length, 'niah_test',
                            f'niah-{task}-{length}-{depth}-{index}', depth=depth / 100,
                            prompt_length=True)
                        row['depth_percent'] = depth
                        stream.write(json.dumps(row) + '\n')
                    print(f'prepared NIAH {task} length={length} depth={depth}', flush=True)
    manifest = dict(format_version=1, tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
                    seed=seed, lengths=lengths, depths=depths, samples_per_cell=samples,
                    examples=2 * len(lengths) * len(depths) * samples,
                    source_manifest=str(data_directory / 'manifest.json'))
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tokenizer', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--source-jsonl', help='local documents with text and optional prompt/answer')
    parser.add_argument('--source-dataset', default='Yukang/LongAlpaca-12k')
    parser.add_argument('--max-source-docs', type=int, default=12000)
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096, 8192, 16384])
    parser.add_argument('--train-per-cell', type=int, default=512)
    parser.add_argument('--validation-per-cell', type=int, default=8)
    parser.add_argument('--test-per-cell', type=int, default=16)
    parser.add_argument('--min-distance', type=int, default=768)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--niah-from', help='prepared training-data directory; generate only final heatmap data')
    parser.add_argument('--depths', nargs='+', type=int, default=list(range(0, 101, 10)))
    parser.add_argument('--samples-per-cell', type=int, default=50)
    args = parser.parse_args()
    if min(args.lengths + [args.train_per_cell, args.validation_per_cell,
                          args.test_per_cell, args.samples_per_cell, args.max_source_docs]) < 1:
        parser.error('Lengths and counts must be positive')
    if args.min_distance < 0 or any(not 0 <= d <= 100 for d in args.depths):
        parser.error('Invalid distance or depth')
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('--out-dir must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.niah_from:
        build_niah(output, tokenizer, args.niah_from, sorted(set(args.lengths)),
                   sorted(set(args.depths)), args.samples_per_cell, args.seed)
    else:
        if args.source_jsonl:
            records = read_jsonl(args.source_jsonl)
        else:
            from datasets import load_dataset
            records = list(itertools.islice(load_dataset(args.source_dataset, split='train',
                                                         streaming=True), args.max_source_docs))
        sources = partition_sources(records, args.seed)
        build_training(output, tokenizer, sources, sorted(set(args.lengths)),
                       dict(train=args.train_per_cell, validation=args.validation_per_cell,
                            test=args.test_per_cell), args.seed, args.min_distance)
    print(f'Wrote {output / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
