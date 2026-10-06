"""Independent synthetic supervision, balanced sampling, and retrieval validation."""
from collections import defaultdict, OrderedDict
import hashlib
import inspect
import json
from pathlib import Path
import random
import re
import uuid

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from transformers import DataCollatorForSeq2Seq

TASK_WEIGHTS = dict(single_number=.15, single_uuid=.15, multikey=.10,
                    multiquery=.10, multivalue=.10, tracing=.10,
                    aggregation=.10, instruction=.20)
SYNTHETIC_TASKS = tuple(k for k in TASK_WEIGHTS if k != 'instruction')


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows, append=False):
    with Path(path).open('a' if append else 'w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def tokenizer_fingerprint(tokenizer):
    payload = dict(vocab=tokenizer.get_vocab(), bos=tokenizer.bos_token_id,
                   eos=tokenizer.eos_token_id,
                   backend=(tokenizer.backend_tokenizer.to_str()
                            if hasattr(tokenizer, 'backend_tokenizer') else None))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def encode_record(row, tokenizer, max_length=None):
    prompt = tokenizer.encode(row['prompt'], add_special_tokens=True)
    answer = tokenizer.encode(row['answer'], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer.append(tokenizer.eos_token_id)
    if not prompt or not answer or (max_length and len(prompt) + len(answer) > max_length):
        raise ValueError(f"Invalid/overlength example {row.get('id')}; never truncate retrieval data")
    if row.get('prompt_tokens', len(prompt)) != len(prompt):
        raise ValueError('Prepared prompt token count changed')
    if row.get('input_tokens', len(prompt) + len(answer)) != len(prompt) + len(answer):
        raise ValueError('Prepared input token count changed')
    return dict(input_ids=prompt + answer, attention_mask=[1] * (len(prompt) + len(answer)),
                labels=[-100] * len(prompt) + answer)


class SyntheticGenerator:
    def __init__(self, tokenizer, documents, seed, forbidden_values=()):
        if not documents or any(not d.get('text', '').strip() for d in documents):
            raise ValueError('Nonempty source documents are required')
        self.tokenizer, self.documents = tokenizer, documents
        self.rng = random.Random(seed)
        self.values = set(forbidden_values)
        self.document_tokens = OrderedDict()

    def value(self, kind='number'):
        while True:
            value = (str(self.rng.randrange(1000000, 10000000)) if kind == 'number'
                     else str(uuid.UUID(int=self.rng.getrandbits(128))))
            if value not in self.values:
                self.values.add(value)
                return value

    def key(self):
        return 'record_' + self.value('uuid').replace('-', '')[:16]

    def task(self, task, length):
        kind = 'uuid' if task == 'single_uuid' else 'number'
        if task in ('multikey', 'multiquery', 'multivalue'):
            kind = self.rng.choice(('number', 'uuid'))
        count = 4 if task in ('multikey', 'multiquery', 'multivalue') else 1
        values = [self.value(kind) for _ in range(count)]
        keys = [self.key() for _ in range(count)]
        if task in ('single_number', 'single_uuid', 'multikey', 'multiquery'):
            facts = [f'Register {k} has value {v}.' for k, v in zip(keys, values)]
            selected = list(range(count)) if task == 'multiquery' else [self.rng.randrange(count)]
            question = ('Return the values for registers ' + ', '.join(keys[i] for i in selected)
                        + ', in that order. Output only the values separated by commas.')
            answers = [values[i] for i in selected]
        elif task == 'multivalue':
            facts = [f'Register {keys[0]} has values {", ".join(values)}.']
            question = f'Return all values for register {keys[0]}, in their listed order, separated by commas.'
            answers = values
        elif task == 'tracing':
            keys = [self.key() for _ in range(5)]
            facts = [f'Register {keys[0]} has value {values[0]}.'] + [
                f'Register {keys[i]} copies the value of register {keys[i-1]}.' for i in range(1, 5)]
            question = f'Resolve register {keys[-1]}. Output only its final numeric value.'
            answers = values
        elif task == 'aggregation':
            winners = self.rng.choice((2, 3, 5, 10) if length >= 8192 else (2, 3, 5))
            labels = ['label_' + self.value() for _ in range(winners + 3)]
            counts = list(range(20, 20-winners, -1)) + [3, 2, 1]
            facts = [f'EVENT: {label}' for label, n in zip(labels, counts) for _ in range(n)]
            question = (f'Considering only EVENT entries, return the {winners} most frequent labels, '
                        'highest frequency first, separated by commas.')
            answers = labels[:winners]
        else:
            raise ValueError(f'Unknown task {task}')
        return facts, question, answers, values + keys

    def example(self, task, length, split, identifier, depth=None, prompt_length=False,
                far_fraction=.8, min_distance=768):
        facts, question, answers, identifiers = self.task(task, length)
        answer = ', '.join(answers)
        answer_tokens = len(self.tokenizer.encode(answer, add_special_tokens=False))
        answer_tokens += int(self.tokenizer.eos_token_id is not None)
        target = length if prompt_length else length - answer_tokens
        if target < 32:
            raise ValueError('Length too short for task and answer')
        # Reuse a token pool for all render adjustments; preserve the full
        # question and facts. Source documents are split before generation.
        pool, document_ids = [], set()
        while len(pool) < length * 2:
            document = self.rng.choice(self.documents)
            identity = document['id']
            if identity not in self.document_tokens:
                tokens = self.tokenizer.encode(document['text'], add_special_tokens=False)
                if not tokens:
                    raise ValueError('Source document tokenized to empty')
                self.document_tokens[identity] = tokens
                if len(self.document_tokens) > 128:
                    self.document_tokens.popitem(last=False)
            pool.extend(self.document_tokens[identity])
            pool.extend(self.tokenizer.encode('\n\n', add_special_tokens=False))
            document_ids.add(document['id'])
        require_far = depth is None and self.rng.random() < far_fraction
        for attempt in range(40):
            filler_count = max(0, target - len(self.tokenizer.encode(
                '\n'.join(facts) + question, add_special_tokens=False)) - 40)
            depth_cap = (min(.70, max(0, 1 - min_distance / max(1, filler_count)))
                         if require_far else .98)
            depths = ([float(depth)] * len(facts) if depth is not None else
                      [self.rng.uniform(0, depth_cap) for _ in facts])
            prompt = None
            for _ in range(30):
                pieces, last = [], 0
                for position, fact in sorted(zip(depths, facts)):
                    stop = int(position * filler_count)
                    pieces += [self.tokenizer.decode(pool[last:stop], skip_special_tokens=True),
                               '\n' + fact + '\n']
                    last = stop
                pieces += [self.tokenizer.decode(pool[last:filler_count], skip_special_tokens=True)]
                context = ''.join(pieces)
                prompt = f'Read the following document.\n\n{context}\n\nQuestion: {question}\nAnswer: '
                n = len(self.tokenizer.encode(prompt, add_special_tokens=True))
                if target - 16 <= n <= target:
                    break
                next_count = max(0, filler_count + target - n - (1 if n > target else 0))
                if next_count == filler_count:
                    prompt = None
                    break
                filler_count = next_count
            else:
                prompt = None
            if prompt is None:
                continue
            distances = [len(self.tokenizer.encode(prompt.rsplit(fact, 1)[1], add_special_tokens=False))
                         for fact in facts]
            if require_far and min(distances) < min_distance:
                continue
            # Reject accidental answer occurrences in the filler, which can
            # otherwise make a distant retrieval example solvable locally.
            filler_text = self.tokenizer.decode(pool[:filler_count], skip_special_tokens=True)
            if any(v in filler_text for v in answers):
                continue
            row = dict(id=identifier, split=split, task=task, length_bucket=length,
                       prompt=prompt, answer=answer, answers=answers,
                       needle_depths=depths, needle_distances=distances,
                       document_ids=sorted(document_ids), prompt_tokens=n,
                       input_tokens=n + answer_tokens, require_far=require_far,
                       identifiers=identifiers)
            if not prompt_length:
                encode_record(row, self.tokenizer, length)
            return row
        raise ValueError(f'Could not construct intact {task} example at {length} tokens')


class ContextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.records, self.tokenizer, self.max_length = rows, tokenizer, max_length
        self.metric = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        features = encode_record(row, self.tokenizer, self.max_length)
        features['_context_task'] = row['task']
        features['_context_length'] = row['length_bucket']
        return features


class BalancedContextSampler(Sampler):
    """Sample task weights and uniform eligible length buckets, without padding."""
    def __init__(self, rows, max_length, seed=0, weights=None):
        self.groups = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            self.groups[row['task']][int(row['length_bucket'])].append(index)
        self.max_length, self.seed, self.epoch = max_length, seed, 0
        self.weights = dict(weights or TASK_WEIGHTS)
        if any(v <= 0 for v in self.weights.values()):
            raise ValueError('Task sampling weights must be positive')
        self.size = len(rows)
        self._eligible()

    def _eligible(self):
        eligible = {task: {n: ids for n, ids in groups.items() if n <= self.max_length}
                    for task, groups in self.groups.items() if self.weights.get(task, 0) > 0}
        eligible = {task: groups for task, groups in eligible.items() if groups}
        if set(self.weights) - set(eligible):
            raise ValueError(f'Missing task/length cells for sampler: {set(self.weights) - set(eligible)}')
        return eligible

    def set_max_length(self, value):
        self.max_length = value
        self._eligible()

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.size):
            # Read current maximum on each yield: the trainer can advance the
            # curriculum after an update without rebuilding the dataloader.
            eligible = self._eligible()
            tasks = sorted(eligible)
            task = rng.choices(tasks, weights=[self.weights[t] for t in tasks], k=1)[0]
            length = rng.choice(sorted(eligible[task]))
            yield rng.choice(eligible[task][length])


def load_context_data(config, tokenizer):
    directory = Path(config.data.path)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['tokenizer_fingerprint'] != tokenizer_fingerprint(tokenizer):
        raise ValueError('Prepared data tokenizer does not match the training checkpoint')
    if int(config.data.micro_batch_size) != 1:
        raise ValueError('Long-context examples require micro_batch_size=1; never pad mixed lengths')
    maximum = int(config.model.max_length)
    allowed = config.data.get('allowed_lengths', None)
    if allowed is not None and set(manifest['lengths']) != set(allowed):
        raise ValueError('Prepared corpus lengths differ from allowed_lengths; use a separate reduced data directory')
    if maximum not in manifest['lengths']:
        raise ValueError('Prepared corpus lacks the configured final training length')
    datasets = {split: ContextDataset(read_jsonl(directory / f'{split}.jsonl'), tokenizer, maximum)
                for split in ('train', 'validation', 'test')}
    if any(not len(dataset) for dataset in datasets.values()):
        raise ValueError('All data splits must be nonempty')
    if allowed is not None and any(row['length_bucket'] not in allowed
                                  or row['input_tokens'] > maximum
                                  for dataset in datasets.values() for row in dataset.records):
        raise ValueError('Prepared rows exceed the allowed training/validation lengths')
    sampler = BalancedContextSampler(datasets['train'].records,
                                     int(config.data.get('initial_max_length', 8192)),
                                     int(config.data.get('seed', 0)),
                                     config.data.get('task_weights', None))
    collate = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, return_tensors='pt')
    def collate_with_metadata(features):
        tasks = [f.pop('_context_task') for f in features]
        lengths = [f.pop('_context_length') for f in features]
        batch = collate(features)
        batch['context_tasks'], batch['context_lengths'] = tasks, lengths
        return batch
    return {name: DataLoader(dataset, batch_size=1, num_workers=0,
                            sampler=sampler if name == 'train' else None,
                            collate_fn=collate_with_metadata, pin_memory=True)
            for name, dataset in datasets.items()}


def normalize_answer(text):
    # Exact match after whitespace/case normalization. Do not accept arbitrary
    # substring matches or explanatory text as a successful retrieval.
    text = re.sub(r'\s+', ' ', text.strip()).casefold()
    return re.sub(r'\s*,\s*', ',', text)


def generate_answer(model, tokenizer, row, max_new_tokens=256):
    from Training.utils import model_autocast
    ids = tokenizer.encode(row['prompt'], add_special_tokens=True)
    tensor = torch.tensor([ids], device=model.device)
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    parameters = inspect.signature(base.forward).parameters
    kwargs = {name: 1 for name in ('logits_to_keep', 'num_logits_to_keep') if name in parameters}
    with torch.no_grad(), model_autocast(model):
        sequence = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                  max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                  temperature=1.0, top_p=1.0,
                                  use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id, **kwargs)
    if sequence.shape[1] < len(ids) or not torch.equal(sequence[0, :len(ids)], tensor[0]):
        raise RuntimeError('Generation changed/truncated the supplied prompt')
    return tokenizer.decode(sequence[0, len(ids):], skip_special_tokens=True).strip()


def retrieval_validation(model, tokenizer, rows, per_cell=4, max_new_tokens=256,
                         samples_path=None, step=None, input_tokens=None):
    if per_cell < 1:
        raise ValueError('validation_per_cell must be positive')
    groups = defaultdict(list)
    for row in rows:
        if row['task'] != 'instruction':
            groups[(row['task'], int(row['length_bucket']))].append(row)
    if not groups:
        raise ValueError('No retrieval validation examples')
    cell_scores, lengths = [], defaultdict(list)
    cell_recalls, length_recalls, samples = [], defaultdict(list), []
    for (task, length), examples in sorted(groups.items()):
        results, recalls = [], []
        for row in examples[:per_cell]:
            prediction = generate_answer(model, tokenizer, row, max_new_tokens)
            normalized = normalize_answer(prediction)
            exact = float(normalized == normalize_answer(row['answer']))
            answers = row.get('answers', [row['answer']])
            recall = sum(normalize_answer(a) in normalized for a in answers) / len(answers)
            results.append(exact)
            recalls.append(recall)
            samples.append(dict(id=row.get('id'), step=step, input_tokens=input_tokens,
                task=task, length_bucket=length, answer=row['answer'], answers=answers,
                prediction=prediction, exact_match=exact, substring_recall=recall,
                prompt_tokens=row.get('prompt_tokens'), needle_distances=row.get('needle_distances')))
        score = sum(results) / len(results)
        recall = sum(recalls) / len(recalls)
        print(f'validation retrieval {task} length={length}: {score:.3f} '
              f'(substring recall={recall:.3f}, n={len(results)})', flush=True)
        cell_scores.append(score)
        lengths[length].append(score)
        cell_recalls.append(recall)
        length_recalls[length].append(recall)
    if samples_path is not None:
        # Append: each eval truncated the file, so only the newest step
        # survived and no trend across steps could ever be measured. The
        # dataset writers in prepare_long_context still truncate.
        write_jsonl(samples_path, samples, append=True)
    return {'eval/retrieval_accuracy': sum(cell_scores) / len(cell_scores),
            'eval/retrieval_substring_recall': sum(cell_recalls) / len(cell_recalls),
            **{f'eval/retrieval_accuracy_{length}': sum(scores) / len(scores)
               for length, scores in lengths.items()},
            **{f'eval/retrieval_substring_recall_{length}': sum(scores) / len(scores)
               for length, scores in length_recalls.items()}}
