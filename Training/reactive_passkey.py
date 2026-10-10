"""Conversation-preserving preprocessing for ReactiveAI/passkey-retrieval."""

import hashlib
import random
import re
from collections import Counter


def conversation_id(row):
    # Group variants of the same source query before selecting any turns.
    query = row['interactions'][0]['query']
    return hashlib.sha256(' '.join(query.split()).encode()).hexdigest()


def tokenize_conversation(row, tokenizer, first_turn=False, include_label=True):
    """Plain role delimiters work with the Anchor-I base-model tokenizer.

    All assistant spans are supervised for training. Evaluation uses only the
    first retrieval turn; generation inputs never contain its supplied answer.
    No packing, truncation, or invented chat special tokens.
    """
    turns = row['interactions'][:1] if first_turn else row['interactions']
    if not include_label and not first_turn:
        raise ValueError('Generation evaluation must use the first retrieval turn')
    ids = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    labels = [-100] * len(ids)
    for turn in turns:
        prompt = tokenizer.encode('\n### User:\n' + turn['query'] +
                                  '\n\n### Assistant:\n', add_special_tokens=False)
        answer = tokenizer.encode(turn['answer'], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            answer.append(tokenizer.eos_token_id)
        ids.extend(prompt)
        labels.extend([-100] * len(prompt))
        if include_label:
            ids.extend(answer)
            labels.extend(answer)
        else:
            labels = answer
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def retrieval_distance(row, tokenizer):
    """Conservative distance from the last key mention to the first question.

    Recognize quoted passphrases and food/animal lists in the first answer.
    Unknown formats or missing evidence are excluded from distance-qualified
    evaluation. For multi-item answers, every item must be beyond the window.
    """
    first = row['interactions'][0]
    quoted = re.search(r'\bis\s*:?\s*([\'\"])(.+?)\1', first['answer'])
    if quoted:
        values = [quoted.group(2)]
    else:
        listed = re.search(r'\bare\s+(.+?)[.!]?$', first['answer'].strip())
        if not listed:
            return None
        values = [v.strip() for v in re.split(r',\s*(?:and\s+)?|\s+and\s+',
                                              listed.group(1)) if v.strip()]
    # In this corpus the retrieval question is the final paragraph. Requiring
    # this boundary avoids counting the question itself toward the distance.
    parts = first['query'].rsplit('\n\n', 1)
    if len(parts) != 2 or '?' not in parts[1]:
        return None
    passage, question = parts
    distances = []
    for value in values:
        pattern = r'(?<!\w)' + re.escape(value) + r'(?!\w)'
        if re.search(pattern, question, flags=re.IGNORECASE):
            return 0
        matches = list(re.finditer(pattern, passage, flags=re.IGNORECASE))
        if not matches:
            return None
        # Subtract one token for a possible tokenization boundary merge.
        distances.append(max(0, len(tokenizer.encode(
            passage[matches[-1].end():], add_special_tokens=False)) - 1))
    return min(distances) if distances else None


def prepare_conversations(source, tokenizer, max_tokens, num_val=100,
                          min_distance=2048, seed=42, num_train=None):
    if max_tokens < 1 or num_val < 1 or min_distance < 0:
        raise ValueError('Invalid conversation length, validation count, or distance')
    if num_train is not None and num_train < 1:
        raise ValueError('num_train_docs must be positive or null')
    rows, seen, counts = [], set(), Counter()
    for row in source:
        turns = row.get('interactions')
        if not isinstance(turns, list) or not turns or any(
                not isinstance(t, dict) or any(not isinstance(t.get(k), str)
                    or not t[k].strip() for k in ('query', 'answer')) for t in turns):
            counts['invalid'] += 1
            continue
        cid = conversation_id(row)
        if cid in seen:
            counts['duplicate_query'] += 1
            continue
        seen.add(cid)
        row = {'interactions': turns, 'conversation_id': cid}
        length = len(tokenize_conversation(row, tokenizer)['input_ids'])
        if length > max_tokens:
            counts['too_long'] += 1
            continue
        row['token_count'] = length
        row['retrieval_distance'] = retrieval_distance(row, tokenizer)
        rows.append(row)
    random.Random(seed).shuffle(rows)
    eligible = [r for r in rows if r['retrieval_distance'] is not None
                and r['retrieval_distance'] > min_distance]
    if len(eligible) < num_val:
        raise ValueError(f'Only {len(eligible)} conversations have verified retrieval '
                         f'distance > {min_distance}; need {num_val}. '
                         f'Accepted {len(rows)}; rejected {dict(counts)}')
    validation = eligible[:num_val]
    held_out = {r['conversation_id'] for r in validation}
    train = [r for r in rows if r['conversation_id'] not in held_out]
    if num_train is not None:
        train = train[:num_train]
    if not train:
        raise ValueError('No training conversations remain after validation split')
    print(f'-> ReactiveAI: {len(train)} train conversations / {len(validation)} '
          f'held-out first-turn retrieval examples; distance > {min_distance}; '
          f'max conversation tokens={max(r["token_count"] for r in rows)}; '
          f'rejected={dict(counts)}')
    return train, validation



def mix_repetition_rows(train, validation, config, tokenizer):
    """Add approximately the requested example fraction, keeping source splits."""
    from Training.synthetic_niah import SyntheticNIAH, TRAIN_TEMPLATES, VALIDATION_TEMPLATES
    fraction = float(config.data.get('synthetic_fraction', 0))
    if not 0 < fraction < 1:
        raise ValueError('synthetic_fraction must be between zero and one')
    lengths = [int(x) for x in config.data.get('synthetic_lengths', [4096, 8192, 16384])]
    if not lengths or min(lengths) < 128 or max(lengths) > int(config.model.max_length):
        raise ValueError('Invalid synthetic_lengths for model.max_length')
    seed = int(config.data.get('seed', 42))
    mixed = []
    for rows, split, templates, offset in (
            (train, 'train', TRAIN_TEMPLATES, 0),
            (validation, 'validation', VALIDATION_TEMPLATES, 10000)):
        count = max(1, round(len(rows) * fraction / (1 - fraction)))
        synthetic = SyntheticNIAH(tokenizer, seed + offset, templates).dataset(count, lengths, split)
        encoded = [dict(tokenize_conversation(row, tokenizer, first_turn=split != 'train'),
                        _retrieval_source='passkey') for row in rows]
        encoded += [dict((k, row[k]) for k in ('input_ids', 'attention_mask', 'labels'))
                    | {'_retrieval_source': 'repetition'} for row in synthetic]
        random.Random(seed + offset).shuffle(encoded)
        print(f'-> {split}: {len(rows)} passkey + {count} repetition examples '
              f'({count / len(encoded):.1%}); synthetic lengths {lengths}')
        mixed.append(encoded)
    return mixed


def load_reactive_data(config, tokenizer):
    from datasets import Dataset, load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    if int(config.data.micro_batch_size) != 1:
        raise ValueError('ReactiveAI conversations require micro_batch_size=1')
    source = load_dataset(config.data.path, name=config.data.subset,
                          split='train', streaming=True)
    train, validation = prepare_conversations(
        source, tokenizer, int(config.model.max_length),
        num_val=int(config.data.num_val_docs),
        min_distance=int(config.data.get('passkey_min_distance', 2048)),
        seed=int(config.data.get('seed', 42)),
        num_train=config.data.get('num_train_docs'))
    tokenizer.padding_side = 'left'
    collate = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                    return_tensors='pt')
    loaders = {}
    mixed = None
    if float(config.data.get('synthetic_fraction', 0)) != 0:
        mixed = dict(zip(('train', 'validation'),
                         mix_repetition_rows(train, validation, config, tokenizer)))

    def collate_sources(features):
        sources = [f['_retrieval_source'] for f in features]
        batch = collate([{k: v for k, v in f.items() if k != '_retrieval_source'}
                         for f in features])
        batch['retrieval_sources'] = sources
        return batch

    for split, rows in (('train', train), ('validation', validation), ('test', validation)):
        dataset = Dataset.from_list(mixed[split] if mixed and split != 'test' else [tokenize_conversation(
            row, tokenizer, first_turn=split != 'train', include_label=split != 'test')
            for row in rows])
        dataset.tokenizer, dataset.metric = tokenizer, None
        # Retain audit metadata outside batches; test is the generation view
        # of validation, not an independent benchmark split.
        dataset.records = rows
        loaders[split] = DataLoader(dataset, batch_size=1, shuffle=split == 'train',
                                     collate_fn=collate_sources if mixed and split != 'test' else collate,
                                     num_workers=0, pin_memory=True)
    return loaders
