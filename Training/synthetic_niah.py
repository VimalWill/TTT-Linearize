"""Synthetic repeated-haystack retrieval with answer-only supervision."""
import random
import string

FILLER = ('The grass is green. The sky is blue. The sun is yellow. '
          'Here we go. There and back again.')
NEEDLE = 'One of the special magic numbers for {key} is: {value}.'
QUESTION = ('What is the special magic number for {key} mentioned in the '
            'provided text?')


# Training and held-out validation use disjoint wording, not just new seeds.
TRAIN_TEMPLATES = (
    (FILLER, NEEDLE, QUESTION),
    ('The train arrives. The bell rings. The doors open. All is quiet.',
     'The access code assigned to {key} is {value}.',
     'Which access code was assigned to {key}?'),
    ('Rain falls on the roof. Wind moves the leaves. The street is empty.',
     'Remember this record: {key} has reference number {value}.',
     'Return the reference number recorded for {key}.'),
)
VALIDATION_TEMPLATES = (
    ('A boat crosses the lake. Clouds cover the hill. The water is still.',
     'The archive lists {value} as the identifier belonging to {key}.',
     'What identifier does the archive give for {key}?'),
    ('The clock ticks. A lamp lights the room. A book rests on the desk.',
     'For {key}, the stored numeric password reads {value}.',
     'Give the stored numeric password for {key}.'),
)

WORDS = ('mountain', 'harbour', 'lantern', 'cascade', 'meadow', 'compass',
         'thunder', 'orchard', 'granite', 'willow', 'ember', 'quarry',
         'beacon', 'driftwood', 'marigold', 'summit', 'cobalt', 'juniper',
         'tundra', 'zephyr')


class SyntheticNIAH:
    """Build repeat-haystack needle examples at a target token length.

    `depth` is where the needle sits as a fraction of the filler, so depth 0
    puts it at the start and 0.9 near the end. Sampling depths uniformly
    matters: a model trained only on late needles learns recency, not
    retrieval.
    """

    def __init__(self, tokenizer, seed=0, templates=None):
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        self.templates = templates
        self._filler_tokens = tokenizer.encode(FILLER, add_special_tokens=False)
        if not self._filler_tokens:
            raise ValueError('Filler tokenized to empty')

    def _key(self):
        return f'{self.rng.choice(WORDS)}-{self.rng.choice(string.ascii_lowercase)}' \
               f'{self.rng.randrange(10, 100)}'

    def _value(self):
        return str(self.rng.randrange(1000000, 10000000))

    def example(self, length, depth=None, identifier=None):
        key, value = self._key(), self._value()
        filler, needle_template, question_template = (
            self.rng.choice(self.templates) if self.templates else (FILLER, NEEDLE, QUESTION))
        filler_tokens = self.tokenizer.encode(filler, add_special_tokens=False)
        if not filler_tokens:
            raise ValueError('Filler tokenized to empty')
        needle = needle_template.format(key=key, value=value)
        question = question_template.format(key=key)
        answer_tokens = len(self.tokenizer.encode(value, add_special_tokens=False))
        answer_tokens += int(self.tokenizer.eos_token_id is not None)

        scaffold = (f'Read the following document.\n\n\n\nQuestion: {question}\nAnswer: ')
        overhead = len(self.tokenizer.encode(scaffold + needle, add_special_tokens=True))
        filler_budget = length - answer_tokens - overhead
        if filler_budget < 64:
            raise ValueError(f'Length {length} too short for a needle plus filler')

        repeats = filler_budget // len(filler_tokens) + 2
        pool = filler_tokens * repeats
        depth = self.rng.random() if depth is None else float(depth)
        if not 0 <= depth <= 1:
            raise ValueError('depth must be between zero and one')
        cut = int(depth * filler_budget)
        decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True)
        context = f'{decode(pool[:cut])}\n{needle}\n{decode(pool[cut:filler_budget])}'
        prompt = (f'Read the following document.\n\n{context}\n\n'
                  f'Question: {question}\nAnswer: ')

        # The filler is fixed text, so a 7-digit value cannot collide with it;
        # assert rather than resample, since a hit would mean the template
        # changed underneath us.
        if value in filler:
            raise ValueError('Needle value collides with the filler')

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # BPE boundaries can make encode(decode(tokens)) longer. Remove only
        # filler and rebuild the intact needle/question until the cap is met.
        while len(ids) + answer_tokens > length:
            filler_budget -= max(1, len(ids) + answer_tokens - length)
            if filler_budget < 0:
                raise ValueError('Prompt and answer exceed requested length')
            cut = int(depth * filler_budget)
            context = f'{decode(pool[:cut])}\n{needle}\n{decode(pool[cut:filler_budget])}'
            prompt = (f'Read the following document.\n\n{context}\n\n'
                      f'Question: {question}\nAnswer: ')
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        answer = self.tokenizer.encode(value, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            answer = answer + [self.tokenizer.eos_token_id]
        distance = len(ids) - len(self.tokenizer.encode(
            prompt.rsplit(needle, 1)[0], add_special_tokens=True))
        return {
            'id': identifier, 'task': 'niah_repeat', 'length_bucket': length,
            'prompt': prompt, 'answer': value, 'answers': [value],
            'needle_depth': depth, 'needle_distance': distance,
            'prompt_tokens': len(ids),
            'input_ids': ids + answer,
            'attention_mask': [1] * (len(ids) + len(answer)),
            # Answer-only supervision: the haystack is filler by construction
            # and training the model to predict it is what teaches repetition.
            'labels': [-100] * len(ids) + answer,
        }

    def dataset(self, count, lengths, split='train'):
        rows = []
        for index in range(count):
            length = lengths[index % len(lengths)]
            rows.append(self.example(length, identifier=f'{split}_{index}'))
        return rows


class _Rows:
    """Minimal Dataset over pre-encoded rows, keeping `records` for validation."""

    def __init__(self, rows):
        self.records = rows

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {k: row[k] for k in ('input_ids', 'attention_mask', 'labels')}


def load_synthetic_niah(config, tokenizer):
    """Dataloaders for repeat-haystack needle retrieval, generated on the fly.

    No prepared corpus: the haystack is a fixed sentence repeated, so there
    is nothing to download or pre-tokenise. Train and validation use
    different seeds, so the validation needles, keys and depths are unseen.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq

    if int(config.data.micro_batch_size) != 1:
        raise ValueError('Variable-length needle examples require micro_batch_size=1')
    lengths = [int(x) for x in config.data.get('lengths', [4096, 8192, 16384])]
    maximum = int(config.model.max_length)
    if max(lengths) > maximum:
        raise ValueError(f'lengths up to {max(lengths)} exceed max_length {maximum}')
    seed = int(config.data.get('seed', 0))
    train = SyntheticNIAH(tokenizer, seed=seed).dataset(
        int(config.data.get('num_train_docs', 4000)), lengths, 'train')
    # A different seed, so no validation key, value or depth was trained on.
    validation = SyntheticNIAH(tokenizer, seed=seed + 10_000).dataset(
        int(config.data.get('num_val_docs', 100)), lengths, 'validation')
    print(f'-> synthetic NIAH: {len(train)} train / {len(validation)} held-out '
          f'at lengths {lengths}; repeated-filler haystack', flush=True)

    collate = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return {name: DataLoader(_Rows(rows), batch_size=1, num_workers=0,
                             shuffle=(name == 'train'), collate_fn=collate,
                             pin_memory=True)
            for name, rows in (('train', train), ('validation', validation),
                               ('test', validation))}
