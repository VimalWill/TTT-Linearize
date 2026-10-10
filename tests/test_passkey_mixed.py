"""Offline checks for mixed stage-2 data and the actual HF batch collator."""
import unittest
from unittest.mock import patch
from collections import Counter

from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

from Training.reactive_passkey import load_reactive_data
from Training.synthetic_niah import SyntheticNIAH, TRAIN_TEMPLATES, VALIDATION_TEMPLATES


class MixedPasskeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = Tokenizer(models.BPE(unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        raw.train_from_iterator(
            [' '.join(t) for t in TRAIN_TEMPLATES + VALIDATION_TEMPLATES],
            trainers.BpeTrainer(vocab_size=300, special_tokens=['[UNK]', '[EOS]'],
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,
                                               eos_token='[EOS]', pad_token='[EOS]')

    def config(self, fraction=0.2):
        return OmegaConf.create({
            'data': {'path': 'unused', 'subset': 'unused', 'micro_batch_size': 1,
                     'num_val_docs': 4, 'synthetic_fraction': fraction,
                     'synthetic_lengths': [512, 1024, 2048], 'seed': 42},
            'model': {'max_length': 2048}})

    def loaders(self, fraction=0.2):
        def row(i):
            return {'interactions': [
                {'query': f'Find code {i}.', 'answer': str(i)},
                {'query': 'Repeat it.', 'answer': str(i)}]}
        with patch('datasets.load_dataset', return_value=[]), patch(
                'Training.reactive_passkey.prepare_conversations',
                return_value=([row(i) for i in range(12)], [row(i) for i in range(20, 24)])):
            return load_reactive_data(self.config(fraction), self.tokenizer)

    def test_mix_and_real_collator(self):
        loaders = self.loaders()
        self.assertEqual(len(loaders['train'].dataset), 15)
        self.assertEqual(len(loaders['validation'].dataset), 5)
        self.assertEqual(len(loaders['test'].dataset), 4)
        seen = Counter()
        for batch in loaders['train']:
            source = batch['retrieval_sources'][0]
            seen[source] += 1
            self.assertEqual(batch['input_ids'].shape, batch['labels'].shape)
            self.assertTrue((batch['labels'] != -100).any())
            if source == 'repetition':
                labels = batch['labels'][0].tolist()
                first = next(i for i, x in enumerate(labels) if x != -100)
                self.assertNotIn(-100, labels[first:])
                self.assertLessEqual(len(labels), 2048)
        self.assertEqual(seen, {'passkey': 12, 'repetition': 3})
        # Generation view retains only the original held-out first question.
        item = loaders['test'].dataset[0]
        prompt = self.tokenizer.decode(item['input_ids'])
        self.assertNotIn('Repeat it.', prompt)

    def test_no_mix_preserves_original_loader(self):
        loaders = self.loaders(0)
        self.assertEqual(len(loaders['train'].dataset), 12)
        self.assertNotIn('retrieval_sources', next(iter(loaders['train'])))

    def test_bpe_caps_and_unseen_templates(self):
        self.assertFalse(set(TRAIN_TEMPLATES) & set(VALIDATION_TEMPLATES))
        for templates in (TRAIN_TEMPLATES, VALIDATION_TEMPLATES):
            gen = SyntheticNIAH(self.tokenizer, seed=42, templates=templates)
            for length in (4096, 8192, 16384):
                for depth in (0, 0.5, 1):
                    row = gen.example(length, depth)
                    self.assertLessEqual(len(row['input_ids']), length)
                    self.assertGreater(len(row['input_ids']), length - 100)
                    self.assertIn(row['answer'], row['prompt'])
                    self.assertEqual(row['labels'][:row['prompt_tokens']],
                                     [-100] * row['prompt_tokens'])

    def test_invalid_fraction_rejected(self):
        for fraction in (-0.2, 1):
            with self.assertRaises(ValueError):
                self.loaders(fraction)


if __name__ == '__main__':
    unittest.main()
