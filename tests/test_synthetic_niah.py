"""Repeat-haystack needle generation.

The point of this data is distribution: it must look like RULER's
niah_single_1, which hides a needle in one sentence repeated thousands of
times. So the tests check the properties that make it that task -- exact
target lengths, answer-only supervision, needles spread across depths rather
than parked at the end, and held-out examples the model has not trained on.
"""
import unittest

from Training.synthetic_niah import FILLER, SyntheticNIAH


class WhitespaceTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        ids = [abs(hash(w)) % 1000 + 10 for w in text.split()]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(f'w{i}' for i in ids)


class SyntheticNIAHTests(unittest.TestCase):
    def setUp(self):
        self.gen = SyntheticNIAH(WhitespaceTokenizer(), seed=0)

    def test_hits_the_target_length(self):
        for length in (512, 2048, 8192):
            row = self.gen.example(length)
            self.assertEqual(len(row['input_ids']), length)

    def test_only_the_answer_is_supervised(self):
        """Training on the haystack is what teaches a model to repeat it."""
        row = self.gen.example(2048)
        prompt = row['prompt_tokens']
        self.assertEqual(row['labels'][:prompt], [-100] * prompt)
        self.assertTrue(all(x != -100 for x in row['labels'][prompt:]))
        self.assertEqual(len(row['labels']), len(row['input_ids']))
        self.assertEqual(len(row['attention_mask']), len(row['input_ids']))

    def test_needle_is_in_the_prompt(self):
        row = self.gen.example(2048)
        self.assertIn(row['answer'], row['prompt'])

    def test_depths_are_spread_not_parked_at_the_end(self):
        """A model trained only on late needles learns recency, not recall."""
        depths = [self.gen.example(1024)['needle_depth'] for _ in range(100)]
        self.assertLess(min(depths), 0.2)
        self.assertGreater(max(depths), 0.7)
        self.assertLess(abs(sum(depths) / len(depths) - 0.45), 0.12)

    def test_deeper_needle_sits_nearer_the_question(self):
        early = self.gen.example(4096, depth=0.05)
        late = self.gen.example(4096, depth=0.85)
        self.assertLess(late['needle_distance'], early['needle_distance'])

    def test_answer_cannot_hide_in_the_filler(self):
        """A value occurring in the haystack would be solvable locally."""
        for _ in range(50):
            self.assertNotIn(self.gen.example(1024)['answer'], FILLER)

    def test_too_short_is_rejected_rather_than_truncated(self):
        with self.assertRaises(ValueError):
            self.gen.example(16)

    def test_separate_seeds_give_disjoint_needles(self):
        """Validation must not reuse a key or value the model trained on."""
        train = SyntheticNIAH(WhitespaceTokenizer(), seed=0).dataset(40, [1024], 'train')
        held = SyntheticNIAH(WhitespaceTokenizer(), seed=10_000).dataset(40, [1024], 'validation')
        self.assertFalse({r['answer'] for r in train} & {r['answer'] for r in held})

    def test_dataset_cycles_the_requested_lengths(self):
        rows = self.gen.dataset(6, [512, 1024], 'train')
        self.assertEqual([r['length_bucket'] for r in rows], [512, 1024] * 3)


if __name__ == '__main__':
    unittest.main()
