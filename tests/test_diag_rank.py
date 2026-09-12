import json
import unittest

import torch

from diag_rank import (collect_states, memory_owners, observe_states, rank_records,
                       spectrum_stats, split_tokens, summarize, validate_tokens)
from test_shared_memory import tiny_model


class RankMetricTests(unittest.TestCase):
    def test_known_spectrum_and_zero(self):
        result = spectrum_stats(torch.tensor([4., 2., 1.]))
        self.assertEqual((result['r90'], result['r95'], result['r99']), (2, 2, 3))
        self.assertAlmostEqual(result['stable_rank'], 21 / 16)
        self.assertEqual(result['rel_error_r8'], 0)
        zero = spectrum_stats(torch.zeros(8))
        self.assertEqual(zero['r99'], 0)
        self.assertIsNone(zero['stable_rank'])
        self.assertIsNone(zero['rel_error_r8'])
        json.dumps(zero, allow_nan=False)

    def test_flat_spectrum_and_tail_error(self):
        result = spectrum_stats(torch.ones(128))
        self.assertEqual(result['r95'], 122)
        self.assertEqual(result['r99'], 127)
        self.assertAlmostEqual(result['stable_rank'], 128)
        self.assertAlmostEqual(result['rel_error_r32'], (96 / 128) ** .5)

    def test_identical_token_partition_and_validation(self):
        tokens = split_tokens(list(range(16)), 4, 3)
        self.assertEqual(tokens.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        validate_tokens({'test': tokens}, 4, 3, 16)
        with self.assertRaises(ValueError):
            split_tokens([1, 2], 4, 1)
        with self.assertRaises(ValueError):
            validate_tokens({'test': tokens}, 4, 3, 10)


class RankObservationTests(unittest.TestCase):
    @torch.no_grad()
    def test_observation_preserves_outputs_and_collects_only_owners(self):
        for groups in (None, [[0, 1, 2]], [[0, 1], [2, 3]]):
            with self.subTest(groups=groups):
                model = tiny_model(groups, muon=True).eval()
                mods = [layer.self_attn for layer in model.model.layers]
                ids = torch.randint(1, 32, (1, 13))
                before = model(ids, use_cache=False).logits
                parameters = {n: p.clone() for n, p in model.named_parameters()}
                captured = []
                with observe_states(mods, lambda m, traj: captured.append(m.layer_idx)):
                    observed = model(ids, use_cache=False).logits
                torch.testing.assert_close(observed, before, rtol=0, atol=0)
                self.assertEqual(captured, [m.layer_idx for m in memory_owners(mods)])
                snapshots = collect_states(model.model, mods, ids, [0, 1, 3, 4])
                self.assertEqual(len(snapshots), len(memory_owners(mods)) * 3 * 3)
                self.assertEqual({s['chunk'] for s in snapshots}, {0, 1, 3})
                self.assertTrue(all(s['tokens_written'] == s['chunk'] * 4 for s in snapshots))
                for s in snapshots:
                    self.assertEqual(tuple(s['states'].shape), (2, 8, 8))
                    if s['chunk'] == 0:
                        torch.testing.assert_close(s['states'], getattr(mods[s['layer']], s['matrix']))
                # Snapshots are detached copies, not views of live model weights.
                snapshots[0]['states'].zero_()
                for name, p in model.named_parameters():
                    torch.testing.assert_close(p, parameters[name], rtol=0, atol=0)
                for group in groups or []:
                    self.assertIs(mods[group[0]].w0, mods[group[-1]].w0)
                self.assertTrue(all(not hasattr(m, '_ttt_state_observer') for m in mods))

    @torch.no_grad()
    def test_short_sequence_has_only_initial_state_and_records_each_head(self):
        model = tiny_model([[0, 1, 2]]).eval()
        mods = [layer.self_attn for layer in model.model.layers]
        snapshots = collect_states(model.model, mods, torch.ones(1, 4, dtype=torch.long), [0, 1, 4])
        self.assertEqual(len(snapshots), 2 * 3)
        self.assertEqual({s['chunk'] for s in snapshots}, {0})
        records = list(rank_records(snapshots, 'test', 0, 'cpu'))
        self.assertEqual(len(records), 2 * 3 * 2)
        self.assertEqual(len(summarize(records)), 2 * 3)
        json.dumps(records, allow_nan=False)

    @torch.no_grad()
    def test_observer_restoration_on_failure_and_cached_prefill(self):
        model = tiny_model([[0, 1, 2]], muon=True).eval()
        mods = [layer.self_attn for layer in model.model.layers]
        ids = torch.ones(1, 9, dtype=torch.long)
        reference = model(ids, use_cache=True)
        observed_layers = []
        with observe_states(mods, lambda m, traj: observed_layers.append(m.layer_idx)):
            result = model(ids, use_cache=True)
        self.assertEqual(observed_layers, [0, 3])
        torch.testing.assert_close(reference.logits, result.logits, rtol=0, atol=0)
        self.assertNotIn('w0', result.past_key_values.states[1])
        sentinel = lambda *args: None
        mods[0]._ttt_state_observer = sentinel
        def fail(*args):
            raise RuntimeError('observer failed')
        with self.assertRaisesRegex(RuntimeError, 'observer failed'):
            with observe_states(mods, fail):
                model(ids, use_cache=False)
        self.assertIs(mods[0]._ttt_state_observer, sentinel)
        self.assertFalse(hasattr(mods[3], '_ttt_state_observer'))


if __name__ == '__main__':
    unittest.main()
