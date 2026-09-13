"""Similarity statistics and readout instrumentation on a small real model."""
import unittest
from unittest.mock import patch

import torch

from diag_similarity import (
    aggregate_pairs, center_gram, collect_readouts, pairwise_similarity,
    query_positions,
)
from test_shared_memory import tiny_model


class SimilarityTests(unittest.TestCase):
    def test_query_alignment_and_sampling(self):
        mask = torch.ones(17, dtype=torch.bool)
        positions = query_positions(mask, 4, 99, 1)
        self.assertEqual(positions.tolist(), list(range(4, 16)))
        selected = query_positions(mask, 4, 5, 1)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected.tolist(), query_positions(mask, 4, 5, 1).tolist())
        self.assertTrue(torch.isin(selected, positions).all())

    def test_cka_invariance_and_signed_cosine(self):
        torch.manual_seed(3)
        x = torch.randn(64, 12, dtype=torch.float64)
        rotation = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64))[0]
        features = torch.stack([x, x @ rotation * 3 + 7, -x])
        stats = pairwise_similarity(features, shuffles=20)
        for metric in ('linear_cka', 'debiased_cka'):
            torch.testing.assert_close(stats[metric], torch.ones(3, 3, dtype=torch.float64))
        self.assertAlmostEqual(stats['mean_cosine'][0, 2].item(), -1)
        self.assertLess(stats['shuffled_debiased_cka'].abs().max().item(), .1)

    def test_high_dimension_bias_control(self):
        torch.manual_seed(7)
        stats = pairwise_similarity(torch.randn(2, 64, 2048))
        self.assertGreater(stats['linear_cka'][0, 1].item(), .9)
        self.assertLess(abs(stats['debiased_cka'][0, 1].item()), .1)

    def test_undefined_and_invalid_inputs(self):
        stats = pairwise_similarity(torch.zeros(2, 8, 4))
        for value in stats.values():
            self.assertTrue(torch.isnan(value).all())
        for x in (torch.zeros(2, 3, 4), torch.full((2, 8, 4), float('nan'))):
            with self.assertRaises(ValueError):
                pairwise_similarity(x)

    def test_u_centering(self):
        torch.manual_seed(4)
        x = torch.randn(2, 20, 8, dtype=torch.float64)
        gram = center_gram(x @ x.transpose(-1, -2), debiased=True)
        torch.testing.assert_close(gram.sum(-1), torch.zeros(2, 20, dtype=torch.float64))
        self.assertTrue((gram.diagonal(dim1=-2, dim2=-1) == 0).all())

    def test_aggregation_missing_values(self):
        rows = []
        for value in (.2, .6, None):
            row = dict(corpus='test', slice='other', component='current', layer_i=0, layer_j=1)
            row.update({k: value for k in ('linear_cka', 'debiased_cka', 'shuffled_debiased_cka', 'mean_cosine')})
            rows.append(row)
        summary = aggregate_pairs(rows)[0]
        self.assertEqual(summary['n_sequences'], 3)
        self.assertEqual(summary['debiased_cka']['n_valid'], 2)
        self.assertAlmostEqual(summary['debiased_cka']['mean'], .4)

    @torch.no_grad()
    def test_real_model_history_and_output_invariance(self):
        for groups in (None, [[0, 1, 2]]):
            with self.subTest(groups=groups):
                model = tiny_model(groups).eval()
                ids = torch.randint(1, 32, (1, 16))
                mods = [layer.self_attn for layer in model.model.layers]
                weights = {name: value.clone() for name, value in model.state_dict().items()}
                expected = model(ids, use_cache=False).logits
                observed = []
                hook = model.register_forward_hook(lambda m, i, o: observed.append(o.logits.clone()))
                try:
                    features = collect_readouts(model, mods, ids, torch.arange(16))
                finally:
                    hook.remove()
                torch.testing.assert_close(observed[0], expected, atol=0, rtol=0)
                for layer, values in features.items():
                    self.assertEqual(values['current'].shape, (16, 16))
                    torch.testing.assert_close(values['history_effect'][:4], torch.zeros(4, 16), atol=1e-6, rtol=0)
                    self.assertGreater(values['history_effect'][4:].norm().item(), 1e-5)
                    torch.testing.assert_close(values['current'] - values['initial'], values['history_effect'])
                    self.assertFalse(mods[layer].ttt_norm._forward_hooks)
                    self.assertFalse(mods[layer]._forward_hooks)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, weights[name], atol=0, rtol=0)

                # Check the claimed residual-stream contribution against an
                # actual intervention at the last layer (same upstream inputs).
                outputs = []
                hook = mods[-1].register_forward_hook(lambda m, i, o: outputs.append(o[0].clone()))
                try:
                    model(ids, use_cache=False)
                    mods[-1]._ablate_ttt = True
                    model(ids, use_cache=False)
                finally:
                    mods[-1]._ablate_ttt = False
                    hook.remove()
                torch.testing.assert_close(
                    (outputs[0] - outputs[1])[0], features[3]['current'], atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_hooks_removed_after_error(self):
        model = tiny_model().eval()
        mods = [layer.self_attn for layer in model.model.layers]
        with patch('diag_similarity.initial_readout', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                collect_readouts(model, mods, torch.ones(1, 8, dtype=torch.long), torch.arange(4))
        for layer in mods:
            for module in (layer, layer.q_proj, layer.ttt_norm, layer.ttt_scale_proj):
                self.assertFalse(module._forward_hooks)


if __name__ == '__main__':
    unittest.main()
