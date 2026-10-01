"""Cached inference correctness and bounded state, without GPU/downloads."""
from types import SimpleNamespace
import unittest

import torch
from transformers import MistralConfig, MistralForCausalLM

from measure_flops import (
    cache_storage_bytes, cached_request, count_cached_inference, inference_kwargs,
)


class CachedEfficiencyTests(unittest.TestCase):
    def test_fresh_requests_reuse_cache_and_feed_only_one_token_during_decode(self):
        class Model:
            def __init__(self):
                self.calls = []
                self.caches = []

            def forward(self, input_ids, use_cache, past_key_values=None,
                        num_logits_to_keep=0, return_dict=True):
                if past_key_values is None:
                    past_key_values = SimpleNamespace(length=0)
                    self.caches.append(past_key_values)
                past_key_values.length += input_ids.shape[1]
                self.calls.append((input_ids.shape[1], use_cache,
                                   num_logits_to_keep, past_key_values))
                return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], 1, 32),
                                       past_key_values=past_key_values)

            __call__ = forward

        model = Model()
        for _ in range(2):
            cached_request(model, torch.ones(2, 12, dtype=torch.long), 6,
                           inference_kwargs(model), lambda: None)
        self.assertEqual([c[0] for c in model.calls], [12] + [1] * 6 + [12] + [1] * 6)
        self.assertTrue(all(c[1] is True and c[2] == 1 for c in model.calls))
        self.assertIsNot(model.caches[0], model.caches[1])
        for i, cache in enumerate(model.caches):
            self.assertTrue(all(c[3] is cache for c in model.calls[i*7:(i+1)*7]))
            self.assertEqual(cache.length, 18)

    def test_storage_measurement_counts_view_backing_and_deduplicates_aliases(self):
        source = torch.empty(100, dtype=torch.float32)
        view = source[-2:]
        self.assertEqual(cache_storage_bytes({'k': view, 'alias': view}), 400)
        self.assertEqual(cache_storage_bytes({'k': view.clone()}), 8)

    def test_cached_ttt_logits_match_full_prefix_and_prefill_storage_is_bounded(self):
        from LinearTTT.model.LinearizeMistral.Configuration import LigerMistralGLAConfig
        from LinearTTT.model.LinearizeMistral.LinearizeMistral import LigerMistralGLAForCausalLM
        from LinearTTT.model.LinearizeLlama.Configuration import LigerGLAConfig
        from LinearTTT.model.LinearizeLlama.LinearizeLlama import LigerGLAForCausalLM

        families = ((LigerMistralGLAConfig, LigerMistralGLAForCausalLM),
                    (LigerGLAConfig, LigerGLAForCausalLM))
        for Config, Model in families:
            for arm in ('anchor_f', 'anchor_i', 'anchor'):
                with self.subTest(family=Model.__name__, arm=arm), \
                        torch.no_grad(), torch._dynamo.config.patch(disable=True):
                    torch.manual_seed(0)
                    shared = arm != 'anchor_f'
                    cfg = Config(
                        vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=3, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, lact_chunk_size=4,
                        window_size=4, ttt_inner_loss='l2', ttt_use_muon=True,
                        ttt_share_groups=[[0, 1]] if shared else None,
                        ttt_reader_alignment='linear' if shared else 'none',
                        ttt_layer_indices=[0, 1] if arm == 'anchor' else None)
                    model = Model(cfg).eval()
                    if shared:
                        alignment = model.model.layers[1].self_attn.ttt_reader_alignment
                        alignment.weight.add_(0.1 * torch.randn_like(alignment.weight))
                    kwargs = inference_kwargs(model)
                    ids = torch.randint(32, (2, 30))
                    first = model(input_ids=ids[:, :12], use_cache=True, **kwargs)
                    cache = first.past_key_values
                    retained = cache_storage_bytes(cache)
                    longer = model(input_ids=ids[:, :24], use_cache=True, **kwargs)
                    self.assertEqual(cache_storage_bytes(longer.past_key_values), retained)
                    del longer
                    # Includes updates immediately after a full pending chunk,
                    # another complete chunk, readers, private writers and SWA.
                    for position in range(12, 18):
                        output = model(input_ids=ids[:, position:position+1],
                                       past_key_values=cache, use_cache=True, **kwargs)
                        reference = model(input_ids=ids[:, :position+1],
                                          use_cache=False, **kwargs)
                        torch.testing.assert_close(output.logits, reference.logits,
                                                   atol=2e-6, rtol=2e-5)
                        self.assertEqual(cache.get_seq_length(), position + 1)
                        for state in cache.states.values():
                            self.assertLessEqual(state['k'].shape[2], 5)
                            self.assertLessEqual(state['v'].shape[2], 5)

    def test_hf_baseline_and_cached_counter_support_last_position_logits(self):
        model = MistralForCausalLM(MistralConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=64, sliding_window=4)).eval()
        ids = torch.ones(2, 12, dtype=torch.long)
        result = cached_request(model, ids, 6, inference_kwargs(model), lambda: None)
        self.assertGreater(result['prefill_cache_bytes'], 0)
        self.assertGreater(result['decode_cache_bytes'], 0)
        prefill, decode = count_cached_inference(model, ids, 6, inference_kwargs(model))
        self.assertGreater(prefill, 0)
        self.assertGreater(decode, 0)


if __name__ == '__main__':
    unittest.main()
