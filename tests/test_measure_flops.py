"""Efficiency counter checks without CUDA or checkpoint downloads."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from measure_flops import count_forward, load_benchmark_config, set_window_backend


class FlopMeasurementTests(unittest.TestCase):
    def test_checkpoint_override_precedes_required_env_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cfg.yml'
            path.write_text('model:\n  pretrained_model_name_or_path: ${oc.env:TTT_INIT}\n'
                            '  max_length: 2048\n')
            with patch.dict(os.environ, {}, clear=True):
                config = load_benchmark_config(path, '/trained/checkpoint', 8192)
        self.assertEqual(config.model.pretrained_model_name_or_path, '/trained/checkpoint')
        self.assertEqual(config.model.max_length, 8192)

    def test_window_switch_reaches_both_backbones_and_restores(self):
        from LinearTTT.model.LinearizeLlama import LinearizeLlama as llama
        from LinearTTT.model.LinearizeMistral import LinearizeMistral as mistral
        before = llama._force_sdpa_window, mistral._force_sdpa_window
        try:
            set_window_backend(True)
            self.assertTrue(llama._force_sdpa_window)
            self.assertTrue(mistral._force_sdpa_window)
            set_window_backend(False)
            self.assertFalse(llama._force_sdpa_window)
            self.assertFalse(mistral._force_sdpa_window)
        finally:
            llama.use_sdpa_sliding_window(before[0])
            mistral.use_sdpa_sliding_window(before[1])

    def test_counter_includes_nested_compiled_newton_schulz_iterations(self):
        from LinearTTT.model.LinearizeLlama.ttt_ops import zeropower_via_newtonschulz5

        class Orthogonalize(torch.nn.Module):
            def forward(self, input_ids, use_cache):
                return zeropower_via_newtonschulz5(input_ids)

        # The vendored routine runs five iterations, with three square bmm
        # operations each. Count those dispatched operations, not just the
        # containing compiled call. Unregistered elementwise work is excluded.
        batch, width = 2, 4
        before = torch._dynamo.config.disable
        total = count_forward(Orthogonalize(), torch.randn(batch, width, width))
        self.assertEqual(total, 5 * 3 * 2 * batch * width ** 3)
        self.assertEqual(torch._dynamo.config.disable, before)


if __name__ == '__main__':
    unittest.main()
