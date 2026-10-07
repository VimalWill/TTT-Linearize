"""The SDPA attention that replaces MAD's flash-attn wrapper.

It exists because mad/model/layers/__init__.py imports flash_attn at import
time, and flash-attn has no aarch64 wheels, so the registry is unreachable on
the GH200 nodes. Being a replacement, it has to agree with the semantics it
replaces: flash-attn's (left, right) window convention, partial rotary, and
causality.
"""
import unittest

import torch

from LinearTTT.mad.attention import SDPAAttention


class SDPAAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 64, 64)

    def influence(self, layer, source, target):
        """How much perturbing token `source` moves the output at `target`."""
        with torch.no_grad():
            perturbed = self.x.clone()
            perturbed[:, source] += 50.0
            return (layer(self.x)[:, target] - layer(perturbed)[:, target]).abs().max().item()

    def test_shape_is_preserved(self):
        for heads, rotary in ((1, 0), (4, 8), (16, 0)):
            layer = SDPAAttention(dim=64, n_heads=heads, rotary_emb_dim=rotary)
            self.assertEqual(layer(self.x).shape, self.x.shape)

    def test_causal(self):
        layer = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=8).eval()
        self.assertEqual(self.influence(layer, source=40, target=32), 0.0)

    def test_window_bounds_are_flash_attn_semantics(self):
        """(left, right): a query sees i-left .. i+right, -1 meaning unbounded."""
        layer = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=0,
                              window_size=(8, -1)).eval()
        self.assertGreater(self.influence(layer, 36, 40), 0.0, 'inside the window')
        self.assertGreater(self.influence(layer, 32, 40), 0.0, 'the far edge is inside')
        self.assertEqual(self.influence(layer, 31, 40), 0.0, 'one past the edge')
        self.assertEqual(self.influence(layer, 10, 40), 0.0, 'far outside')

    def test_mask_path_matches_is_causal_path(self):
        """An unbounded window takes a fast path; both must give one answer."""
        layer = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=0).eval()
        with torch.no_grad():
            fast = layer(self.x)
            layer.window = (self.x.shape[1], -1)   # same constraint, dense mask
            masked = layer(self.x)
        self.assertLess((fast - masked).abs().max().item(), 1e-6)

    def test_rotary_changes_the_output(self):
        torch.manual_seed(1)
        plain = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=0).eval()
        torch.manual_seed(1)
        rotary = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=8).eval()
        with torch.no_grad():
            self.assertGreater((plain(self.x) - rotary(self.x)).abs().max().item(), 1e-4)

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            SDPAAttention(dim=64, n_heads=5)                       # does not divide
        with self.assertRaises(ValueError):
            SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=7)     # odd rotary
        with self.assertRaises(ValueError):
            SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=64)    # exceeds head dim

    def test_accepts_mads_unused_config_keys(self):
        """MAD passes the whole layer config as kwargs; extras must not raise."""
        layer = SDPAAttention(dim=64, n_heads=4, rotary_emb_dim=8, max_length=1280,
                              use_flash_attn=True, dwconv=False, use_alibi=False)
        self.assertEqual(layer(self.x).shape, self.x.shape)


if __name__ == '__main__':
    unittest.main()
