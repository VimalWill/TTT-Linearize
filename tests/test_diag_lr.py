import unittest

import torch
from torch import nn
from torch.nn import functional as F

from diag_lr import probe, stats


class FunctionalLRLayer(nn.Module):
    """Exercise the functional projection that bypasses Linear's hooks."""

    def __init__(self):
        super().__init__()
        self.layer_idx = 2
        self.lr_proj = nn.Linear(2, 3).to(torch.bfloat16)
        self.base_lr_inv = -4.0
        self.retention_proj = nn.Sequential(
            nn.Linear(2, 1), nn.Sigmoid()).to(torch.bfloat16)

    def forward(self, hidden_states):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            lr = F.linear(hidden_states.float(), self.lr_proj.weight.float(),
                          self.lr_proj.bias.float())
        self.actual_lr = F.softplus(lr + self.base_lr_inv)
        return self.actual_lr


class ProbeModel(nn.Module):
    device = torch.device('cpu')

    def __init__(self, fail=False):
        super().__init__()
        self.layer = FunctionalLRLayer()
        self.fail = fail

    def forward(self, input_ids, use_cache=False):
        result = self.layer(hidden_states=input_ids)
        if self.fail:
            raise RuntimeError('simulated forward failure')
        return result


class LRProbeTests(unittest.TestCase):
    def test_matches_functional_fp32_lr_under_autocast(self):
        model = ProbeModel()
        hidden = torch.tensor([[[1.25, -0.5], [0.75, 2.0]]], dtype=torch.bfloat16)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            measured = probe(model, [model.layer], hidden)
        self.assertEqual(set(measured), {2})
        self.assertEqual(measured[2]['lr'], stats(model.layer.actual_lr))
        self.assertFalse(model.layer._forward_pre_hooks)

    def test_removes_hooks_when_model_fails(self):
        model = ProbeModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'simulated forward failure'):
            probe(model, [model.layer], torch.ones(1, 2, 2, dtype=torch.bfloat16))
        self.assertFalse(model.layer._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
