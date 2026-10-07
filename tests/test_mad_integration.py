"""Initialization/accounting and hybrid objective regressions without Lightning."""
import ast
import copy
from pathlib import Path
import unittest

import torch
from torch import nn

from LinearTTT.mad.attention import SDPAAttention
from LinearTTT.mad.hybrid import TTTHybridBlock, hybridize_mad_model
from LinearTTT.mad.objective import configure_objective, mad_loss
from LinearTTT.mad.register import PreserveTTTInitialization, total_state_dim
from LinearTTT.mad.ttt_block import TTTBlock


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 32)
        self.layers = nn.ModuleList([SDPAAttention(32, n_heads=4, rotary_emb_dim=8)
                                     for _ in range(4)])
        self.out = nn.Linear(32, 16)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = x + layer(x)
        return self.out(x)


class MADIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_both_upstream_initializers_preserve_ttt_defaults(self):
        # Execute the actual upstream initialization methods, isolated from
        # their optional Lightning/Triton imports; do not duplicate their logic.
        root = Path(__file__).resolve().parents[1] / 'third_party/mad-lab/mad/model'
        for filename in ('language_model.py', 'auto_encoder.py'):
            tree = ast.parse((root / filename).read_text())
            method = next(n for n in ast.walk(tree)
                          if isinstance(n, ast.FunctionDef) and n.name == '_init_weights')
            namespace = {'nn': nn}
            exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'), namespace)
            base = type('UpstreamInit', (nn.Module,), {'_init_weights': namespace['_init_weights']})
            compatible = type('Compatible', (PreserveTTTInitialization, base), {})()
            compatible.block = TTTBlock(32)
            compatible.apply(compatible._init_weights)
            self.assertTrue((compatible.block.gate_proj.bias == -3).all())
            self.assertTrue((compatible.block.retention_proj[0].bias == 4).all())
            self.assertEqual(compatible.block.gate_proj.weight.count_nonzero().item(), 0)
            self.assertEqual(compatible.block.retention_proj[0].weight.count_nonzero().item(), 0)

    def test_sharing_reduces_unique_parameters_and_survives_reload(self):
        blocks = nn.ModuleList([TTTBlock(32, chunk_size=16) for _ in range(3)])
        before = sum(p.numel() for p in blocks.parameters())
        state = blocks[0].state_dim()
        for reader in blocks[1:]:
            reader.share_with(blocks[0])
        self.assertEqual(sum(p.numel() for p in blocks.parameters()), before - 2 * state)
        self.assertEqual(total_state_dim(blocks), state)
        saved = copy.deepcopy(blocks.state_dict())
        rebuilt = nn.ModuleList([TTTBlock(32, chunk_size=16) for _ in range(3)])
        for reader in rebuilt[1:]:
            reader.share_with(rebuilt[0])
        rebuilt.load_state_dict(saved)
        x = torch.randn(2, 64, 32)
        rebuilt[0](x)
        rebuilt[2](x).square().mean().backward()
        self.assertIs(rebuilt[2].w0, rebuilt[0].w0)
        self.assertGreater(rebuilt[0].w0.grad.abs().sum().item(), 0)

    def test_sharing_rejects_incompatible_geometry(self):
        writer = TTTBlock(32, chunk_size=16)
        for other in (TTTBlock(32, chunk_size=32), TTTBlock(32, num_heads=2)):
            with self.assertRaises(ValueError):
                other.share_with(writer)

    def test_hybrid_conversion_and_distillation_gradients(self):
        teacher = SmallModel()
        model = hybridize_mad_model(copy.deepcopy(teacher), hybrid_window=16,
                                    swa_window=32, chunk_size=16)
        self.assertEqual(sum(isinstance(b, TTTHybridBlock) for b in model.layers), 3)
        self.assertEqual(model.layers[3].window, (32, 0))
        self.assertIs(model.layers[0].w0, model.layers[2].w0)
        for old, new in zip(teacher.layers[:3], model.layers[:3]):
            torch.testing.assert_close(old.Wqkv.weight, new.qkv.weight)
            self.assertEqual(new.attention.window, (16, 0))
        configure_objective(model, 'distill')
        tokens = torch.randint(0, 16, (2, 64))
        targets = tokens.clone()
        targets[:, :32] = -100
        loss, logits, metrics = mad_loss(model, tokens, targets, 'distill', 3., .5)
        torch.testing.assert_close(loss.detach(), .5 * metrics['loss_ce'] + 3 * metrics['loss_mse'])
        expected_ce = nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        torch.testing.assert_close(metrics['loss_ce'], expected_ce.detach())
        loss.backward()
        self.assertIsNone(model.layers[0].qkv.weight.grad)
        self.assertIsNone(model.embedding.weight.grad)
        self.assertGreater(model.layers[0].w0.grad.abs().sum().item(), 0.)
        configure_objective(model, 'task')
        loss, _, metrics = mad_loss(model, tokens, targets)
        torch.testing.assert_close(loss.detach(), metrics['loss_ce'])

    def test_hybrid_swa_branch_is_causal(self):
        model = hybridize_mad_model(SmallModel(), hybrid_window=16,
                                    swa_window=32, chunk_size=16).eval()
        x = torch.randint(0, 16, (1, 64))
        y = x.clone()
        y[:, 48:] = (y[:, 48:] + 1) % 16
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :48], model(y)[:, :48])


if __name__ == '__main__':
    unittest.main()
