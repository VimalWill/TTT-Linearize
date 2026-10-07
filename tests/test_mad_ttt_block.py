"""The MAD TTT block, with causality as the load-bearing check.

Cross-layer sharing was acausal once in this project: followers read a memory
fitted on the whole sequence, so every shared number was contaminated. The
trajectory fix is what prevents it, and these tests pin it -- for the writer,
which fits the memory, and for a reader, which queries the writer's state.
"""
import unittest

import torch

from LinearTTT.mad.ttt_block import TTTBlock


def perturbed_prefix_matches(block, x, cut, run_first=None):
    """Change token `cut` and check that outputs before it are unchanged."""
    with torch.no_grad():
        if run_first is not None:
            run_first(x)
        before = block(x)
        y = x.clone()
        y[:, cut] += 10.0
        if run_first is not None:
            run_first(y)
        after = block(y)
    return (before[:, :cut] - after[:, :cut]).abs().max().item()


class MADTTTBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 96, 32)

    def test_output_shape_and_finiteness(self):
        block = TTTBlock(dim=32, heads=4, chunk_size=16)
        out = block(self.x)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_writer_is_causal(self):
        block = TTTBlock(dim=32, heads=4, chunk_size=16)
        for cut in (17, 48, 80):
            drift = perturbed_prefix_matches(block, self.x, cut)
            self.assertLess(drift, 1e-5, f'writer leaked position {cut}')

    def test_reader_is_causal(self):
        """The reader queries the writer's trajectory, never its final state.

        A final state has seen the whole sequence; handing that to another
        layer leaks the future. This is the check that failed before the
        trajectory fix.
        """
        writer = TTTBlock(dim=32, heads=4, chunk_size=16)
        reader = TTTBlock(dim=32, heads=4, chunk_size=16).share_with(writer)
        for cut in (17, 48, 80):
            drift = perturbed_prefix_matches(reader, self.x, cut, run_first=writer)
            self.assertLess(drift, 1e-5, f'reader leaked position {cut}')

    def test_reader_without_writer_raises(self):
        writer = TTTBlock(dim=32, heads=4, chunk_size=16)
        reader = TTTBlock(dim=32, heads=4, chunk_size=16).share_with(writer)
        with self.assertRaises(RuntimeError):
            reader(self.x)

    def test_reader_keeps_private_parameters(self):
        """Only the fast weights are shared -- that is what makes state cheap."""
        writer = TTTBlock(dim=32, heads=4, chunk_size=16)
        reader = TTTBlock(dim=32, heads=4, chunk_size=16).share_with(writer)
        shared = {id(p) for p in (writer.w0, writer.w1, writer.w2)}
        for name in ('w0', 'w1', 'w2', 'qkv', 'out', 'gate_proj'):
            self.assertNotIn(id(getattr(reader, name)), shared)
        self.assertNotIn(reader, writer.children())

    def test_fast_weights_receive_gradient(self):
        block = TTTBlock(dim=32, heads=4, chunk_size=16)
        block(self.x).square().mean().backward()
        for name in ('w0', 'w1', 'w2'):
            grad = getattr(block, name).grad
            self.assertIsNotNone(grad, f'{name} got no gradient')
            self.assertGreater(grad.abs().max().item(), 0.0, f'{name} gradient is zero')

    def test_dim_must_divide_into_heads(self):
        with self.assertRaises(ValueError):
            TTTBlock(dim=30, heads=4)

    def test_block_cannot_share_with_itself(self):
        block = TTTBlock(dim=32, heads=4)
        with self.assertRaises(ValueError):
            block.share_with(block)


if __name__ == '__main__':
    unittest.main()
