import unittest
from types import SimpleNamespace

import torch

from analyze_reader_maps import apply_reader_component, polar_factors


class ReaderGeometryTests(unittest.TestCase):
    def test_recovers_noncommuting_rotation_and_stretch(self):
        q = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        p = torch.tensor([[2., .5], [.5, 1.]], dtype=torch.float64)
        chi = q @ p
        got_q, got_p, _ = polar_factors(chi.unsqueeze(0))
        torch.testing.assert_close(got_q[0], q)
        torch.testing.assert_close(got_p[0], p)
        # The model stores row readouts and multiplies by weight.T.
        y = torch.tensor([[1., 3.]], dtype=torch.float64)
        torch.testing.assert_close(y @ chi.T, (y @ got_p[0].T) @ got_q[0].T)

    def test_identity_rotation_stretch_reflection_and_singular(self):
        matrices = torch.stack([
            torch.eye(2), torch.tensor([[0., -1.], [1., 0.]]),
            torch.diag(torch.tensor([2., .5])), torch.diag(torch.tensor([-1., 1.])),
            torch.diag(torch.tensor([0., 1.]))])
        q, p, _ = polar_factors(matrices)
        torch.testing.assert_close(q @ p, matrices.double())
        torch.testing.assert_close(q.transpose(-1, -2) @ q,
                                   torch.eye(2, dtype=torch.float64).expand(5, 2, 2))
        torch.testing.assert_close(p, p.transpose(-1, -2))
        self.assertTrue((torch.linalg.eigvalsh(p) >= 0).all())
        self.assertLess(torch.linalg.det(q[3]).item(), 0)
        torch.testing.assert_close(p[1], torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(q[2], torch.eye(2, dtype=torch.float64))

    def test_control_replaces_only_reader_and_preserves_dtype(self):
        for component in ('orthogonal', 'stretch'):
            reader = torch.nn.Module()
            reader.weight = torch.nn.Parameter(torch.tensor([[[0., -1.], [2., 0.]]]))
            module = SimpleNamespace(ttt_reader_alignment=reader,
                                     _identity_reader_alignment=True,
                                     other=torch.tensor([7.]))
            expected = polar_factors(reader.weight)[0 if component == 'orthogonal' else 1]
            count = apply_reader_component([module, SimpleNamespace()], component)
            self.assertEqual(count, 1)
            torch.testing.assert_close(reader.weight, expected.float())
            self.assertFalse(module._identity_reader_alignment)
            self.assertEqual(module.other.item(), 7.)
        with self.assertRaisesRegex(ValueError, 'enabled reader maps'):
            apply_reader_component([], 'orthogonal')
        with self.assertRaisesRegex(ValueError, 'finite'):
            polar_factors(torch.full((2, 2), float('nan')))


if __name__ == '__main__':
    unittest.main()
