import unittest
from types import SimpleNamespace

import torch

from analyze_reader_maps import apply_reader_component, polar_factors
from plot_reader_rotation import angles, planes, select_example, strongest_plane


class ReaderGeometryTests(unittest.TestCase):
    def test_invariant_plane_and_projected_stages_with_leakage(self):
        # Embed a 60-degree plane in a rotated four-dimensional coordinate frame.
        q = torch.eye(4, dtype=torch.float64)
        q[:2, :2] = torch.tensor([[.5, -3 ** .5 / 2], [3 ** .5 / 2, .5]])
        q, _, _ = polar_factors(q)  # remove construction's float32 rounding
        frame, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(42)))
        q = frame @ q @ frame.T
        basis, reduced = strongest_plane(q)
        torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.float64))
        torch.testing.assert_close(q @ basis, basis @ reduced)
        self.assertAlmostEqual(torch.atan2(reduced[1, 0], reduced[0, 0]).rad2deg().item(),
                               60., places=5)
        p = torch.diag(torch.tensor([.5, 1., 2., 3.], dtype=torch.float64))
        projected = basis.T @ p @ basis
        torch.testing.assert_close(basis.T @ q @ p @ basis, reduced @ projected)
        self.assertGreater((p @ basis - basis @ projected).norm().item(), .01)

    def test_identity_and_half_turn_planes(self):
        for q in (torch.eye(4, dtype=torch.float64),
                  torch.diag(torch.tensor([-1., -1., 1., 1.], dtype=torch.float64))):
            basis, reduced = strongest_plane(q)
            torch.testing.assert_close(q @ basis, basis @ reduced)
        self.assertEqual(planes(angles(torch.eye(4).unsqueeze(0))).item(), 0.)
        with self.assertRaisesRegex(ValueError, 'reflection'):
            angles(torch.diag(torch.tensor([-1., 1.])).unsqueeze(0))

    def test_example_selection_and_projection(self):
        chi = torch.tensor([[[0., -1.], [2., 0.]]], dtype=torch.float64)
        example = select_example({31: dict(chi=chi, ang=angles(chi))})
        self.assertEqual((example['layer'], example['head']), (31, 0))
        self.assertAlmostEqual(example['angle_degrees'], 90.)
        self.assertAlmostEqual(example['stretch_out_of_plane_energy_fraction'], 0.)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            select_example({31: dict(chi=chi, ang=angles(chi))}, head=-1)

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
