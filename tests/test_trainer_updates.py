"""Exercise the real training loop with known gradients and optimizer state."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tqdm import tqdm

from Training.trainer import DefaultTrainer


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=.1, momentum=.9)
        self.updates = []

    def step(self, closure=None):
        self.updates.append(self.param_groups[0]['params'][0].grad.detach().clone())
        return super().step(closure)


class BrokenBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, kind):
        ctx.kind = kind
        ctx.save_for_backward(weight)
        return weight.new_tensor(1.)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.kind == 'backward_error':
            raise RuntimeError('deliberate backward failure')
        weight, = ctx.saved_tensors
        return torch.full_like(weight, float(ctx.kind)) * grad_output, None


def build_trainer(batches, accumulation=8, max_norm=1.):
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    trainer.args = SimpleNamespace(num_train_epochs=2, max_grad_norm=max_norm)
    trainer.gradient_accumulation_steps = accumulation
    trainer.train_loader = batches
    trainer.initial_eval = False
    trainer.compute_loss_backprop = False
    trainer.optimizer = RecordingSGD(model.parameters())
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=100)
    trainer.scheduler_step_after_epoch = False
    trainer.step = trainer.grad_step = 0
    trainer.max_steps = -1
    trainer.logging_steps = 1
    trainer.eval_strategy = 'no'

    def compute_loss(m, batch, return_outputs):
        if isinstance(batch, str):
            if batch == 'nonfinite_loss':
                loss = m.weight.sum() * float('nan')
            else:
                loss = BrokenBackward.apply(m.weight, batch)
        else:
            loss = (m.weight * m.weight.new_tensor(batch)).sum()
        return loss, {}

    trainer.compute_loss = compute_loss
    return trainer, model


class TrainerUpdateTests(unittest.TestCase):
    def setUp(self):
        quiet = patch('Training.trainer.tqdm', side_effect=lambda seq, **kw: tqdm(seq, disable=True))
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_clips_before_optimizer_step(self):
        trainer, model = build_trainer([[30000., 40000.]], accumulation=1)
        trainer.train_step(model, 0)
        torch.testing.assert_close(trainer.optimizer.updates[0], torch.tensor([[.6, .8]], dtype=torch.float64))
        self.assertEqual(trainer.grad_step, 1)

    def test_clips_after_averaging_accumulated_gradients(self):
        trainer, model = build_trainer([[6., 8.], [-4., -8.]], accumulation=2, max_norm=.5)
        trainer.train_step(model, 0)
        torch.testing.assert_close(trainer.optimizer.updates[0], torch.tensor([[.5, 0.]], dtype=torch.float64),
                                   atol=1e-6, rtol=1e-6)

    def test_partial_window_and_epoch_boundaries(self):
        trainer, model = build_trainer([[float(i), 0.] for i in range(1, 10)], max_norm=0)
        trainer.train_step(model, 0)
        self.assertIsNone(model.weight.grad)
        trainer.train_step(model, 1)
        observed = torch.cat(trainer.optimizer.updates)
        expected = torch.tensor([[4.5, 0.], [9., 0.], [4.5, 0.], [9., 0.]], dtype=torch.float64)
        torch.testing.assert_close(observed, expected)
        self.assertEqual(trainer.step, 18)
        self.assertEqual(trainer.grad_step, 4)
        self.assertEqual(trainer.scheduler.last_epoch, 4)

    def test_epoch_shorter_than_accumulation_window(self):
        trainer, model = build_trainer([[2., 0.], [4., 0.], [6., 0.]], max_norm=None)
        trainer.train_step(model, 0)
        torch.testing.assert_close(trainer.optimizer.updates[0], torch.tensor([[4., 0.]], dtype=torch.float64))
        self.assertEqual(trainer.grad_step, 1)

    def test_failed_batch_resets_accumulation_window(self):
        for failure in ('nonfinite_loss', 'backward_error'):
            with self.subTest(failure=failure):
                trainer, model = build_trainer([[2., 0.], failure, [4., 0.], [8., 0.]],
                                                accumulation=2, max_norm=0)
                trainer.train_step(model, 0)
                self.assertEqual(trainer.grad_step, 1)
                torch.testing.assert_close(trainer.optimizer.updates[0], torch.tensor([[6., 0.]], dtype=torch.float64))

    def test_nonfinite_gradients_skip_optimizer_and_scheduler(self):
        for failure in ('nan', 'inf'):
            with self.subTest(failure=failure):
                trainer, model = build_trainer([[2., 0.], failure, [4., 0.], [8., 0.]],
                                                accumulation=2, max_norm=0)
                trainer.train_step(model, 0)
                self.assertEqual(len(trainer.optimizer.updates), 1)
                self.assertEqual(trainer.grad_step, 1)
                self.assertEqual(trainer.scheduler.last_epoch, 1)
                expected = torch.tensor([[6., 0.]], dtype=torch.float64)
                torch.testing.assert_close(trainer.optimizer.updates[0], expected)
                torch.testing.assert_close(trainer.optimizer.state[model.weight]['momentum_buffer'], expected)
                torch.testing.assert_close(model.weight, -.1 * expected)

    def test_failed_final_batch_leaves_no_pending_gradients(self):
        trainer, model = build_trainer([[2., 0.], 'nonfinite_loss'])
        trainer.train_step(model, 0)
        self.assertEqual(trainer.grad_step, 0)
        self.assertEqual(trainer.scheduler.last_epoch, 0)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(len(trainer.optimizer.state), 0)

    def test_invalid_accumulation_rejected(self):
        for accumulation in (0, -1, 1.5, True):
            with self.subTest(accumulation=accumulation):
                trainer, model = build_trainer([[1., 0.]], accumulation=accumulation)
                with self.assertRaises(ValueError):
                    trainer.train_step(model, 0)


if __name__ == '__main__':
    unittest.main()
