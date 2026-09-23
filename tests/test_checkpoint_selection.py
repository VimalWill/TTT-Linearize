"""Regression checks for CE-based model selection during attention transfer."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Training.trainer import DefaultTrainer


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='ttt-checkpoint-test-')
        self.addCleanup(self.directory.cleanup)
        args = SimpleNamespace(
            metric_for_best_model='eval/loss_ce', num_train_epochs=1,
            gradient_accumulation_steps=1, eval_strategy='steps',
            greater_is_better=False, load_best_model_at_end=True,
            logging_steps=1, max_steps=100, eval_steps=25,
            save_total_limit=3, save_steps=100000,
        )
        self.model = torch.nn.Linear(1, 1)
        self.model.device = torch.device('cpu')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.trainer = DefaultTrainer(
            self.model, [], [], args, (self.optimizer, self.scheduler), None,
            SimpleNamespace(train=SimpleNamespace(output_dir=self.directory.name)),
        )
        self.metrics = {'eval/loss_ce': 6.8305265, 'eval/loss_total': 37.1194014}
        self.calls = []

        def evaluate(model, step, **kwargs):
            self.calls.append(step)
            model.eval()
            return dict(self.metrics)

        self.trainer.compute_eval_metrics = evaluate
        self.saved = []
        self.save_patch = patch('Training.trainer.save_checkpoint',
                                side_effect=lambda model, tokenizer, path:
                                self.saved.append((self.trainer.grad_step, path)))
        self.save_patch.start()
        self.addCleanup(self.save_patch.stop)

    def test_initial_model_survives_lower_total_but_higher_ce(self):
        self.trainer.train_step(self.model, epoch=0)
        self.assertTrue(self.model.training)
        self.assertEqual(self.trainer.best_val_metric_step, 0)
        self.assertEqual(self.trainer.best_val_metric, self.metrics['eval/loss_ce'])
        self.assertEqual(len(self.saved), 1)

        self.trainer.grad_step = 25
        self.metrics = {'eval/loss_ce': 9.5083953, 'eval/loss_total': 19.0398493}
        self.trainer.eval_step(self.model, step=25)
        self.assertEqual(self.trainer.best_val_metric_step, 0)
        self.assertEqual(len(self.saved), 1)

        self.trainer.grad_step = 50
        self.metrics = {'eval/loss_ce': 6.0, 'eval/loss_total': 25.0}
        self.trainer.eval_step(self.model, step=50)
        self.assertEqual(self.trainer.best_val_metric_step, 50)
        self.assertEqual(len(self.saved), 2)

    def test_periodic_snapshot_does_not_replace_best_reference(self):
        self.trainer.eval_step(self.model, step=0)
        best_path = self.trainer.best_val_checkpoint_path
        self.trainer.save_steps = 25
        self.trainer.grad_step = 25
        self.metrics['eval/loss_ce'] = 9.5
        self.trainer.eval_step(self.model, step=25)
        self.assertTrue(self.saved[-1][1].endswith('/default_25'))
        self.assertEqual(self.trainer.best_val_checkpoint_path, best_path)
        self.assertEqual(self.trainer.best_val_metric_step, 0)

    def test_initial_evaluation_runs_once(self):
        self.trainer.train_step(self.model, epoch=0)
        self.trainer.train_step(self.model, epoch=1)
        self.assertEqual(self.calls, [0])

    def test_epoch_end_evaluation_can_select_best(self):
        self.trainer.eval_step(self.model, step=0)
        self.trainer.grad_step = 7
        self.metrics['eval/loss_ce'] = 6.0
        self.trainer.eval_step(self.model, step=7)
        self.assertEqual(self.trainer.best_val_metric_step, 7)

    def test_nonfinite_metric_is_rejected(self):
        self.metrics['eval/loss_ce'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite checkpoint metric'):
            self.trainer.eval_step(self.model, step=0)
        self.assertEqual(self.saved, [])


if __name__ == '__main__':
    unittest.main()
