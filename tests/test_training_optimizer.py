"""Optimizer precision, accumulation, and scheduling regressions (CPU)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from LinearTTT.model.LinearizeLlama.LinearizeLlama import ReaderOutputAlignment as LlamaReader
from LinearTTT.model.LinearizeMistral.LinearizeMistral import ReaderOutputAlignment as MistralReader
from Training.train import set_trainable_params
from Training.trainer import DefaultTrainer
from Training.utils import get_optimizer_and_scheduler


class ReaderPrecisionTests(unittest.TestCase):
    def test_joint_training_accumulates_small_reader_updates(self):
        for reader_type in (LlamaReader, MistralReader):
            with self.subTest(backbone=reader_type.__module__):
                model = torch.nn.Module()
                model.ttt_reader_alignment = reader_type(1, 4)
                model.q_proj = torch.nn.Linear(4, 4)
                model.to(torch.bfloat16)
                config = OmegaConf.create({
                    'model': {'attn_varient': 'ttt'},
                    'train': {'lr': 0.0002},
                })
                set_trainable_params(model, config)
                reader = model.ttt_reader_alignment
                self.assertEqual(reader.weight.dtype, torch.float32)
                self.assertFalse(model.q_proj.weight.requires_grad)
                self.assertEqual(model.q_proj.weight.dtype, torch.bfloat16)
                optimizer, _ = get_optimizer_and_scheduler(model, config)
                x = torch.eye(4, dtype=torch.bfloat16).unsqueeze(0)
                for _ in range(32):
                    optimizer.zero_grad()
                    y = reader(x)
                    self.assertEqual(y.dtype, torch.bfloat16)
                    y.float().square().sum().backward()
                    optimizer.step()
                diagonal = reader.weight.detach().diagonal(dim1=-2, dim2=-1)
                self.assertTrue(torch.all(diagonal < 0.995))
                self.assertEqual(optimizer.state[reader.weight]['exp_avg'].dtype,
                                 torch.float32)


class TrainerOptimizerTests(unittest.TestCase):
    def make_trainer(self, schedule='linear', accumulation=1):
        directory = tempfile.TemporaryDirectory(prefix='ttt-optimizer-test-')
        self.addCleanup(directory.cleanup)
        config = OmegaConf.create({'train': {
            'lr': 0.0002, 'max_steps': 4, 'lr_scheduler': schedule,
            'output_dir': directory.name,
        }})
        model = torch.nn.Linear(1, 1, bias=False)
        model.device = torch.device('cpu')
        args = SimpleNamespace(
            metric_for_best_model='eval/loss_ce', num_train_epochs=1,
            gradient_accumulation_steps=accumulation, eval_strategy='no',
            greater_is_better=False, load_best_model_at_end=False,
            logging_steps=1, max_steps=4, eval_steps=25,
            save_total_limit=3, save_steps=100000, max_grad_norm=0,
        )
        trainer = DefaultTrainer(
            model, [], [], args, get_optimizer_and_scheduler(model, config),
            None, config,
        )
        trainer.initial_eval = False
        return trainer

    def test_linear_schedule_counts_updates_not_evaluations_or_skips(self):
        trainer = self.make_trainer()
        trainer.compute_eval_metrics = Mock(return_value={'eval/loss_ce': 7.0})
        with patch('Training.trainer.save_checkpoint'):
            trainer.eval_step(trainer.model, step=0)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 0.0002)
        for step in range(1, 5):
            trainer.model.weight.grad = torch.ones_like(trainer.model.weight)
            self.assertTrue(trainer._optimizer_step(1, 1))
            expected = 0.0002 * (1 - step / 4)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], expected)
            trainer.eval_step(trainer.model, step=step)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], expected)
            trainer.model.weight.grad = torch.full_like(trainer.model.weight, float('nan'))
            self.assertFalse(trainer._optimizer_step(1, 1))
            self.assertEqual(trainer.grad_step, step)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], expected)

    def test_raw_loss_reporting_preserves_accumulated_gradient(self):
        trainer = self.make_trainer(accumulation=2)
        trainer.model.weight.data.fill_(1.0)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = None
        trainer.train_loader = [None] * 3
        logged = []
        trainer.wandb = SimpleNamespace(log=lambda values, step: logged.append(dict(values)))

        def compute_loss(model, inputs, return_outputs):
            loss = model(torch.ones(1, 1)).square().mean()
            return loss, {'loss_ce': loss.item()}

        trainer.compute_loss = compute_loss
        trainer.train_step(trainer.model, epoch=0)
        # Two microbatches average to one update, and the final one forms a
        # correctly normalized partial window: 1 -> .8 -> .64.
        self.assertAlmostEqual(trainer.model.weight.item(), 0.64, places=6)
        self.assertEqual(trainer.grad_step, 2)
        self.assertEqual(len(logged), 2)
        self.assertAlmostEqual(logged[0]['train/loss'], 1.0)
        self.assertAlmostEqual(logged[1]['train/loss'], 0.64, places=6)
        self.assertAlmostEqual(logged[1]['train/loss_mean'], 0.88, places=6)

    def test_legacy_plateau_remains_metric_driven(self):
        trainer = self.make_trainer(schedule='plateau')
        self.assertTrue(trainer.scheduler_step_after_epoch)
        trainer.model.weight.grad = torch.ones_like(trainer.model.weight)
        trainer._optimizer_step(1, 1)
        self.assertEqual(trainer.scheduler.last_epoch, 0)

    def test_linear_schedule_requires_a_budget(self):
        config = OmegaConf.create({'train': {'lr': 0.0002, 'lr_scheduler': 'linear'}})
        with self.assertRaisesRegex(ValueError, 'positive max_steps'):
            get_optimizer_and_scheduler(torch.nn.Linear(1, 1), config)

    def test_mistral_stages_and_arms_share_validation_boundary(self):
        root = Path(__file__).resolve().parents[1] / 'Configs'
        configs = [OmegaConf.load(root / f'ttt_{stage}_mistral_{arm}.yml')
                   for stage in ('at', 'ar') for arm in ('anchor', 'l2')]
        self.assertEqual({c.data.num_val_docs for c in configs}, {100})


if __name__ == '__main__':
    unittest.main()
