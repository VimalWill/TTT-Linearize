"""Passkey supervision and continuation checks, without downloads or a GPU."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from Training.dataloader import collect_passkey_rows, template_and_tokenize_passkey
from Training.train import continue_stage2, set_trainable_params
from Training.trainer import save_checkpoint
from eval import load_ttt_params, ruler_task_specs
from LinearTTT.model.LinearizeLlama.ttt_l2 import (
    block_causal_lact_swiglu_l2, read_lact_swiglu_l2,
)


class CharTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [ord(c) for c in text]


def sample(filler='x' * 100, answer='12345'):
    return {'prompt': f'Remember {answer}. {filler}\nThe pass key is {answer}',
            'answer': answer, 'num_tokens': 1}


class PasskeyDataTests(unittest.TestCase):
    def test_answer_only_loss_and_prompt_only_generation(self):
        tok, row = CharTokenizer(), sample()
        train = template_and_tokenize_passkey(row, tok)
        generation = template_and_tokenize_passkey(row, tok, include_label=False)
        n = len(generation['input_ids'])
        self.assertEqual(train['input_ids'][:n], generation['input_ids'])
        self.assertEqual(train['labels'][:n], [-100] * n)
        self.assertEqual(train['labels'][n:], tok.encode('12345', False) + [2])
        self.assertEqual(generation['labels'], train['labels'][n:])
        marker = tok.encode('The pass key is ', False)
        self.assertEqual(generation['input_ids'][-len(marker):], marker)
        all_tokens = template_and_tokenize_passkey(row, tok, mask_prompt=False)
        self.assertEqual(all_tokens['labels'], all_tokens['input_ids'])

    def test_filter_checks_actual_tokens_and_last_occurrence(self):
        row = sample()
        duplicate = dict(row)
        too_close = sample('x' * 100 + ' 12345')
        too_long = sample('x' * 250)
        # Reported num_tokens is deliberately wrong. No example is truncated.
        rows, skipped = collect_passkey_rows(
            [row, duplicate, too_close, too_long, sample('')], CharTokenizer(),
            count=10, max_tokens=200, min_tokens=80, min_distance=80)
        self.assertEqual(rows, [{'prompt': row['prompt'], 'answer': row['answer']}])
        self.assertEqual(skipped, dict(invalid=0, too_long=1, too_short=1,
                                       too_close=1, duplicate=1))

    def test_inconsistent_or_missing_answer_is_rejected(self):
        """Strict in the formatter, counted-and-skipped at the dataset boundary.

        A malformed streaming row must not terminate an otherwise usable
        corpus, but a bad row handed straight to the formatter is a caller bug
        and still raises.
        """
        row = sample()
        row['answer'] = '99999'
        with self.assertRaisesRegex(ValueError, 'suffix'):
            template_and_tokenize_passkey(row, CharTokenizer())

        good = sample()
        malformed = [
            {'prompt': 'No needle. The pass key is 12345', 'answer': '12345'},
            {'prompt': sample()['prompt']},                  # missing answer
            {'prompt': None, 'answer': '12345'},             # not a string
            {'prompt': 'no marker at all', 'answer': '12345'},
        ]
        rows, skipped = collect_passkey_rows(
            malformed + [good], CharTokenizer(), 10, 200, min_distance=10)
        self.assertEqual(rows, [{'prompt': good['prompt'], 'answer': good['answer']}])
        self.assertEqual(skipped['invalid'], len(malformed))

    def test_unique_answers_prevents_key_overlap_across_the_split(self):
        rows, skipped = collect_passkey_rows(
            [sample(), sample('y' * 100), sample(answer='98765')], CharTokenizer(),
            10, 200, unique_answers=True)
        self.assertEqual([row['answer'] for row in rows], ['12345', '98765'])
        self.assertEqual(skipped['duplicate'], 1)


class ContinuationTests(unittest.TestCase):
    def test_continued_adapter_and_ttt_reload_against_original_base(self):
        torch.manual_seed(0)
        base = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, use_cache=False))
        # A base-model TTT parameter tests overlay/persistence separately from LoRA.
        base.model.layers[0].self_attn.ttt_qk_scale = torch.nn.Parameter(torch.ones(1))
        config = OmegaConf.create({'model': {'attn_varient': 'ttt'}, 'train': {}})
        model = get_peft_model(copy.deepcopy(base), LoraConfig(
            task_type='CAUSAL_LM', r=2, target_modules=['q_proj']))
        set_trainable_params(model, config)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.fill_(0.2)
                elif 'ttt_qk_scale' in name:
                    param.fill_(3)
        ids = torch.tensor([[1, 4, 8, 12]])
        with tempfile.TemporaryDirectory() as directory:
            before, after = str(Path(directory) / 'before'), str(Path(directory) / 'after')
            save_checkpoint(model, Mock(), before)
            continued = continue_stage2(copy.deepcopy(base), before)
            set_trainable_params(continued, config)
            model.eval()
            continued.eval()
            torch.testing.assert_close(model(ids).logits, continued(ids).logits)
            self.assertEqual(continued.base_model.model.model.layers[0].self_attn.ttt_qk_scale.item(), 3)
            # A real outer optimizer step after loading the existing adapter.
            continued.train()
            optimizer = torch.optim.AdamW([p for p in continued.parameters() if p.requires_grad], lr=.01)
            continued(ids, labels=ids).loss.backward()
            optimizer.step()
            with torch.no_grad():
                continued.base_model.model.model.layers[0].self_attn.ttt_qk_scale.fill_(4)
            save_checkpoint(continued, Mock(), after)
            reloaded = copy.deepcopy(base)
            self.assertEqual(load_ttt_params(reloaded, after), 1)
            from peft import PeftModel
            reloaded = PeftModel.from_pretrained(reloaded, after).merge_and_unload()
            continued.eval()
            reloaded.eval()
            torch.testing.assert_close(continued(ids).logits, reloaded(ids).logits)
            self.assertEqual(reloaded.model.layers[0].self_attn.ttt_qk_scale.item(), 4)
            self.assertFalse(torch.allclose(model(ids).logits, continued(ids).logits))

    def test_missing_ttt_overlay_fails_before_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'saved TTT weights'):
                continue_stage2(torch.nn.Linear(1, 1), directory)

    def test_reader_overlay_preserves_small_fp32_updates(self):
        model = torch.nn.Module()
        model.ttt_reader_alignment = torch.nn.Linear(2, 2, bias=False)
        model.other = torch.nn.Linear(2, 2, bias=False)
        model.to(torch.bfloat16)
        saved = torch.eye(2) * 1.0007
        with tempfile.TemporaryDirectory() as directory:
            torch.save({'base_model.model.ttt_reader_alignment.weight': saved},
                       Path(directory) / 'ttt_params.pt')
            self.assertEqual(load_ttt_params(model, directory), 1)
        self.assertEqual(model.ttt_reader_alignment.weight.dtype, torch.float32)
        torch.testing.assert_close(model.ttt_reader_alignment.weight, saved, rtol=0, atol=0)
        self.assertEqual(model.other.weight.dtype, torch.bfloat16)


class RetrievalPlumbingTests(unittest.TestCase):
    def test_final_answer_gradient_reaches_early_memory_writes(self):
        # Exercise the actual eager math, bypassing only compile/autocast on CPU.
        op = inspect.unwrap(block_causal_lact_swiglu_l2)
        read = inspect.unwrap(read_lact_swiglu_l2)
        for shared_reader in (False, True):
            with self.subTest(shared_reader=shared_reader):
                torch.manual_seed(7)
                weights = [torch.randn(1, 4, 4) * .2 for _ in range(3)]
                q = torch.randn(1, 16, 4)
                k = torch.randn(1, 16, 4, requires_grad=True)
                v = torch.randn(1, 16, 4, requires_grad=True)
                lr = torch.full((1, 16, 1), .01, requires_grad=True)
                result = op(*weights, q, k, v, lr, lr, lr, chunk_size=4,
                            retention=torch.full((1, 16, 1), .99),
                            return_trajectory=shared_reader)
                output = read(result[-1], q, chunk_size=4) if shared_reader else result
                output[:, -1].square().sum().backward()
                for grad in (k.grad[:, :4], v.grad[:, :4], lr.grad[:, :4]):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                # The final chunk is read-only, so its K/V do not influence its output.
                self.assertEqual(v.grad[:, -4:].abs().sum().item(), 0)

    def test_2048_metric_is_registered_without_changing_other_tasks(self):
        common = ModuleType('lm_eval.tasks.ruler.common_utils')
        common.aggregate_metrics = lambda values: sum(values) / len(values)
        common.DEFAULT_SEQ_LENGTHS = [4096]
        ruler = ModuleType('lm_eval.tasks.ruler')
        ruler.common_utils = common
        with patch.dict('sys.modules', {'lm_eval.tasks.ruler': ruler}):
            specs = ruler_task_specs(['niah_single_1', 'swde'], [2048])
        self.assertEqual(specs[1], 'swde')
        metrics = {m['metric']: m for m in specs[0]['metric_list']}
        self.assertIn('2048', metrics)
        self.assertIn('4096', metrics)
        self.assertIs(metrics['2048']['aggregation'], common.aggregate_metrics)


if __name__ == '__main__':
    unittest.main()
