"""Long-context data, curriculum, loss, validation and reporting regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf

from Training.long_context import (BalancedContextSampler, SYNTHETIC_TASKS, SyntheticGenerator,
    encode_record, generate_answer, load_context_data, read_jsonl, retrieval_validation, tokenizer_fingerprint)
from Training.trainer import DefaultTrainer, FinetuneTrainer
from Training.utils import TokenLinearScheduler, get_optimizer_and_scheduler
from evaluate_long_context import RULER_TASKS, ruler_scores, run_matrix
from plot_long_context import summarize_niah, wilson_interval
from prepare_long_context import build_niah, build_training, partition_sources


class CharTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 2
    name_or_path = 'test-character-tokenizer'

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return ''.join(chr(int(t)) for t in tokens if int(t) > 2)

    def get_vocab(self):
        return {chr(i): i for i in range(256)}


class LongContextDataTests(unittest.TestCase):
    def test_fast_tokenizer_preparation_loader_and_heldout_grid(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
        from transformers import PreTrainedTokenizerFast
        texts = [f'Document {i}: ordinary filler about different subjects. ' * 60 for i in range(40)]
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[UNK]', '[BOS]', '[EOS]']))
        backend.post_processor = processors.TemplateProcessing(single='[BOS] $A',
            special_tokens=[('[BOS]', backend.token_to_id('[BOS]'))])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='[BOS]',
                                            eos_token='[EOS]', pad_token='[EOS]')
        sources = [dict(text=t, prompt='Summarize: ' + t, answer='ordinary subjects') for t in texts]
        with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()):
            train_dir, niah_dir = Path(root) / 'train', Path(root) / 'niah'
            train_dir.mkdir()
            niah_dir.mkdir()
            build_training(train_dir, tokenizer, partition_sources(sources, 4), [4096, 8192],
                           dict(train=1, validation=1, test=1), 5)
            config = OmegaConf.create({'data': dict(path=str(train_dir), micro_batch_size=1,
                                                   initial_max_length=8192),
                                      'model': dict(max_length=8192)})
            loaders = load_context_data(config, tokenizer)
            batch = next(iter(loaders['train']))
            self.assertEqual(len(batch['context_tasks']), 1)
            self.assertEqual(batch['input_ids'].shape[0], 1)
            self.assertEqual(batch['input_ids'].shape, batch['labels'].shape)
            build_niah(niah_dir, tokenizer, train_dir, [4096], [0, 50, 100], 1, 10)
            rows = read_jsonl(niah_dir / 'niah.jsonl')
            forbidden = {r['value'] for r in read_jsonl(train_dir / 'forbidden_values.jsonl')}
            test_docs = {r['id'] for r in read_jsonl(train_dir / 'documents_test.jsonl')}
            self.assertEqual(len(rows), 6)
            for row in rows:
                self.assertTrue(4080 <= row['prompt_tokens'] <= 4096)
                self.assertEqual(row['prompt_tokens'], len(tokenizer.encode(row['prompt'])))
                self.assertFalse(set(row['answers']) & forbidden)
                self.assertTrue(set(row['document_ids']) <= test_docs)

    def test_all_synthetic_tasks_are_intact_answer_only_and_distant(self):
        tokenizer = CharTokenizer()
        documents = [dict(id='d1', text='This is ordinary document filler. ' * 100)]
        generator = SyntheticGenerator(tokenizer, documents, 1)
        for task in SYNTHETIC_TASKS:
            with self.subTest(task=task):
                row = generator.example(task, 4096, 'train', task, far_fraction=1)
                encoded = encode_record(row, tokenizer, 4096)
                self.assertLessEqual(len(encoded['input_ids']), 4096)
                self.assertGreaterEqual(len(encoded['input_ids']), 4080)
                self.assertGreaterEqual(min(row['needle_distances']), 768)
                prefix_length = row['prompt_tokens']
                self.assertEqual(encoded['labels'][:prefix_length], [-100] * prefix_length)
                self.assertEqual(encoded['labels'][prefix_length:],
                                 tokenizer.encode(row['answer'], False) + [2])
                self.assertTrue(all(a in row['prompt'] for a in row['answers']))
                with self.assertRaises(ValueError):
                    encode_record(row, tokenizer, 2048)

    def test_sources_and_answer_identifiers_are_disjoint_across_splits(self):
        sources = [dict(instruction=f'Question {i}: ' + 'ordinary filler ' * 50,
                        input='', output='a short source answer') for i in range(40)]
        sources.append(sources[0])
        partition = partition_sources(sources, 4)
        self.assertEqual(sum(map(len, partition.values())), 40)
        ids = {k: {d['id'] for d in v} for k, v in partition.items()}
        self.assertFalse(ids['train'] & ids['validation'] or ids['train'] & ids['test'])
        self.assertFalse(ids['validation'] & ids['test'])
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            manifest = build_training(directory, CharTokenizer(), partition, [4096, 8192],
                                      dict(train=1, validation=1, test=1), 5)
            rows = {s: read_jsonl(Path(directory) / f'{s}.jsonl') for s in partition}
            self.assertEqual(manifest['tokenizer_fingerprint'], tokenizer_fingerprint(CharTokenizer()))
        identifiers = {}
        for split, examples in rows.items():
            self.assertTrue(all(set(r['document_ids']) <= ids[split] for r in examples))
            identifiers[split] = {v for r in examples if r['task'] != 'instruction' for v in r['answers']}
        self.assertFalse(identifiers['train'] & identifiers['validation'])
        self.assertFalse(identifiers['train'] & identifiers['test'])

    def test_sampler_changes_phase_mid_iterator_and_balances_task_weights(self):
        rows = [dict(task=task, length_bucket=length) for task in ('a', 'b')
                for length in (4096, 8192, 16384) for _ in range(100)]
        sampler = BalancedContextSampler(rows, 8192, 1, dict(a=.75, b=.25))
        iterator = iter(sampler)
        first = [rows[next(iterator)] for _ in range(200)]
        self.assertTrue(all(r['length_bucket'] <= 8192 for r in first))
        self.assertTrue(120 <= sum(r['task'] == 'a' for r in first) <= 180)
        sampler.set_max_length(16384)
        later = [rows[i] for i in iterator]
        self.assertTrue(any(r['length_bucket'] == 16384 for r in later))


class LongContextTrainingTests(unittest.TestCase):
    def test_training_stops_on_tokens_validates_final_update_and_rejects_underrun(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = OmegaConf.create({'train': dict(lr=.1, lr_scheduler='token_linear',
                max_input_tokens=20, min_lr=.01, output_dir=directory)})
            model = torch.nn.Linear(1, 1)
            model.device = torch.device('cpu')
            batch = dict(input_ids=torch.ones(1, 3, dtype=torch.long),
                         attention_mask=torch.ones(1, 3), context_tasks=['single_number'],
                         context_lengths=[4096])
            args = SimpleNamespace(metric_for_best_model='eval/retrieval_accuracy', num_train_epochs=1,
                gradient_accumulation_steps=2, eval_strategy='no', greater_is_better=True,
                load_best_model_at_end=False, logging_steps=1, max_steps=-1,
                eval_steps=100, save_total_limit=3, save_steps=100000, max_grad_norm=0)
            trainer = DefaultTrainer(model, [batch] * 20, [], args,
                                     get_optimizer_and_scheduler(model, config), None, config)
            trainer.initial_eval = False
            def compute_loss(model, inputs, return_outputs):
                loss = model(torch.ones(1, 1)).square().mean()
                return loss, {'loss_ce': loss.item()}
            trainer.compute_loss = compute_loss
            trainer.compute_eval_metrics = Mock(return_value={'eval/retrieval_accuracy': .25})
            with patch('Training.trainer.save_checkpoint') as save:
                trainer.train()
            self.assertEqual(trainer.input_tokens, 24)  # Only one accumulation window may overshoot.
            self.assertEqual(trainer.grad_step, 4)
            self.assertEqual(trainer.step, 8)
            self.assertEqual(trainer.training_exposure['single_number/4096'], dict(examples=8, tokens=24))
            self.assertEqual(trainer.eval_metrics_by_step['eval_step'], [4])
            self.assertTrue(save.call_args.args[2].endswith('/last_ckpt'))
            trainer.input_tokens = 0
            trainer.train_step = Mock(return_value=(model, True))
            with self.assertRaisesRegex(RuntimeError, 'before token budget'):
                trainer.train()

    def test_token_schedule_and_successful_update_accounting(self):
        model = torch.nn.Linear(1, 1)
        model.device = torch.device('cpu')
        with tempfile.TemporaryDirectory() as directory:
            config = OmegaConf.create({'train': dict(lr=.1, lr_scheduler='token_linear',
                max_input_tokens=100, warmup_tokens=20, min_lr=.01,
                phase_transition_tokens=40, final_max_length=16384, output_dir=directory)})
            args = SimpleNamespace(metric_for_best_model='eval/retrieval_accuracy', num_train_epochs=1,
                gradient_accumulation_steps=2, eval_strategy='no', greater_is_better=True,
                load_best_model_at_end=False, logging_steps=1, max_steps=-1,
                eval_steps=100, save_total_limit=3, save_steps=100000, max_grad_norm=0)
            sampler = Mock(max_length=8192)
            trainer = DefaultTrainer(model, SimpleNamespace(sampler=sampler), [], args,
                                      get_optimizer_and_scheduler(model, config), None, config)
            self.assertEqual(trainer.scheduler.get_last_lr(), [0])
            trainer.pending_input_tokens = 20
            model.weight.grad = torch.ones_like(model.weight)
            self.assertTrue(trainer._optimizer_step(2, 2))
            self.assertEqual(trainer.input_tokens, 20)
            self.assertAlmostEqual(trainer.scheduler.get_last_lr()[0], .1)
            trainer.pending_input_tokens = 20
            model.weight.grad = torch.full_like(model.weight, float('nan'))
            self.assertFalse(trainer._optimizer_step(2, 2))
            self.assertEqual(trainer.input_tokens, 20)
            self.assertEqual(trainer.pending_input_tokens, 0)
            trainer.pending_input_tokens = 40
            model.weight.grad = torch.ones_like(model.weight)
            self.assertTrue(trainer._optimizer_step(2, 2))
            self.assertEqual(trainer.input_tokens, 60)
            self.assertAlmostEqual(trainer.scheduler.get_last_lr()[0], .055)
            sampler.set_max_length.assert_called_once_with(16384)
            trainer.scheduler.step_tokens(100)
            self.assertAlmostEqual(trainer.scheduler.get_last_lr()[0], .01)
            trainer.compute_eval_metrics = Mock(return_value={'eval/retrieval_accuracy': 0})
            with patch('Training.trainer.save_checkpoint') as save:
                trainer.eval_step(model, step=0)
            save.assert_called_once()  # Initial zero accuracy must still produce a checkpoint.

    def test_suffix_only_loss_equals_full_logits_loss_and_gradients(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(5)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=64)).eval()
        inputs = dict(input_ids=torch.randint(4, 250, (1, 20)), attention_mask=torch.ones(1, 20))
        inputs['labels'] = inputs['input_ids'].clone()
        inputs['labels'][:, :16] = -100
        trainer = object.__new__(FinetuneTrainer)
        trainer.config = OmegaConf.create({'data': {'name': 'long_context'}})
        trainer.criterion = torch.nn.CrossEntropyLoss()
        actual = trainer.compute_loss(model, inputs)
        actual.backward()
        gradient = model.model.embed_tokens.weight.grad.clone()
        model.zero_grad()
        expected = model(**inputs, use_cache=False).loss
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(gradient, model.model.embed_tokens.weight.grad)

    def test_generated_validation_uses_prompt_only_cache_and_macro_cells(self):
        tokenizer = CharTokenizer()
        class Model:
            device = torch.device('cpu')
            def forward(self, logits_to_keep=0):
                pass
            def generate(self, input_ids, **kwargs):
                self.last_kwargs, self.prompt = kwargs, input_ids
                answer = torch.tensor([tokenizer.encode('12345', False)])
                return torch.cat([input_ids, answer], dim=1)
        model = Model()
        rows = [dict(prompt='Remember 12345. Return the value: ', answer='12345',
                     task='single_number', length_bucket=4096)] * 5
        rows += [dict(prompt='Remember 99999. Return the value: ', answer='99999',
                      task='single_number', length_bucket=8192)]
        scores = retrieval_validation(model, tokenizer, rows, per_cell=4)
        self.assertEqual(scores['eval/retrieval_accuracy'], .5)
        self.assertTrue(model.last_kwargs['use_cache'])
        self.assertEqual(model.last_kwargs['logits_to_keep'], 1)
        self.assertTrue(tokenizer.decode(model.prompt[0]).endswith('value: '))


class LongContextReportingTests(unittest.TestCase):
    def test_matrix_continues_after_oom_and_error_and_forwards_generation_limit(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(out_dir=str(Path(root) / 'results'), data_dir=root,
                anchor_f_cfg='f.yml', anchor_i_cfg='i.yml', anchor_cfg='deploy.yml',
                anchor_f_ckpt='f', anchor_i_ckpt='i', anchor_f_adapter=None, anchor_i_adapter=None,
                benchmarks=['niah', 'ruler'], lengths=[4096], no_ablation=True, niah_data='niah.jsonl',
                base='base', seed=1, max_new_tokens=17, ruler_limit=2)
            commands = []
            def worker(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index('--out') + 1])
                if '--worker' in command:
                    self.assertEqual(command[command.index('--max-new-tokens') + 1], '17')
                    failed = '--baseline' in command
                    output.write_text(json.dumps(dict(status='oom' if failed else 'ok')))
                    return SimpleNamespace(returncode=2 if failed else 0)
                if '--baseline' in command:
                    return SimpleNamespace(returncode=1)  # No JSON: later arms must still run.
                output.with_suffix('.json').write_text(json.dumps(dict(results={
                    f'{task}/4096,none': .5 for task in RULER_TASKS})))
                return SimpleNamespace(returncode=0)
            with patch('evaluate_long_context.subprocess.run', side_effect=worker), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_matrix(args), 1)
            report = json.loads((Path(args.out_dir) / 'summary.json').read_text())
            self.assertEqual(len(commands), 8)
            self.assertEqual([c['status'] for c in report['cases']],
                             ['oom', 'ok', 'ok', 'ok', 'error', 'ok', 'ok', 'ok'])

    def test_full_ruler_requires_every_task_and_discards_unrequested_sentinels(self):
        results = {f'{task}/16384,none': .5 for task in RULER_TASKS}
        results.update({f'{task}/4096,none': -1 for task in RULER_TASKS})
        self.assertEqual(len(ruler_scores(dict(results=results), 16384)), 13)
        results[f'{RULER_TASKS[0]}/16384,none'] = -1
        with self.assertRaises(ValueError):
            ruler_scores(dict(results=results), 16384)

    def test_paired_gains_and_failed_lengths_remain_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = []
            for ablate, outcomes in [(None, [1, 1]), ('ttt', [0, 1])]:
                path = Path(directory) / ('on.json' if ablate is None else 'off.json')
                rows = [dict(id=str(i), task='single_number', length_bucket=4096,
                             depth_percent=50, answer='12345', prompt_tokens=4096,
                             exact_match=outcome) for i, outcome in enumerate(outcomes)]
                path.with_suffix('.samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                cases.append(dict(benchmark='niah', status='ok', arm='anchor_i',
                                  ablate=ablate, result_path=str(path)))
            cases.append(dict(benchmark='niah', status='oom', arm='anchor_i',
                              ablate=None, result_path='missing.json'))
            cells, gains = summarize_niah(dict(cases=cases))
        self.assertEqual(len(cells), 2)
        self.assertEqual(gains[0]['accuracy_gain'], .5)
        self.assertEqual(gains[0]['count'], 2)
        low, high = wilson_interval(1, 2)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)


if __name__ == '__main__':
    unittest.main()
