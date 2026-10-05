"""Long-context data, curriculum, loss, validation and reporting regressions."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from transformers import BatchEncoding

from Training.long_context import (BalancedContextSampler, SYNTHETIC_TASKS, SyntheticGenerator,
    encode_record, generate_answer, load_context_data, read_jsonl, retrieval_validation, tokenizer_fingerprint)
from Training.trainer import DefaultTrainer, FinetuneTrainer
from Training.utils import TokenLinearScheduler, get_optimizer_and_scheduler, model_autocast
from evaluate_long_context import RULER_TASKS, ruler_scores, run_matrix, selected_arms
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
                                                   initial_max_length=8192, allowed_lengths=[4096, 8192]),
                                      'model': dict(max_length=8192)})
            loaders = load_context_data(config, tokenizer)
            batch = next(iter(loaders['train']))
            self.assertEqual(len(batch['context_tasks']), 1)
            self.assertEqual(batch['input_ids'].shape[0], 1)
            self.assertEqual(batch['input_ids'].shape, batch['labels'].shape)
            # Feed the actual preparation -> dataset -> HF collator output into
            # training, so a plain-dict fixture cannot hide token-count failures.
            self.assertIsInstance(batch, BatchEncoding)
            model = torch.nn.Linear(1, 1)
            model.device = torch.device('cpu')
            expected_tokens = int(batch['attention_mask'].sum().item())
            config.train = dict(lr=.1, lr_scheduler='token_linear', warmup_tokens=expected_tokens,
                                max_input_tokens=expected_tokens * 2, output_dir=root)
            args = SimpleNamespace(metric_for_best_model='eval/retrieval_accuracy', num_train_epochs=1,
                gradient_accumulation_steps=1, eval_strategy='no', greater_is_better=True,
                load_best_model_at_end=False, logging_steps=1, max_steps=-1,
                eval_steps=100, save_total_limit=3, save_steps=100000, max_grad_norm=0)
            trainer = DefaultTrainer(model, [batch] * 2, [], args,
                                     get_optimizer_and_scheduler(model, config), tokenizer, config)
            trainer.initial_eval = False
            trainer.compute_loss = lambda model, data, return_outputs: (
                model(torch.ones(1, 1)).square().mean(), {})
            _, stopped = trainer.train_step(model, 0)
            self.assertTrue(stopped)
            self.assertEqual(trainer.input_tokens, expected_tokens * 2)
            self.assertEqual(trainer.grad_step, 2)
            manifest_path = train_dir / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['lengths'].append(16384)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'allowed_lengths'):
                load_context_data(config, tokenizer)
            manifest['lengths'].remove(16384)
            manifest_path.write_text(json.dumps(manifest))
            validation_path = train_dir / 'validation.jsonl'
            original = validation_path.read_text()
            malformed = dict(loaders['validation'].dataset.records[0], length_bucket=16384,
                             input_tokens=16384)
            validation_path.write_text(original + json.dumps(malformed) + '\n')
            with self.assertRaisesRegex(ValueError, 'allowed training/validation lengths'):
                load_context_data(config, tokenizer)
            validation_path.write_text(original)
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
    def test_best_restore_requires_complete_adapter_and_ttt_weights(self):
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
        from Training.train import set_trainable_params
        from Training.trainer import save_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            config = OmegaConf.create({'model': {'attn_varient': 'ttt'},
                'data': {'name': 'long_context'}, 'train': dict(lr=.01,
                max_input_tokens=1, lr_scheduler='token_linear', output_dir=directory)})
            base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, head_dim=8))
            base.model.layers[0].self_attn.ttt_qk_scale = torch.nn.Parameter(torch.ones(2))
            model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2, target_modules=['q_proj']))
            set_trainable_params(model, config)
            expected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            best = Path(directory, 'best_ckpt')
            save_checkpoint(model, Mock(), str(best))
            args = SimpleNamespace(metric_for_best_model='eval/retrieval_accuracy', num_train_epochs=1,
                gradient_accumulation_steps=1, eval_strategy='no', greater_is_better=True,
                load_best_model_at_end=True, logging_steps=1, max_steps=-1,
                eval_steps=100, save_total_limit=3, save_steps=100000, max_grad_norm=0)
            trainer = DefaultTrainer(model, [], [], args, get_optimizer_and_scheduler(model, config),
                                     Mock(), config)
            trainer.train_step = Mock(return_value=(model, True))
            trainer.input_tokens = trainer.grad_step = 1
            trainer.eval_metrics_by_step['eval_step'] = [1]
            trainer.best_val_checkpoint_path = str(best)
            trainer.save_latest_checkpoint = Mock()
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.add_(.25)
            trainer.train()
            for name, param in model.named_parameters():
                if name in expected:
                    torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
            ttt = best / 'ttt_params.pt'
            saved_ttt = ttt.read_bytes()
            torch.save({}, ttt)
            with self.assertRaisesRegex(RuntimeError, 'Could not restore'):
                trainer.train()
            ttt.unlink()
            with self.assertRaisesRegex(RuntimeError, 'Could not restore'):
                trainer.train()
            ttt.write_bytes(saved_ttt)
            (best / 'adapter_model.safetensors').unlink()
            with self.assertRaisesRegex(RuntimeError, 'Could not restore'):
                trainer.train()

    def test_mixed_precision_ttt_updates_and_checkpoint_continuation(self):
        from peft import LoraConfig, get_peft_model
        from LinearTTT.model.LinearizeLlama.Configuration import LigerGLAConfig
        from LinearTTT.model.LinearizeLlama.LinearizeLlama import LigerGLAForCausalLM
        from Training.train import set_trainable_params, continue_stage2
        from Training.trainer import save_checkpoint
        for shared in (False, True):
            with self.subTest(shared=shared), torch._dynamo.config.patch(disable=True), \
                    tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(3)
                cfg = LigerGLAConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                    lact_chunk_size=4, window_size=4, ttt_inner_loss='l2', ttt_gate='sigmoid',
                    ttt_scale_init_bias=-3., ttt_share_groups=[[0, 1]] if shared else [],
                    ttt_reader_alignment='linear' if shared else 'none', use_cache=False)
                config = OmegaConf.create({'model': {'attn_varient': 'ttt'},
                                           'data': {'name': 'long_context'}, 'train': {}})
                base = LigerGLAForCausalLM(cfg).to(torch.bfloat16)
                set_trainable_params(base, config)
                original = copy.deepcopy(base)
                model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2,
                                                       target_modules=['q_proj', 'k_proj', 'v_proj']))
                set_trainable_params(model, config)
                self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad))
                self.assertEqual(model.get_input_embeddings().weight.dtype, torch.bfloat16)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
                trainer = object.__new__(FinetuneTrainer)
                trainer.config, trainer.criterion = config, torch.nn.CrossEntropyLoss()
                ids = torch.randint(3, 32, (1, 13))
                labels = ids.clone()
                labels[:, :10] = -100
                inputs = BatchEncoding(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
                gate = model.base_model.model.model.layers[0].self_attn.ttt_scale_proj.bias
                before = gate.detach().clone()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                loss = trainer.compute_loss(model, inputs)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                self.assertFalse(torch.equal(gate, before))
                save_checkpoint(model, Mock(), directory)
                restored = continue_stage2(original, directory)
                set_trainable_params(restored, config)
                model.eval()
                restored.eval()
                with torch.no_grad(), model_autocast(model):
                    torch.testing.assert_close(model(ids, use_cache=False).logits,
                                               restored(ids, use_cache=False).logits, rtol=0, atol=0)
                tokenizer = SimpleNamespace(encode=lambda *a, **kw: ids[0].tolist(),
                    decode=lambda *a, **kw: 'generated', pad_token_id=0, eos_token_id=2)
                self.assertEqual(generate_answer(restored, tokenizer, {'prompt': 'test'}, 3), 'generated')
                path = Path(directory) / 'ttt_params.pt'
                saved = torch.load(path, weights_only=True)
                del saved[next(k for k in saved if 'ttt_scale_proj.bias' in k)]
                torch.save(saved, path)
                with self.assertRaisesRegex(ValueError, 'Incomplete TTT checkpoint'):
                    continue_stage2(original, directory)

    def test_reduced_recipe_cannot_enter_16k_phase_and_scales_warmup(self):
        root = Path(__file__).resolve().parents[1]
        cfg = OmegaConf.load(root / 'Configs/ttt_ar_llama_long_context_anchor_f_reduced.yml')
        self.assertIsNone(cfg.model.ttt_layer_indices)
        self.assertEqual(list(cfg.model.ttt_share_groups), [])
        self.assertEqual(cfg.model.ttt_reader_alignment, 'none')
        self.assertEqual(list(cfg.data.allowed_lengths), [4096, 8192])
        self.assertEqual(cfg.model.max_length, 8192)
        self.assertEqual(cfg.data.initial_max_length, 8192)
        self.assertEqual(cfg.train.final_max_length, 8192)
        self.assertEqual(cfg.train.phase_transition_tokens, 0)
        self.assertEqual(cfg.train.max_input_tokens, 40000000)
        self.assertEqual(cfg.train.warmup_tokens / cfg.train.max_input_tokens, .03)
        optimizer, scheduler = get_optimizer_and_scheduler(torch.nn.Linear(1, 1), cfg)
        scheduler.step_tokens(cfg.train.warmup_tokens)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4)
        scheduler.step_tokens(cfg.train.max_input_tokens)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-5)
        full = OmegaConf.load(root / 'Configs/ttt_ar_llama_long_context_anchor_f.yml')
        self.assertEqual(full.train.max_input_tokens, 100000000)
        self.assertEqual(full.train.final_max_length, 16384)

    def test_training_stops_on_tokens_validates_final_update_and_rejects_underrun(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = OmegaConf.create({'train': dict(lr=.1, lr_scheduler='token_linear',
                max_input_tokens=20, warmup_tokens=6, min_lr=.01, output_dir=directory)})
            torch.manual_seed(9)
            model = torch.nn.Linear(1, 1)
            model.device = torch.device('cpu')
            initial_weight = model.weight.detach().clone()
            # The real Hugging Face collator returns BatchEncoding, not dict.
            batch = BatchEncoding(dict(input_ids=torch.ones(1, 3, dtype=torch.long),
                         attention_mask=torch.ones(1, 3), context_tasks=['single_number'],
                         context_lengths=[4096]))
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
            def evaluate(model, step):
                # A timeout/error in validation must not leave only step-0 weights.
                progress = json.loads(Path(directory, 'last_ckpt/training_progress.json').read_text())
                self.assertEqual(progress['optimizer_steps'], step)
                self.assertEqual(progress['input_tokens'], 24)
                return {'eval/retrieval_accuracy': .25}
            trainer.compute_eval_metrics = Mock(side_effect=evaluate)
            self.assertEqual(trainer.scheduler.get_last_lr(), [0])
            with patch('Training.trainer.save_checkpoint') as save:
                trainer.train()
            self.assertEqual(trainer.input_tokens, 24)  # Only one accumulation window may overshoot.
            self.assertEqual(trainer.grad_step, 4)
            self.assertEqual(trainer.step, 8)
            self.assertEqual(trainer.training_exposure['single_number/4096'], dict(examples=8, tokens=24))
            self.assertGreater(trainer.optimizer.param_groups[0]['lr'], 0)
            self.assertFalse(torch.equal(model.weight.detach(), initial_weight))
            self.assertEqual(trainer.eval_metrics_by_step['eval_step'], [4])
            self.assertTrue(save.call_args_list[0].args[2].endswith('/last_ckpt'))
            self.assertEqual(sum(c.args[2].endswith('/last_ckpt') for c in save.call_args_list), 1)
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
            trainer.compute_loss = Mock()
            for bad_batch in (BatchEncoding({}), BatchEncoding(dict(
                    input_ids=torch.ones(1, 3, dtype=torch.long), attention_mask=torch.zeros(1, 3)))):
                trainer.train_loader = [bad_batch]
                trainer.initial_eval = False
                with self.assertRaisesRegex(ValueError, 'positive count'):
                    trainer.train_step(model, 0)
            trainer.compute_loss.assert_not_called()
            with patch.object(trainer.optimizer, 'step') as optimizer_step:
                with self.assertRaisesRegex(RuntimeError, 'no counted input tokens'):
                    trainer._optimizer_step(2, 2)
            optimizer_step.assert_not_called()
            trainer.train_loader = SimpleNamespace(sampler=sampler)
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
            self.assertTrue(save.call_args_list[0].args[2].endswith('/last_ckpt'))
            self.assertTrue(save.call_args_list[1].args[2].endswith('/best_ckpt'))

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

    def test_validation_diagnoses_format_errors_without_changing_exact_match(self):
        rows = [dict(id=str(i), prompt='Retrieve the value:', answer='12345', answers=['12345'],
                     task='single_number', length_bucket=4096) for i in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'validation_retrieval_samples.jsonl'
            with patch('Training.long_context.generate_answer',
                       side_effect=['The value is 12345.', '12346']):
                metrics = retrieval_validation(None, None, rows, samples_path=path,
                                               step=100, input_tokens=4000000)
            samples = read_jsonl(path)
        self.assertEqual(metrics['eval/retrieval_accuracy'], 0)
        self.assertEqual(metrics['eval/retrieval_substring_recall'], .5)
        self.assertEqual([r['prediction'] for r in samples], ['The value is 12345.', '12346'])
        self.assertTrue(all(r['step'] == 100 and r['input_tokens'] == 4000000 for r in samples))


class LongContextReportingTests(unittest.TestCase):
    def test_reduced_matrix_needs_no_anchor_i_and_keeps_paired_unseen_lengths(self):
        with tempfile.TemporaryDirectory() as root:
            args = SimpleNamespace(out_dir=str(Path(root) / 'results'), data_dir=root,
                anchor_f_cfg='f_reduced.yml', anchor_i_cfg='i.yml', anchor_cfg='deploy.yml',
                anchor_f_ckpt='f', anchor_i_ckpt=None, anchor_f_adapter='continued', anchor_i_adapter=None,
                benchmarks=['niah'], lengths=[16384, 32768], no_ablation=False, niah_data='niah.jsonl',
                base='base', seed=1, max_new_tokens=256, ruler_limit=500,
                arms=['baseline', 'anchor_f'])
            commands = []
            def worker(command, **kwargs):
                commands.append(command)
                Path(command[command.index('--out') + 1]).write_text(json.dumps(dict(status='ok')))
                return SimpleNamespace(returncode=0)
            with patch('evaluate_long_context.subprocess.run', side_effect=worker), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_matrix(args), 0)
            report = json.loads((Path(args.out_dir) / 'summary.json').read_text())
            self.assertEqual(len(commands), 6)
            for length in args.lengths:
                cases = [c for c in report['cases'] if c['length'] == length]
                self.assertEqual([(c['arm'], c['ablate']) for c in cases],
                                 [('baseline', None), ('anchor_f', None), ('anchor_f', 'ttt')])
                paired = [c['command'] for c in cases if c['arm'] == 'anchor_f']
                self.assertEqual(paired[0][paired[0].index('--niah-data') + 1],
                                 paired[1][paired[1].index('--niah-data') + 1])
            args.arms = ['anchor_i']
            with self.assertRaisesRegex(ValueError, 'source checkpoint'):
                selected_arms(args)

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
        for invalid in (None, '0.5', float('nan')):
            results[f'{RULER_TASKS[0]}/16384,none'] = invalid
            with self.assertRaises(ValueError):
                ruler_scores(dict(results=results), 16384)
        del results[f'{RULER_TASKS[0]}/16384,none']
        results[RULER_TASKS[0]] = .5
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


class ShortSequenceWindowTests(unittest.TestCase):
    """Short sequences must bypass flex_attention without changing the result.

    Stage 1 died at step 425 with `CUDA error: an illegal memory access` on
    seq_len 78: each distinct length builds its own BlockMask, enough of them
    exhaust Dynamo's recompile budget, and the eager fallback mishandles a
    BlockMask on a sequence shorter than its 128-token block. Long-context
    training mixes lengths by design, so the guard has to hold.
    """

    def test_guard_matches_the_reference_window_exactly(self):
        import torch
        from LinearTTT.model.LinearizeLlama.LinearizeLlama import (
            sliding_window_attention, use_sdpa_sliding_window)
        torch.manual_seed(0)
        window = 512
        try:
            for n in (1, 78, 511, 512, 513, 1024):
                q, k, v = (torch.randn(1, 4, n, 32) for _ in range(3))
                use_sdpa_sliding_window(True)
                reference = sliding_window_attention(q, k, v, window_size=window, causal=True)
                use_sdpa_sliding_window(False)
                guarded = sliding_window_attention(q, k, v, window_size=window, causal=True)
                self.assertTrue(torch.allclose(reference, guarded, atol=1e-5),
                                f'window output changed at seq_len {n}')
        finally:
            use_sdpa_sliding_window(False)
