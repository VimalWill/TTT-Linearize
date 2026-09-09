"""Small real-model regression tests; no checkpoint downloads or FLA required.

Use the repository's pinned transformers==4.45.0 and PyTorch:
TORCHDYNAMO_DISABLE=1 python -m unittest discover -s tests -v
Set TTT_TEST_DEVICE=cuda to exercise compiled BF16 CUDA operators instead.
"""
import copy
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from LinearTTT import LigerGLAConfig, LigerGLAForCausalLM
from LinearTTT.model.LinearizeLlama.ttt_l2 import (
    block_causal_lact_swiglu_l2 as write_memory,
    read_lact_swiglu_l2 as read_memory,
)
from test_causality import assert_causal


DEVICE = os.environ.get('TTT_TEST_DEVICE', 'cpu')


def tiny_model(groups=None, momentum=True, muon=False):
    torch.manual_seed(42)
    cfg = LigerGLAConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, lact_chunk_size=4, window_size=4,
        ttt_inner_loss='l2', ttt_share_groups=groups,
        ttt_use_momentum=momentum, ttt_use_muon=muon,
        ttt_base_lr=0.05, ttt_scale_init_bias=1.0,
        bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=False,
    )
    dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32
    return LigerGLAForCausalLM(cfg).to(device=DEVICE, dtype=dtype)


class SharedMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.ids = torch.randint(1, 32, (2, 15), device=DEVICE)

    def test_full_model_causality(self):
        for groups in (None, [[0, 1, 2]], [[0, 1], [2, 3]]):
            with self.subTest(groups=groups):
                model = tiny_model(groups, muon=True).eval()
                assert_causal(model, self.ids, 4, atol=1e-3)

    def test_final_state_negative_control(self):
        # Reproduce the old handoff without retaining acausal production code.
        torch.manual_seed(7)
        weights = [torch.randn(1, 4, 4, device=DEVICE) * .4 for _ in range(3)]
        q, k, v, reader_q = [torch.randn(1, 13, 4, device=DEVICE) * .5 for _ in range(4)]
        lr = torch.full((1, 13, 1), .1, device=DEVICE)
        retention = torch.full_like(lr, .98)
        def run(values):
            out, *_, traj = write_memory(*weights, q, k, values, lr, lr, lr,
                                         chunk_size=4, retention=retention, return_trajectory=True)
            torch.testing.assert_close(out, read_memory(traj, q, 4))
            bad = tuple(t[-1:].expand_as(t) for t in traj)
            return read_memory(traj, reader_q, 4), read_memory(bad, reader_q, 4)
        good, bad = run(v)
        changed = v.clone(); changed[:, 5] += 1
        good_changed, bad_changed = run(changed)
        torch.testing.assert_close(good[:, :5], good_changed[:, :5], rtol=0, atol=0)
        self.assertGreater((bad[:, :5] - bad_changed[:, :5]).abs().max().item(), 1e-6)

    def test_trajectory_shape_validation(self):
        traj = tuple(torch.ones(1, 1, 4, 4, device=DEVICE) for _ in range(3))
        with self.assertRaises(ValueError):
            read_memory(traj, torch.ones(1, 9, 4, device=DEVICE), 4)

    @torch.no_grad()
    def test_decode_matches_full_prefix(self):
        for groups in (None, [[0, 1, 2]], [[0, 1], [2, 3]]):
            for start in (3, 4, 5, 8):
                with self.subTest(groups=groups, prefill=start):
                    model = tiny_model(groups, muon=True).eval()
                    full = model(self.ids, use_cache=False).logits
                    prefill = model(self.ids[:, :start], use_cache=True)
                    torch.testing.assert_close(prefill.logits, full[:, :start], atol=1e-3, rtol=1e-3)
                    cache = prefill.past_key_values
                    self.assertEqual(cache.get_seq_length(), start)
                    for end in range(start, self.ids.shape[1]):
                        out = model(self.ids[:, end:end+1], past_key_values=cache, use_cache=True)
                        cache = out.past_key_values
                        self.assertEqual(cache.get_seq_length(), end+1)
                        torch.testing.assert_close(out.logits[:, 0], full[:, end], atol=1e-3, rtol=1e-3)
                    if groups:
                        for group in groups:
                            for reader in sorted(group)[1:]:
                                self.assertNotIn('w0', cache.states[reader])

    @torch.no_grad()
    def test_interleaved_requests_and_generate(self):
        model = tiny_model([[0, 1, 2]]).eval()
        a = model(self.ids[:1, :4], use_cache=True).past_key_values
        b = model(self.ids[1:, :7], use_cache=True).past_key_values
        # An uncached evaluation must not overwrite either request's fast weights.
        model(self.ids, use_cache=False)
        for row, end, cache in ((0, 4, a), (1, 7, b)):
            got = model(self.ids[row:row+1, end:end+1], past_key_values=cache, use_cache=True).logits
            full = model(self.ids[row:row+1, :end+1], use_cache=False).logits
            torch.testing.assert_close(got[:, -1], full[:, -1], atol=1e-3, rtol=1e-3)
        kwargs = dict(max_new_tokens=6, do_sample=False)
        cached = model.generate(self.ids[:, :3], use_cache=True, **kwargs)
        uncached = model.generate(self.ids[:, :3], use_cache=False, **kwargs)
        self.assertTrue(torch.equal(cached, uncached))

    def test_checkpointing_gradients_and_multiple_forwards(self):
        for muon in (False, True):
            model = tiny_model([[0, 1, 2]], muon=muon).train()
            checked = copy.deepcopy(model)
            checked.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            # Keep two forwards alive to catch accidental cross-microbatch stores.
            def backward(m):
                losses = [m(ids, labels=ids, use_cache=False, output_attentions=True)
                          for ids in (self.ids, self.ids.flip(1))]
                loss = sum(o.loss + sum(a[0].float().square().mean() for a in o.attentions)
                           for o in losses)
                loss.backward()
                return loss
            torch.testing.assert_close(backward(model), backward(checked))
            for (name, p), (other_name, other) in zip(model.named_parameters(), checked.named_parameters()):
                self.assertEqual(name, other_name)
                self.assertEqual(p.grad is None, other.grad is None, name)
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    torch.testing.assert_close(p.grad, other.grad, atol=1e-4, rtol=1e-3, msg=name)
            # A writer learns updates; the read-only followers don't run lr_proj.
            self.assertIsNotNone(model.model.layers[0].self_attn.lr_proj.weight.grad)
            self.assertIsNone(model.model.layers[1].self_attn.lr_proj.weight.grad)

    @torch.no_grad()
    def test_save_reload_shared_parameters_and_outputs(self):
        model = tiny_model([[0, 1], [2, 3]]).eval()
        expected = model(self.ids).logits
        for safe in (False, True):
            with self.subTest(safe=safe), tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder, safe_serialization=safe)
                restored = LigerGLAForCausalLM.from_pretrained(folder, torch_dtype=model.dtype).to(DEVICE).eval()
                for leader, reader in ((0, 1), (2, 3)):
                    a, b = (restored.model.layers[i].self_attn for i in (leader, reader))
                    for w in ('w0', 'w1', 'w2'):
                        self.assertIs(getattr(a, w), getattr(b, w))
                torch.testing.assert_close(expected, restored(self.ids).logits)

    def test_invalid_groups_rejected(self):
        for groups in ([[]], [[0, 0]], [[0, 1], [1, 2]], [[-1, 0]], [[0, 4]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                tiny_model(groups)

    def test_two_stage_training_and_peft_checkpoint(self):
        from omegaconf import OmegaConf
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from Training.train import set_trainable_params
        from Training.trainer import DefaultTrainer, FinetuneTrainer, save_checkpoint
        from eval import load_ttt_params
        cfg = OmegaConf.create({'model': {'attn_variant': 'ttt'},
                                'train': {'mse_factor': 10, 'lm_loss_weight': 1}})
        batch = {'input_ids': self.ids, 'labels': self.ids}
        model = tiny_model([[0, 1, 2]])

        def step(m, trainer_class):
            set_trainable_params(m, cfg)
            m.train()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            trainer = trainer_class.__new__(trainer_class)
            trainer.config = cfg
            trainer.criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=.001)
            before = next(p for n, p in m.named_parameters() if n.endswith('self_attn.w0')).detach().clone()
            loss = trainer.compute_loss(m, batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            after = next(p for n, p in m.named_parameters() if n.endswith('self_attn.w0'))
            self.assertFalse(torch.equal(before, after))
            optimizer.zero_grad()

        step(model, DefaultTrainer)
        with tempfile.TemporaryDirectory() as folder:
            base_path, adapter_path = folder + '/base', folder + '/adapter'
            tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
            save_checkpoint(model, tokenizer, base_path)
            stage2 = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=2,
                target_modules=['q_proj', 'k_proj', 'v_proj'],
            ))
            step(stage2, FinetuneTrainer)
            stage2.eval()
            with torch.no_grad():
                expected = stage2(self.ids, use_cache=False).logits
            save_checkpoint(stage2, tokenizer, adapter_path)
            restored = LigerGLAForCausalLM.from_pretrained(
                base_path, device_map={'': DEVICE}, torch_dtype=model.dtype
            )
            self.assertGreater(load_ttt_params(restored, adapter_path), 0)
            restored = PeftModel.from_pretrained(
                restored, adapter_path, device_map={'': DEVICE}
            ).merge_and_unload().eval()
            with torch.no_grad():
                torch.testing.assert_close(expected, restored(self.ids, use_cache=False).logits,
                                           atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_sharded_device_map_reload(self):
        model = tiny_model([[0, 1], [2, 3]]).eval()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder, max_shard_size='10KB')
            restored = LigerGLAForCausalLM.from_pretrained(
                folder, device_map={'': DEVICE}, torch_dtype=model.dtype
            ).eval()
            self.assertIs(restored.model.layers[0].self_attn.w0, restored.model.layers[1].self_attn.w0)
            torch.testing.assert_close(model(self.ids).logits, restored(self.ids).logits)

    def test_checkpointing_default_and_reentrant_rejection(self):
        model = tiny_model([[0, 1, 2]]).train()
        model.gradient_checkpointing_enable()
        model(self.ids, labels=self.ids).loss.backward()
        self.assertIsNotNone(model.model.layers[0].self_attn.w0.grad)
        with self.assertRaises(ValueError):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})


if __name__ == '__main__':
    unittest.main()
