"""Reader alignment invariance, loading, training scope, and causal decode."""
import tempfile
import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from LinearTTT import LigerGLAConfig, LigerGLAForCausalLM
from LinearTTT.model.LinearizeLlama.LinearizeLlama import ReaderOutputAlignment
from Training.train import resume_stage2, set_trainable_params
from Training.trainer import save_checkpoint
from diag_similarity import collect_readouts


def tiny(alignment='linear'):
    torch.manual_seed(42)
    return LigerGLAForCausalLM(LigerGLAConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, lact_chunk_size=4, window_size=4,
        ttt_inner_loss='l2', ttt_share_groups=[[0, 1, 2]],
        ttt_reader_alignment=alignment, ttt_use_momentum=True,
        ttt_use_muon=False, ttt_base_lr=.05, ttt_scale_init_bias=1.,
        bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=False,
    )).eval()


class ReaderAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.ids = torch.randint(1, 32, (2, 15))

    def test_batched_head_mapping_orientation(self):
        layer = ReaderOutputAlignment(2, 3)
        x = torch.randn(6, 5, 3)
        torch.testing.assert_close(layer(x), x, atol=0, rtol=0)
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight))
        expected = torch.stack([x[i] @ layer.weight[i % 2].T for i in range(6)])
        torch.testing.assert_close(layer(x), expected)

    @torch.no_grad()
    def test_identity_matches_existing_model_and_scope(self):
        baseline, aligned = tiny('none'), tiny()
        missing = aligned.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(set(missing.missing_keys), {
            f'model.layers.{i}.self_attn.ttt_reader_alignment.weight' for i in (1,2)})
        self.assertEqual(missing.unexpected_keys, [])
        torch.testing.assert_close(aligned(self.ids).logits, baseline(self.ids).logits, atol=0, rtol=0)
        for i in (0,3):
            self.assertIsNone(aligned.model.layers[i].self_attn.ttt_reader_alignment)
        self.assertIsNot(aligned.model.layers[1].self_attn.ttt_reader_alignment.weight,
                         aligned.model.layers[2].self_attn.ttt_reader_alignment.weight)

    @torch.no_grad()
    def test_load_old_checkpoint_then_save_reload_learned_map(self):
        baseline = tiny('none')
        tok = SimpleNamespace(save_pretrained=lambda path: None)
        with tempfile.TemporaryDirectory() as old, tempfile.TemporaryDirectory() as new:
            save_checkpoint(baseline, tok, old)
            cfg = LigerGLAConfig.from_pretrained(old)
            cfg.ttt_reader_alignment = 'linear'
            aligned = LigerGLAForCausalLM.from_pretrained(old, config=cfg, device_map={'':'cpu'}).eval()
            torch.testing.assert_close(aligned(self.ids).logits, baseline(self.ids).logits, atol=0, rtol=0)
            for i in (1,2):
                w = aligned.model.layers[i].self_attn.ttt_reader_alignment.weight
                torch.testing.assert_close(w, torch.eye(8).repeat(2,1,1), atol=0, rtol=0)
                w.add_(torch.randn_like(w)*.2)
            logits = aligned(self.ids).logits
            self.assertGreater((logits-baseline(self.ids).logits).abs().max().item(), 1e-5)
            save_checkpoint(aligned, tok, new)
            restored = LigerGLAForCausalLM.from_pretrained(new, device_map={'':'cpu'}).eval()
            torch.testing.assert_close(restored(self.ids).logits, logits, atol=0, rtol=0)
            for i in (1,2):
                restored.model.layers[i].self_attn._identity_reader_alignment = True
            torch.testing.assert_close(restored(self.ids).logits, baseline(self.ids).logits, atol=0, rtol=0)

    def test_only_maps_update_with_checkpointing(self):
        model = tiny().train()
        config = OmegaConf.create({'model': {'attn_varient':'ttt'}, 'train':{'reader_alignment_only':True}})
        set_trainable_params(model, config)
        trainable = [n for n,p in model.named_parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 2)
        self.assertTrue(all('ttt_reader_alignment.weight' in n for n in trainable))
        before = {n:p.clone() for n,p in model.named_parameters()}
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        for _ in range(2):
            optimizer.zero_grad()
            logits = model(self.ids, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32), self.ids[:,1:].reshape(-1))
            loss.backward()
            for name,param in model.named_parameters():
                if param.requires_grad:
                    self.assertTrue(torch.isfinite(param.grad).all())
                    self.assertGreater(param.grad.norm().item(), 0)
            optimizer.step()
        for name,param in model.named_parameters():
            if name in trainable:
                self.assertFalse(torch.equal(param, before[name]))
            else:
                torch.testing.assert_close(param, before[name], atol=0, rtol=0)

    @torch.no_grad()
    def test_resume_includes_stage2_ttt_and_lora(self):
        from peft import LoraConfig, get_peft_model
        original = tiny('none')
        stage1 = {k:v.clone() for k,v in original.state_dict().items()}
        wrapped = get_peft_model(original, LoraConfig(task_type='CAUSAL_LM', r=2,
                                                     target_modules=['q_proj']))
        set_trainable_params(wrapped, OmegaConf.create({'model':{'attn_varient':'ttt'}, 'train':{}}))
        for name,param in wrapped.named_parameters():
            if param.requires_grad:
                param.add_(torch.randn_like(param)*.03)
        expected = wrapped.eval()(self.ids, use_cache=False).logits
        tok = SimpleNamespace(save_pretrained=lambda path:None)
        with tempfile.TemporaryDirectory() as path:
            save_checkpoint(wrapped, tok, path)
            target = tiny()
            target.load_state_dict(stage1, strict=False)
            resumed = resume_stage2(target, path).eval()
            torch.testing.assert_close(resumed(self.ids).logits, expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_nonidentity_decode_and_causality(self):
        model = tiny()
        for layer in model.model.layers:
            if layer.self_attn.ttt_reader_alignment is not None:
                w = layer.self_attn.ttt_reader_alignment.weight
                w.add_(torch.randn_like(w)*.2)
        full = model(self.ids, use_cache=False).logits
        for position in (3,4,5,8,14):
            edited = self.ids.clone()
            edited[:,position] = (edited[:,position]+1)%32
            changed = model(edited, use_cache=False).logits
            torch.testing.assert_close(changed[:,:position], full[:,:position], atol=1e-6, rtol=0)
        for start in (3,4,5):
            cache = model(self.ids[:,:start], use_cache=True).past_key_values
            for t in range(start,15):
                output = model(self.ids[:,t:t+1], past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                torch.testing.assert_close(output.logits[:,0], full[:,t], atol=1e-5, rtol=1e-4)
            self.assertNotIn('w0', cache.states[1])

    @torch.no_grad()
    def test_similarity_reference_includes_alignment(self):
        model = tiny()
        mods = [l.self_attn for l in model.model.layers]
        for layer in mods:
            if layer.ttt_reader_alignment is not None:
                layer.ttt_reader_alignment.weight.add_(torch.randn_like(layer.ttt_reader_alignment.weight)*.2)
        features = collect_readouts(model, mods, self.ids[:1], torch.arange(15))
        for layer in (1,2):
            torch.testing.assert_close(features[layer]['history_effect'][:4], torch.zeros(4,16), atol=1e-6, rtol=0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            LigerGLAConfig(ttt_reader_alignment='rotation')
        with self.assertRaises(ValueError):
            LigerGLAConfig(ttt_reader_alignment='linear')
        config = OmegaConf.create({'model':{'attn_varient':'ttt'}, 'train':{'reader_alignment_only':True}})
        with self.assertRaises(ValueError):
            set_trainable_params(tiny('none'), config)


if __name__=='__main__':
    unittest.main()
