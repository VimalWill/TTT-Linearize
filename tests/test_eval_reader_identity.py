import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from eval_reader_identity import aggregate, identity_readers, paired_sequence
from test_reader_alignment import tiny


class IdentityEvaluationTests(unittest.TestCase):
    @torch.no_grad()
    def test_paired_score_matches_full_lm_and_is_read_only(self):
        model = tiny()
        ids = torch.randint(1,32,(1,15))
        before = {n:p.clone() for n,p in model.named_parameters()}
        logits = model(ids, use_cache=False).logits[0,:-1].float()
        expected = F.cross_entropy(logits,ids[0,1:],reduction='none').double().sum().item()/14
        readers = identity_readers(model)
        row = paired_sequence(model,ids,readers,logit_block=3,atol=0)
        self.assertAlmostEqual(row['baseline_ce'],expected,places=7)
        self.assertEqual(row['baseline_token_ppl'],row['identity_token_ppl'])
        self.assertEqual(row['max_abs_logit_delta'],0)
        self.assertEqual(row['argmax_disagreements'],0)
        self.assertTrue(row['within_tolerance'])
        self.assertTrue(all(not r._identity_reader_alignment for r in readers))
        for n,p in model.named_parameters():
            torch.testing.assert_close(p,before[n],atol=0,rtol=0)

    @torch.no_grad()
    def test_nonidentity_rejected_and_detectable(self):
        model = tiny()
        readers = identity_readers(model)
        readers[0].ttt_reader_alignment.weight.add_(torch.randn_like(readers[0].ttt_reader_alignment.weight)*.2)
        with self.assertRaisesRegex(ValueError,'not identity'):
            identity_readers(model)
        row = paired_sequence(model,torch.randint(1,32,(1,15)),readers,logit_block=4,atol=0)
        self.assertFalse(row['within_tolerance'])
        self.assertGreater(row['max_abs_logit_delta'],0)

    def test_flags_restored_on_failure(self):
        model = tiny()
        readers = identity_readers(model)
        readers[0]._identity_reader_alignment = True
        with patch.object(model.model,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                paired_sequence(model,torch.ones(1,8,dtype=torch.long),readers)
        self.assertTrue(readers[0]._identity_reader_alignment)
        self.assertFalse(readers[1]._identity_reader_alignment)

    def test_corpus_ppl_uses_total_nll_not_mean_ppl(self):
        rows = [dict(corpus='test',n_targets=n,baseline_nll_sum=n*ce,identity_nll_sum=n*ce,
                     max_abs_logit_delta=0,max_abs_token_nll_delta=0,argmax_disagreements=0,within_tolerance=True)
                for n,ce in [(2,1.),(6,3.)]]
        summary = aggregate(rows)[0]
        self.assertEqual(summary['n_targets'],8)
        self.assertAlmostEqual(summary['baseline_token_ppl'], math.exp(2.5))


if __name__=='__main__':
    unittest.main()
