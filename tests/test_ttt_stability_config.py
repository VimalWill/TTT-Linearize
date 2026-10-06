import unittest
from pathlib import Path
from omegaconf import OmegaConf

REQUIRED = ('ttt_base_lr', 'ttt_retention_init_bias', 'fw_init_gain',
            'ttt_use_momentum', 'ttt_use_muon')


class TTTStabilityDefaultsTest(unittest.TestCase):
    """Omitting these silently takes the model defaults, which diverge.

    ttt_use_muon defaults to False and an l2 inner loss needs Muon to bound
    each update's spectral norm. A long-context config that leaves it out
    trains a memory that goes non-finite by the third inner update, so the
    omission has to fail here rather than eight GPU-hours later.
    """

    def test_every_ttt_config_pins_the_stability_parameters(self):
        reference = OmegaConf.load('Configs/ttt_ar_unified_longalpaca.yml').model
        configs = sorted(Path('Configs').glob('*long_context*.yml'))
        self.assertTrue(configs, 'no long-context configs found')
        for path in configs:
            model = OmegaConf.load(path).model
            if str(model.get('attn_varient')) != 'ttt':
                continue
            for key in REQUIRED:
                self.assertIn(key, model, f'{path.name} omits {key}')
                self.assertEqual(model[key], reference[key],
                                 f'{path.name}: {key} differs from the working config')


if __name__ == '__main__':
    unittest.main()
