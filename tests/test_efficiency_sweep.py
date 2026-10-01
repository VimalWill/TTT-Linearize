"""OOM isolation and capacity-search regressions; no GPU or downloads."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import measure_flops
import sweep_efficiency
from sweep_efficiency import Arm, CaseRunner, calibrate_batch, find_oom_boundary, runtime_oom


def capacity_probe(length, batch):
    return dict(status='ok' if length * batch <= 262144 else 'oom', phase='timing')


class CapacitySearchTests(unittest.TestCase):
    def test_calibration_chooses_smallest_16k_failure_that_fits_8k(self):
        result = calibrate_batch(capacity_probe)
        self.assertTrue(result['target_met'])
        self.assertEqual(result['batch'], 17)
        self.assertEqual(capacity_probe(8192, result['batch'])['status'], 'ok')
        self.assertEqual(capacity_probe(16384, result['batch'])['status'], 'oom')

    def test_boundary_reports_fit_and_oom_with_requested_resolution(self):
        result = find_oom_boundary(capacity_probe, 17, resolution=256)
        self.assertTrue(result['bracket_found'])
        self.assertEqual(capacity_probe(result['max_tested_fit_tokens'], 17)['status'], 'ok')
        self.assertEqual(capacity_probe(result['min_tested_oom_tokens'], 17)['status'], 'oom')
        self.assertLessEqual(result['interval_width_tokens'], 256)
        exact = find_oom_boundary(capacity_probe, 17, resolution=1)
        self.assertEqual(exact['min_tested_oom_tokens'], exact['max_tested_fit_tokens'] + 1)

    def test_no_failure_within_batch_limit_is_reported(self):
        result = calibrate_batch(lambda length, batch: dict(status='ok'), max_batch=4)
        self.assertFalse(result['target_met'])
        self.assertEqual(result['batch'], 4)
        self.assertFalse(find_oom_boundary(lambda length, batch: dict(status='ok'), 4)['bracket_found'])

    def test_load_failure_and_counter_failure_are_not_runtime_capacity(self):
        self.assertFalse(runtime_oom(dict(status='oom', phase='model_load')))
        self.assertFalse(runtime_oom(dict(status='ok', phase='complete', flops_status='oom')))
        result = calibrate_batch(lambda length, batch: dict(
            status='oom', phase='model_load', error='weights do not fit'))
        self.assertFalse(result['target_met'])


class WorkerStatusTests(unittest.TestCase):
    def test_cuda_oom_classifier_handles_wrappers_without_hiding_other_errors(self):
        original = torch.cuda.OutOfMemoryError('allocation failed')
        wrapped = RuntimeError('compiled execution failed')
        wrapped.__cause__ = original
        self.assertTrue(measure_flops.is_cuda_oom(wrapped))
        self.assertTrue(measure_flops.is_cuda_oom(RuntimeError('CUDA_ERROR_OUT_OF_MEMORY')))
        self.assertFalse(measure_flops.is_cuda_oom(ValueError('bad config')))
        self.assertFalse(measure_flops.is_cuda_oom(RuntimeError('host out of memory')))

    def test_oom_worker_writes_shape_and_phase_before_exit(self):
        def failure(args, record):
            record['phase'] = 'warmup'
            raise torch.cuda.OutOfMemoryError('CUDA out of memory')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            with patch('sys.argv', ['measure_flops.py', '--baseline', '--seq-len', '16384',
                                    '--batch', '17', '--out', str(output)]), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch('measure_flops.run_benchmark', side_effect=failure), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = measure_flops.main()
            result = json.loads(output.read_text())
        self.assertEqual(code, 2)
        self.assertEqual((result['status'], result['phase'], result['seq_len'], result['batch']),
                         ('oom', 'warmup', 16384, 17))

    def test_counter_oom_keeps_successful_runtime_measurement(self):
        class Model(torch.nn.Module):
            device = torch.device('cpu')
            config = SimpleNamespace(vocab_size=32, window_size=8, lact_chunk_size=8,
                                     ttt_layer_indices=None, ttt_share_groups=None)

            def forward(self, input_ids, use_cache):
                return input_ids.float().sum()

        model = Model()
        args = SimpleNamespace(baseline=False, cfg='unused', ckpt='unused', adapter=None,
                               seq_len=32, batch=1, sdpa_window=False, seed=0, warmup=1,
                               reps=1, skip_flops=False, depth=0)
        record = {}
        with patch('Training.train.build_model_config', return_value=model.config), \
                patch('measure_flops.load_benchmark_config'), \
                patch('eval.load_model', return_value=model), \
                patch('measure_flops.set_window_backend') as backend, \
                patch('torch.cuda.get_device_name', return_value='mock device'), \
                patch('torch.cuda.get_device_properties', return_value=SimpleNamespace(total_memory=1000)), \
                patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
                patch('torch.cuda.memory_allocated', return_value=100), \
                patch('torch.cuda.max_memory_allocated', return_value=200), \
                patch('measure_flops.count_forward', side_effect=torch.cuda.OutOfMemoryError('CUDA out of memory')), \
                contextlib.redirect_stderr(io.StringIO()):
            measure_flops.run_benchmark(args, record)
        self.assertEqual(record['status'], 'ok')
        self.assertEqual(record['flops_status'], 'oom')
        self.assertGreater(record['latency_seconds'], 0)
        self.assertEqual(record['peak_allocated_bytes'], 200)
        self.assertFalse(backend.call_args.args[0])


class SubprocessIsolationTests(unittest.TestCase):
    def test_complete_sweep_records_oom_and_error_without_skipping_later_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / 'worker.py'
            worker.write_text('''import argparse, json, sys
p = argparse.ArgumentParser()
p.add_argument('--seq-len', type=int)
p.add_argument('--batch', type=int)
p.add_argument('--out')
p.add_argument('--baseline', action='store_true')
p.add_argument('--cfg', default='')
a, _ = p.parse_known_args()
status = 'oom' if a.baseline and a.batch*a.seq_len > 262144 else 'ok'
if 'mistral_l2' in a.cfg and a.seq_len == 16384: status = 'error'
result = dict(status=status, phase='complete' if status == 'ok' else 'timing',
              seq_len=a.seq_len, batch=a.batch,
              latency_seconds=(2. if a.baseline else 1.) if status == 'ok' else None)
with open(a.out, 'w') as f: json.dump(result, f)
sys.exit(0 if status == 'ok' else 2 if status == 'oom' else 1)
''')
            output = Path(directory) / 'sweep'
            def runner(path, **kwargs):
                return CaseRunner(path, worker=worker, **kwargs)
            with patch('sys.argv', ['sweep_efficiency.py', '--anchor-i-ckpt', 'i',
                                    '--anchor-f-ckpt', 'f', '--out-dir', str(output),
                                    '--batch', '17', '--lengths', '8192', '16384',
                                    '--oom-resolution', '8192']), \
                    patch('sweep_efficiency.CaseRunner', side_effect=runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = sweep_efficiency.main()
            report = json.loads((output / 'summary.json').read_text())
            self.assertTrue((output / 'results.csv').is_file())
        self.assertEqual(code, 1)  # Ordinary errors require attention, after all cases run.
        self.assertEqual(len(report['cases']), 16)
        indexed = {(r['arm'], r['batch'], r['seq_len']): r for r in report['cases']}
        baseline = indexed['baseline', 17, 16384]
        anchor = indexed['anchor_i_deploy', 17, 16384]
        self.assertEqual(baseline['status'], 'oom')
        self.assertEqual(anchor['status'], 'ok')
        self.assertIsNone(anchor['speedup_vs_baseline'])
        self.assertEqual(indexed['anchor_i_deploy', 1, 8192]['speedup_vs_baseline'], 2.)
        self.assertTrue(report['calibration']['target_met'])

    def test_runtime_oom_does_not_stop_the_next_config(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / 'worker.py'
            worker.write_text('''import argparse, json, sys
p = argparse.ArgumentParser()
p.add_argument('--seq-len', type=int)
p.add_argument('--batch', type=int)
p.add_argument('--out')
p.add_argument('--baseline', action='store_true')
a, _ = p.parse_known_args()
failed = a.baseline and a.seq_len >= 16384
result = dict(status='oom' if failed else 'ok', phase='timing' if failed else 'complete',
              seq_len=a.seq_len, batch=a.batch, latency_seconds=None if failed else .1)
with open(a.out, 'w') as f: json.dump(result, f)
sys.exit(2 if failed else 0)
''')
            runner = CaseRunner(Path(directory) / 'cases', worker=worker)
            baseline = runner.run(Arm('baseline', 'teacher'), 16384, 17)
            anchor = runner.run(Arm('anchor_i_deploy', 'teacher', 'config', 'checkpoint'), 16384, 17)
            another = runner.run(Arm('anchor_f', 'teacher', 'config', 'checkpoint'), 32768, 17)
        self.assertEqual(baseline['status'], 'oom')
        self.assertEqual(anchor['status'], 'ok')
        self.assertEqual(another['status'], 'ok')

    def test_crash_without_json_is_an_error_instead_of_a_cuda_oom(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / 'worker.py'
            worker.write_text('raise RuntimeError("broken kernel")\n')
            result = CaseRunner(Path(directory) / 'cases', worker=worker).run(
                Arm('baseline', 'teacher'), 8192, 1)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['phase'], 'subprocess')


if __name__ == '__main__':
    unittest.main()
