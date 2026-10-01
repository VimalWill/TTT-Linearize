"""Cached prefill/decode sweep; each GPU case runs in a fresh worker process.

Default prompt lengths are 4K/8K/16K/32K, followed by 512 cached decode
forwards. Baseline, Anchor-F, Anchor-I full, and Anchor (Anchor-I deployment)
use identical batch sizes and decode lengths. Automatic stress-batch selection
tries to bracket baseline runtime OOM between 8K and 16K prompt tokens.
The JSON/CSV reports separate prefill/decode speedups and actual retained
cache storage. Capacity includes prefill activations and decode state; an OOM
is not necessarily caused by cache alone. Baseline cache policy is the native
HF implementation and is recorded, never changed just to induce an OOM.
"""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


@dataclass
class Arm:
    name: str
    base: str
    cfg: str = None
    ckpt: str = None
    adapter: str = None


def runtime_oom(result):
    return result['status'] == 'oom' and result.get('phase') in ('inputs', 'warmup', 'prefill', 'decode', 'timing')


def calibrate_batch(probe, low=8192, high=16384, max_batch=128):
    """Find the smallest batch that fails high; verify that it fits low.

    Capacity is assumed monotonic in batch, on an otherwise idle GPU. A failure
    to bracket the target is reported explicitly; no allocator limit or slower
    attention backend is introduced just to manufacture an OOM.
    """
    last_fit, candidate = 0, 1
    while True:
        result = probe(high, candidate)
        if runtime_oom(result):
            break
        if result['status'] != 'ok':
            return dict(batch=max(1, last_fit), target_met=False,
                        reason=f"Calibration failed at {result.get('phase')}: {result.get('error')}")
        last_fit = candidate
        if candidate == max_batch:
            return dict(batch=candidate, target_met=False,
                        reason=f'Baseline still fits {high} tokens at max_batch={max_batch}')
        candidate = min(2 * candidate, max_batch)
    # Refine the batch threshold instead of keeping the first power-of-two
    # failure, which could also OOM at the lower endpoint.
    bad = candidate
    while bad - last_fit > 1:
        middle = (last_fit + bad) // 2
        result = probe(high, middle)
        if result['status'] == 'ok':
            last_fit = middle
        elif runtime_oom(result):
            bad = middle
        else:
            return dict(batch=max(1, last_fit), target_met=False,
                        reason=f"Calibration failed at {result.get('phase')}: {result.get('error')}")
    lower = probe(low, bad)
    if lower['status'] == 'ok':
        return dict(batch=bad, target_met=True,
                    reason=f'Baseline fits {low} tokens and runtime-OOMs at {high}')
    return dict(batch=1, target_met=False,
                reason=f'No verified batch bracket: batch {bad} does not fit {low} tokens')


def find_oom_boundary(probe, batch, low=8192, high=16384, resolution=256):
    """Return a measured fit/OOM bracket, never an invented exact token limit."""
    lower, upper = probe(low, batch), probe(high, batch)
    if lower['status'] != 'ok' or not runtime_oom(upper):
        return dict(bracket_found=False, low_status=lower['status'],
                    high_status=upper['status'], reason='Endpoints do not bracket a runtime OOM')
    while high - low > resolution:
        middle = (low + high) // 2
        result = probe(middle, batch)
        if result['status'] == 'ok':
            low = middle
        elif runtime_oom(result):
            high = middle
        else:
            return dict(bracket_found=False, max_tested_fit_tokens=low,
                        min_tested_oom_tokens=high,
                        reason=f"Probe error at {middle}: {result.get('error')}")
    return dict(bracket_found=True, batch=batch, max_tested_fit_tokens=low,
                min_tested_oom_tokens=high, interval_width_tokens=high - low,
                max_tested_fit_total_tokens=batch * low,
                min_tested_oom_total_tokens=batch * high,
                token_unit='prompt tokens per sequence; fixed decode length recorded in config',
                scope='cached_prefill_and_incremental_decode_last_position_logits')


class CaseRunner:
    def __init__(self, directory, worker=None, python=sys.executable,
                 warmup=2, reps=10, seed=0, decode_tokens=512):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.worker = Path(worker or Path(__file__).with_name('measure_flops.py'))
        self.python, self.warmup, self.reps, self.seed = python, warmup, reps, seed
        self.decode_tokens = decode_tokens

    def run(self, arm, length, batch, skip_flops=False, reps=None):
        stem = f'{arm.name}_b{batch}_s{length}'
        output = self.directory / f'{stem}.json'
        log = self.directory / f'{stem}.log'
        command = [self.python, str(self.worker), '--seq-len', str(length),
                   '--batch', str(batch), '--warmup', str(self.warmup),
                   '--reps', str(self.reps if reps is None else reps),
                   '--seed', str(self.seed), '--decode-tokens', str(self.decode_tokens),
                   '--out', str(output)]
        if arm.cfg:
            command += ['--cfg', arm.cfg, '--ckpt', arm.ckpt]
            if arm.adapter:
                command += ['--adapter', arm.adapter]
        else:
            command += ['--baseline', '--base', arm.base]
        if skip_flops:
            command.append('--skip-flops')
        print(f'START {arm.name} batch={batch} length={length}; log={log}', flush=True)
        with log.open('w') as stream:
            # No check=True: OOM/error is a recorded case, not the end of a sweep.
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            result = json.loads(output.read_text())
            if not isinstance(result, dict) or result.get('status') not in ('ok', 'oom', 'error'):
                raise ValueError('Invalid worker status')
            if result.get('seq_len') != length or result.get('batch') != batch:
                raise ValueError('Worker reported a different input shape')
            if result['status'] == 'ok' and not result.get('latency_seconds', 0) > 0:
                raise ValueError('Successful worker report lacks a positive latency')
            if result['status'] == 'ok' and (result.get('use_cache') is not True
                    or result.get('decode_tokens') != self.decode_tokens):
                raise ValueError('Worker did not report the requested cached inference workload')
            if completed.returncode != 0 and result['status'] == 'ok':
                raise ValueError(f'Worker exited {completed.returncode} after reporting success')
        except (OSError, ValueError) as error:
            # SIGKILL, host OOM, and missing reports are errors, not evidence of
            # a CUDA capacity threshold. Preserve the log for diagnosis.
            result = dict(status='error', phase='subprocess',
                          error=f'Worker exit {completed.returncode}; {error}',
                          seq_len=length, batch=batch)
        result.update(arm=arm.name, decode_tokens=self.decode_tokens, returncode=completed.returncode,
                      command=command, log=str(log), result_path=str(output))
        detail = (f" {result['latency_seconds'] * 1e3:.1f} ms"
                  if result['status'] == 'ok' else f" phase={result.get('phase')}")
        print(f"{arm.name:16} batch={batch:<3} length={length:<6} {result['status'].upper()}{detail}", flush=True)
        return result


def write_summary(directory, report):
    path = Path(directory) / 'summary.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    fields = ['arm', 'batch', 'seq_len', 'decode_tokens', 'use_cache', 'scope',
              'status', 'phase', 'latency_seconds',
              'prefill_latency_seconds', 'decode_latency_seconds', 'decode_seconds_per_step',
              'prefill_tokens_per_second', 'decode_tokens_per_second',
              'cache_class', 'prefill_cache_bytes', 'decode_cache_bytes',
              'tokens_per_second', 'peak_allocated_bytes', 'above_resident_bytes',
              'counted_flops', 'decode_counted_flops', 'flops_status', 'baseline_status',
              'prefill_speedup_vs_baseline', 'decode_speedup_vs_baseline',
              'speedup_vs_baseline', 'error', 'log']
    with (Path(directory) / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['cases'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--anchor-i-ckpt', required=True)
    parser.add_argument('--anchor-i-adapter')
    parser.add_argument('--anchor-f-ckpt', required=True)
    parser.add_argument('--anchor-f-adapter')
    parser.add_argument('--anchor-i-cfg', default='Configs/ttt_ar_mistral_anchor.yml')
    parser.add_argument('--anchor-i-deploy-cfg', default='Configs/ttt_deploy_mistral_anchor.yml')
    parser.add_argument('--anchor-f-cfg', default='Configs/ttt_ar_mistral_l2.yml')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096, 8192, 16384, 32768])
    parser.add_argument('--decode-tokens', type=int, default=512,
                        help='incremental cached forwards after each prefill')
    parser.add_argument('--batch', type=int, help='fixed stress batch; otherwise calibrate it')
    parser.add_argument('--max-batch', type=int, default=128, help='automatic calibration upper bound')
    parser.add_argument('--oom-low', type=int, default=8192)
    parser.add_argument('--oom-high', type=int, default=16384)
    parser.add_argument('--oom-resolution', type=int, default=256,
                        help='maximum token width of reported fit/OOM interval; 1 for exact probes')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--reps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-flops', action='store_true', help='skip reference FLOPs in batch-1 cases too')
    args = parser.parse_args()
    if min(args.lengths + [args.max_batch, args.oom_low, args.oom_resolution,
                           args.warmup, args.reps, args.decode_tokens]) < 1 or args.oom_high <= args.oom_low:
        parser.error('Lengths, batches, reps and resolution must be positive; oom-high must exceed oom-low')
    if args.batch is not None and args.batch < 1:
        parser.error('--batch must be positive')
    directory = Path(args.out_dir)
    if directory.exists() and any(directory.iterdir()):
        parser.error('--out-dir must be new or empty')
    directory.mkdir(parents=True, exist_ok=True)
    baseline = Arm('baseline', args.base)
    arms = [baseline,
            Arm('anchor_i_full', args.base, args.anchor_i_cfg, args.anchor_i_ckpt, args.anchor_i_adapter),
            Arm('anchor_i_deploy', args.base, args.anchor_i_deploy_cfg, args.anchor_i_ckpt, args.anchor_i_adapter),
            Arm('anchor_f', args.base, args.anchor_f_cfg, args.anchor_f_ckpt, args.anchor_f_adapter)]
    report = dict(config=vars(args), calibration=None, oom_boundary=None, probes=[], cases=[])
    write_summary(directory, report)
    probe_runner = CaseRunner(directory / 'probes', warmup=args.warmup, reps=1, seed=args.seed,
                              decode_tokens=args.decode_tokens)
    cache = {}

    def probe(length, batch):
        key = length, batch
        if key not in cache:
            cache[key] = probe_runner.run(baseline, length, batch, skip_flops=True)
            report['probes'].append(cache[key])
            write_summary(directory, report)
        return cache[key]

    if args.batch is None:
        report['calibration'] = calibrate_batch(probe, args.oom_low, args.oom_high, args.max_batch)
        batch = report['calibration']['batch']
    else:
        batch = args.batch
        report['calibration'] = dict(batch=batch, target_met=False, reason='User-specified batch')
    report['oom_boundary'] = find_oom_boundary(probe, batch, args.oom_low,
                                               args.oom_high, args.oom_resolution)
    if args.batch is not None:
        report['calibration']['target_met'] = report['oom_boundary']['bracket_found']
    write_summary(directory, report)
    print(f"\nSelected stress batch: {batch}; {report['calibration']['reason']}", flush=True)
    print(f"Baseline OOM boundary: {report['oom_boundary']}\n", flush=True)

    runner = CaseRunner(directory / 'cases', warmup=args.warmup, reps=args.reps, seed=args.seed,
                        decode_tokens=args.decode_tokens)
    # Batch 1 shows context scaling. A second, fixed batch tests capacity; never
    # shrink it per arm or length, since that would invalidate the comparison.
    for current_batch in sorted({1, batch}):
        for length in args.lengths:
            reference = None
            for arm in arms:
                result = runner.run(arm, length, current_batch,
                                    skip_flops=args.skip_flops or current_batch != 1)
                if arm.name == 'baseline':
                    reference = result
                result['baseline_status'] = reference['status']
                result['speedup_vs_baseline'] = (
                    reference['latency_seconds'] / result['latency_seconds']
                    if reference['status'] == result['status'] == 'ok' else None)
                for metric in ('prefill', 'decode'):
                    field = f'{metric}_latency_seconds'
                    result[f'{metric}_speedup_vs_baseline'] = (
                        reference[field] / result[field]
                        if reference['status'] == result['status'] == 'ok'
                        and reference.get(field) and result.get(field) else None)
                report['cases'].append(result)
                write_summary(directory, report)
    print(f'\nWrote {directory / "summary.json"} and {directory / "results.csv"}', flush=True)
    # Ordinary errors require attention; expected OOMs do not fail the job.
    return 1 if any(r['status'] == 'error' or r.get('flops_status') == 'error'
                    for r in report['cases'] + report['probes']) else 0


if __name__ == '__main__':
    sys.exit(main())
