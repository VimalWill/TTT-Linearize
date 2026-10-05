"""Export NIAH heatmaps, paired memory gains, and full RULER tables/curves."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path


def wilson_interval(successes, count):
    if not count:
        return None, None
    p, z = successes / count, 1.959963984540054
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return max(0, center-radius), min(1, center+radius)


def summarize_niah(report):
    cells, samples = defaultdict(list), defaultdict(dict)
    for case in report['cases']:
        if case['benchmark'] != 'niah' or case['status'] != 'ok':
            continue  # Failed/partial lengths are missing data, never zero accuracy.
        path = Path(case['result_path']).with_suffix('.samples.jsonl')
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = (case['arm'], bool(case['ablate']), row['task'],
                       row['length_bucket'], row['depth_percent'])
                cells[key].append(row['exact_match'])
                samples[(case['arm'], bool(case['ablate']), row['task'], row['length_bucket'])][row['id']] = row
    rows = []
    for (arm, ablated, task, length, depth), values in sorted(cells.items()):
        low, high = wilson_interval(sum(values), len(values))
        rows.append(dict(arm=arm, ablated=ablated, task=task, length=length, depth=depth,
                         count=len(values), accuracy=sum(values)/len(values),
                         ci_low=low, ci_high=high))
    # Bootstrap paired differences using exactly the same examples, not the
    # difference between two unrelated standard errors.
    import numpy as np
    rng = np.random.default_rng(2026)
    gains = []
    for (arm, ablated, task, length), enabled in sorted(samples.items()):
        if ablated or arm == 'baseline':
            continue
        disabled = samples.get((arm, True, task, length), {})
        if not disabled:
            continue
        if enabled.keys() != disabled.keys():
            raise ValueError('Memory-on/off samples do not match')
        differences = []
        for identity, on in enabled.items():
            off = disabled[identity]
            if on['answer'] != off['answer'] or on['prompt_tokens'] != off['prompt_tokens']:
                raise ValueError('Memory-on/off evaluation inputs changed')
            differences.append(on['exact_match'] - off['exact_match'])
        differences = np.array(differences, dtype=float)
        bootstraps = differences[rng.integers(0, len(differences), (2000, len(differences)))].mean(axis=1)
        gains.append(dict(arm=arm, task=task, length=length, count=len(differences),
                          accuracy_gain=float(differences.mean()),
                          ci_low=float(np.quantile(bootstraps, .025)),
                          ci_high=float(np.quantile(bootstraps, .975))))
    return rows, gains


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='evaluation matrix output directory')
    parser.add_argument('--out-dir', help='default: RESULTS/plots')
    args = parser.parse_args()
    directory = Path(args.results)
    output = Path(args.out_dir) if args.out_dir else directory / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((directory / 'summary.json').read_text())
    cells, gains = summarize_niah(report)
    write_csv(output / 'niah_cells.csv', cells)
    write_csv(output / 'paired_memory_gains.csv', gains)
    ruler = [dict(arm=c['arm'], ablated=bool(c['ablate']), length=c['length'],
                  macro=c['ruler_macro'], **c['task_scores']) for c in report['cases']
             if c['benchmark'] == 'ruler' and c['status'] == 'ok']
    write_csv(output / 'ruler_scores.csv', ruler)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    lengths = report['config']['lengths']
    depths = sorted(set(r['depth'] for r in cells))
    for arm, ablated, task in sorted(set((r['arm'], r['ablated'], r['task']) for r in cells)):
        values = np.full((len(depths), len(lengths)), np.nan)
        for row in cells:
            if (row['arm'], row['ablated'], row['task']) == (arm, ablated, task):
                values[depths.index(row['depth']), lengths.index(row['length'])] = row['accuracy']
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap('RdYlGn').copy()
        cmap.set_bad('#cccccc')
        display = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(lengths)), [f'{n//1024}K' for n in lengths])
        ax.set_yticks(range(len(depths)), depths)
        ax.set_xlabel('Context length (tokens)')
        ax.set_ylabel('Needle insertion depth (%)')
        ax.set_title(f'{arm} / {task}' + (' / memory off' if ablated else ''))
        for i in range(len(depths)):
            for j in range(len(lengths)):
                text = 'N/A' if np.isnan(values[i,j]) else f'{100*values[i,j]:.0f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9)
        fig.colorbar(display, ax=ax, label='Exact-match accuracy')
        fig.tight_layout()
        stem = f'niah_{arm}_{task}' + ('_memory_off' if ablated else '')
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'{stem}.{extension}', dpi=180)
        plt.close(fig)
    if ruler:
        fig, ax = plt.subplots(figsize=(7, 4))
        for arm, ablated in sorted(set((r['arm'], r['ablated']) for r in ruler)):
            points = sorted((r for r in ruler if (r['arm'], r['ablated']) == (arm, ablated)),
                            key=lambda r: r['length'])
            ax.plot([r['length'] for r in points], [100*r['macro'] for r in points],
                    '--' if ablated else '-', marker='o', label=arm + (' memory off' if ablated else ''))
        ax.set_xscale('log', base=2)
        ax.set_xticks(lengths, [f'{n//1024}K' for n in lengths])
        ax.set_ylim(0, 100)
        ax.set_xlabel('Context length (tokens)')
        ax.set_ylabel('RULER macro score (%)')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(output / f'ruler_by_length.{extension}', dpi=180)
        plt.close(fig)
    print(f'Wrote figures and CSVs to {output}')


if __name__ == '__main__':
    main()
