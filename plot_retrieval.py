"""Figures for the layer-wise retrieval analysis.

Replaces the two-flat-parallel-lines layout, where 30 of 32 layers convey
"nothing happens" and the signal is two points at the left edge.

  dumbbell  one glyph per layer, a segment from d(cloze) to d(letter). The
            segment LENGTH is the retrieval-specific damage and its position is
            the total damage, so the decomposition is read directly instead of
            being inferred from the gap between two series. Keeps layer order,
            which matters because "bottom of the stack" is part of the finding.
  plane     d(cloze) vs d(letter). The y=x diagonal is "no retrieval effect";
            vertical distance below it IS the retrieval score. Makes the
            knowledge/retrieval split geometric.
  agree     MMLU retrieval score vs the AR-slice far-distance dCE, one point
            per layer. Two independent probes; agreement is the validation.

    python3 plot_retrieval.py --mmlu mmlu_anchor.csv --ar retrieval_anchor.csv
"""

import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics as st

SOFTMAX_GA = (16, 17)   # Gather L16H22 / Aggregate L17H24 in Llama-3.1-8B


def read_mmlu(path):
    base_l = base_c = None
    rows = {}
    for r in csv.DictReader(open(path)):
        if r['branch'] == 'baseline':
            base_l, base_c = float(r['letter']), float(r['cloze'] or 'nan')
            continue
        if r['branch'] != 'ttt':
            continue
        rows[int(r['layer'])] = (float(r['letter']),
                                 float(r['cloze']) if r['cloze'] else None)
    if base_l is None:
        raise SystemExit(f'{path} has no baseline row -- rerun diag_mmlu.py')
    return base_l, base_c, rows


def read_ar(path, col):
    out = {}
    for r in csv.DictReader(open(path)):
        if r['branch'] == 'ttt':
            out[int(r['layer'])] = float(r[col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmlu', default='mmlu_anchor.csv')
    ap.add_argument('--ar', default=None, help='retrieval_anchor.csv, for the agreement panel')
    ap.add_argument('--ar-col', default='dCE_ar_x8192+')
    ap.add_argument('--out', default='retrieval_figs')
    args = ap.parse_args()

    bl, bc, rows = read_mmlu(args.mmlu)
    L = sorted(rows)
    dl = {i: (rows[i][0] - bl) * 100 for i in L}
    dc = {i: (rows[i][1] - bc) * 100 for i in L}
    rs = {i: dc[i] - dl[i] for i in L}            # positive = retrieval-specific

    # noise floor from the layers with no effect, so significance is empirical
    quiet = [i for i in L if i not in (0, 1)]
    sd = st.pstdev([dl[i] for i in quiet])

    n = 3 if args.ar else 2
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.1 * n))

    # ---- dumbbell ----
    ax = axes[0]
    ax.axhspan(-3 * sd, 3 * sd, color='0.9', zorder=0, label=f'3σ noise (±{3*sd:.1f} pt)')
    ax.axhline(0, color='0.4', lw=.8, zorder=1)
    for i in L:
        ax.plot([i, i], [dc[i], dl[i]], color='0.55', lw=1.4, zorder=2,
                solid_capstyle='round')
    ax.scatter(L, [dc[i] for i in L], s=26, color='#C2681A', zorder=3,
               label='knowledge (cloze)')
    ax.scatter(L, [dl[i] for i in L], s=26, color='#1F6F6B', zorder=3,
               label='knowledge + retrieval (letter)')
    for g in SOFTMAX_GA:
        ax.axvline(g, color='#C8A200', lw=6, alpha=.25, zorder=0)
    ax.annotate('G&A layers in the\nsoftmax model', xy=(16.5, ax.get_ylim()[0]),
                xytext=(19, -12), fontsize=8, color='0.35',
                arrowprops=dict(arrowstyle='-', color='0.6', lw=.8))
    ax.set_ylabel('Δ accuracy (points)')
    ax.set_xlabel('layer (TTT branch ablated)')
    ax.legend(fontsize=8, loc='lower right', framealpha=.9)
    ax.set_title('segment length = retrieval-specific damage', fontsize=9, loc='left')

    # ---- knowledge / retrieval plane ----
    ax = axes[1]
    lo = min(min(dc.values()), min(dl.values())) - 1
    ax.plot([lo, 3], [lo, 3], color='0.6', lw=1, ls='--',
            label='y = x  (damage is pure knowledge)')
    ax.axhspan(-3 * sd, 3 * sd, color='0.93', zorder=0)
    ax.axvspan(-3 * sd, 3 * sd, color='0.93', zorder=0)
    ax.scatter([dc[i] for i in L], [dl[i] for i in L], s=30,
               c=['#B03030' if i in (0, 1) else '#3B6E8F' for i in L], zorder=3)
    for i in L:
        if abs(rs[i]) > 2 * sd or i in (0, 1):
            ax.annotate(f'L{i}', (dc[i], dl[i]), textcoords='offset points',
                        xytext=(5, -3), fontsize=8)
    ax.set_xlabel('Δ cloze — knowledge only (points)')
    ax.set_ylabel('Δ letter — knowledge\n+ retrieval (points)')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('below the diagonal = retrieval-specific', fontsize=9, loc='left')

    # ---- cross-probe agreement ----
    if args.ar:
        ce = read_ar(args.ar, args.ar_col)
        ax = axes[2]
        xs = [ce[i] for i in L if i in ce]
        ys = [rs[i] for i in L if i in ce]
        ax.scatter(xs, ys, s=30,
                   c=['#B03030' if i in (0, 1) else '#3B6E8F' for i in L if i in ce],
                   zorder=3)
        for i in L:
            if i in ce and (i in (0, 1) or abs(rs[i]) > 2 * sd):
                ax.annotate(f'L{i}', (ce[i], rs[i]), textcoords='offset points',
                            xytext=(5, -3), fontsize=8)
        ax.axhline(0, color='0.6', lw=.8)
        ax.axvline(0, color='0.6', lw=.8)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.set_xlabel(f'AR-slice {args.ar_col} at 32k (nats, symlog)')
        ax.set_ylabel('MMLU retrieval\nscore (points)')
        ax.set_title('two independent probes agree', fontsize=9, loc='left')

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=170, bbox_inches='tight')
    print(f'noise floor: sd {sd:.2f} pt, 3σ ±{3*sd:.2f}')
    print(f'wrote {args.out}.png / .pdf')


if __name__ == '__main__':
    main()
