"""Per-head spectra of actual TTT fast-weight states, without changing updates.

Run --help for the cluster command. Results describe matrix rank, not a
compression intervention or a perplexity benchmark. No lm_eval/metric loading.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def spectrum_stats(s):
    """Descending singular values -> energy ranks; zero spectra are explicit."""
    s = s.detach().double().cpu()
    if s.ndim != 1 or s.numel() == 0 or not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Expected a finite, nonnegative singular-value vector')
    energy = s.square()
    total = float(energy.sum())
    result = {'frobenius_norm': math.sqrt(total), 'spectral_norm': float(s[0]),
              'stable_rank': None, 'r90': 0, 'r95': 0, 'r99': 0}
    for r in (8, 16, 32, 64):
        result[f'rel_error_r{r}'] = None
    if total:
        cumulative = energy.cumsum(0) / total
        result['stable_rank'] = total / float(energy[0])
        for label, threshold in (('r90', .90), ('r95', .95), ('r99', .99)):
            result[label] = min(int(torch.searchsorted(cumulative, threshold)) + 1, len(s))
        for r in (8, 16, 32, 64):
            result[f'rel_error_r{r}'] = math.sqrt(float(energy[r:].sum()) / total)
    return result


def memory_owners(mods):
    return [m for m in mods if m._share_gid is None or m._share_leader]


@contextlib.contextmanager
def observe_states(mods, callback):
    """Scope diagnostic callbacks to independent memories; restore on failure."""
    owners = memory_owners(mods)
    if not owners:
        raise ValueError('No independent TTT memories found')
    if any(m.training or m.ttt_inner_loss != 'l2' for m in owners):
        raise ValueError('State rank probing requires eval mode and l2 memories')
    absent = object()
    previous = []
    try:
        for m in owners:
            previous.append((m, getattr(m, '_ttt_state_observer', absent)))
            m._ttt_state_observer = callback
        yield owners
    finally:
        for m, old in previous:
            if old is absent:
                delattr(m, '_ttt_state_observer')
            else:
                m._ttt_state_observer = old


def trajectory_components(layer, trajectory, chunks, components, decay=None):
    """CPU snapshots at state c: adaptation E_c and the preceding write U_{c-1}.

    Subtract in float64 to reduce diagnostic cancellation, then perform SVD in
    float32. The states themselves retain the operator's finite-precision error.
    """
    components = tuple(dict.fromkeys(components))
    if not components or set(components) - {'state', 'adaptation', 'update'}:
        raise ValueError('Unknown or empty state components')
    available, heads = trajectory[0].shape[:2]
    selected = sorted(set(c for c in chunks if 0 <= c < available))
    need_decay = any(c != 'state' for c in components)
    if need_decay:
        if decay is None or tuple(decay.shape) != (available - 1, heads, 1, 1):
            raise ValueError('Expected one applied per-head decay per transition')
        alpha = decay.detach().double().cpu()
        if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
            raise ValueError('Retention multipliers must be finite and in [0, 1]')
        accumulated = torch.cat([torch.ones(1, heads, 1, 1, dtype=torch.float64),
                                 alpha.cumprod(0)], dim=0)
    needed = sorted(set([0, *selected, *[c - 1 for c in selected if c > 0 and 'update' in components]]))
    positions = {c: i for i, c in enumerate(needed)}
    for name, states in zip(('w0', 'w1', 'w2'), trajectory):
        copied = states.detach().index_select(
            0, torch.tensor(needed, device=states.device)).float().cpu()
        initial = copied[positions[0]].double()
        for chunk in selected:
            current = copied[positions[chunk]]
            reference_norms = torch.linalg.vector_norm(current.double(), dim=(-2, -1))
            for component in components:
                if component == 'state':
                    value = current
                elif component == 'adaptation':
                    value = (current.double() - accumulated[chunk] * initial).float()
                else:
                    if chunk == 0:
                        continue  # no write has occurred before state 0
                    previous = copied[positions[chunk - 1]].double()
                    value = (current.double() - alpha[chunk - 1] * previous).float()
                yield {
                    'layer': layer.layer_idx,
                    'role': 'private' if layer._share_gid is None else 'shared_writer',
                    'matrix': name, 'chunk': chunk, 'component': component,
                    'write_chunk': chunk - 1 if component == 'update' else None,
                    'tokens_written': chunk * layer.lact_chunk_size,
                    'chunk_size': layer.lact_chunk_size,
                    'states': value, '_reference_norms': reference_norms,
                }


@torch.no_grad()
def collect_states(model, mods, ids, chunks, components=('state',)):
    """One sequence -> detached CPU snapshots. No SVD inside model forward."""
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('Probe one [1, sequence_length] sequence at a time')
    if not chunks or min(chunks) < 0:
        raise ValueError('Chunk indices must be nonnegative')
    snapshots = []
    seen = set()

    def capture(layer, trajectory, decay=None):
        seen.add(layer.layer_idx)
        snapshots.extend(trajectory_components(layer, trajectory, chunks, components, decay))

    capture.capture_decay = any(c != 'state' for c in components)

    with observe_states(mods, capture) as owners:
        device = next(model.parameters()).device
        model(input_ids=ids.to(device), use_cache=False)
        missing = {m.layer_idx for m in owners} - seen
        if missing:
            raise RuntimeError(f'Memory observers did not run at layers {sorted(missing)}')
    return snapshots


def rank_records(snapshots, corpus, sequence, svd_device):
    for snapshot in snapshots:
        matrices = snapshot['states']
        if not torch.isfinite(matrices).all():
            raise ValueError(f'Nonfinite states: {corpus}, seq {sequence}, '
                             f'layer {snapshot["layer"]}, {snapshot["matrix"]}')
        singular = torch.linalg.svdvals(matrices.to(svd_device)).cpu()
        for head, values in enumerate(singular):
            metrics = spectrum_stats(values)
            reference = float(snapshot['_reference_norms'][head])
            yield {**{k: v for k, v in snapshot.items() if k not in ('states', '_reference_norms')},
                   'corpus': corpus, 'sequence': sequence, 'head': head,
                   'rows': matrices.shape[-2], 'cols': matrices.shape[-1],
                   'rank_ceiling': min(matrices.shape[-2:]),
                   'state_frobenius_norm': reference,
                   'relative_to_state_norm': metrics['frobenius_norm'] / reference if reference else None,
                   **metrics, 'singular_values': values.tolist()}


def split_tokens(tokens, seq_len, count):
    available = len(tokens) // seq_len
    if available < count:
        raise ValueError(f'Only {available} full sequences available; requested {count}')
    return torch.tensor(tokens[:seq_len * count], dtype=torch.long).reshape(count, seq_len)


def load_corpora(args, tokenizer):
    """Deterministic contiguous WikiText; one prefix per PG19 book where possible."""
    from datasets import load_dataset

    corpora, sources = {}, {}
    if 'wikitext' in args.corpora:
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        ids = tokenizer('\n\n'.join(ds['text']), add_special_tokens=False)['input_ids']
        corpora['wikitext'] = split_tokens(ids, args.seq_len, args.seqs)
        sources['wikitext'] = {'dataset': 'wikitext', 'config': 'wikitext-2-raw-v1',
                               'split': 'validation', 'packing': 'contiguous joined text',
                               'fingerprint': ds._fingerprint}
    if 'pg19' in args.corpora:
        ds = load_dataset(args.pg19_path, split='validation', streaming=True)
        sequences, book_indices = [], []
        # Book-order prefixes are deterministic; skip books shorter than seq_len.
        # This avoids building the training loader or downloading training books.
        for index, book in enumerate(ds):
            ids = tokenizer(book['text'], add_special_tokens=False,
                            truncation=True, max_length=args.seq_len)['input_ids']
            if len(ids) == args.seq_len:
                sequences.append(torch.tensor(ids, dtype=torch.long))
                book_indices.append(index)
            if len(sequences) == args.seqs:
                break
        if len(sequences) < args.seqs:
            raise ValueError(f'Only {len(sequences)} sufficiently long PG19 validation books')
        corpora['pg19'] = torch.stack(sequences)
        sources['pg19'] = {'dataset': args.pg19_path, 'split': 'validation',
                           'packing': 'one prefix per qualifying book', 'book_indices': book_indices}
    return corpora, sources


def validate_tokens(corpora, seq_len, count, vocab_size):
    if not corpora:
        raise ValueError('Token file contains no requested corpora')
    for name, ids in corpora.items():
        if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.long
                or tuple(ids.shape) != (count, seq_len)):
            raise ValueError(f'{name}: expected int64 tokens of shape {(count, seq_len)}')
        if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
            raise ValueError(f'{name}: token IDs outside model vocabulary')


def checkpoint_files(directory):
    """Record file identities without hashing tens of GB of checkpoint weights."""
    if directory is None:
        return None
    root = Path(directory)
    return [{'name': p.name, 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(root.iterdir()) if p.is_file()] if root.is_dir() else []


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['corpus'], row.get('component', 'state'), row['layer'], row['matrix'], row['chunk'])].append(row)
    summary = []
    for (corpus, component, layer, matrix, chunk), group in sorted(grouped.items()):
        item = {'corpus': corpus, 'layer': layer, 'matrix': matrix, 'chunk': chunk,
                'component': component,
                'write_chunk': chunk - 1 if component == 'update' else None,
                'role': group[0]['role'], 'rank_ceiling': group[0]['rank_ceiling'],
                'n_sequences': len({r['sequence'] for r in group}), 'n_observations': len(group)}
        for metric in ('r90', 'r95', 'r99', 'stable_rank', 'frobenius_norm',
                       'state_frobenius_norm', 'relative_to_state_norm'):
            values = torch.tensor([r[metric] for r in group if r.get(metric) is not None],
                                  dtype=torch.float64)
            item[metric] = ({'mean': values.mean().item(), 'median': torch.quantile(values, .5).item(),
                             'p90': torch.quantile(values, .9).item(), 'max': values.max().item()}
                            if values.numel() else None)
        summary.append(item)
    return summary


def plot_heatmaps(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    for corpus, component in sorted({(row['corpus'], row.get('component', 'state')) for row in summary}):
        component_rows = [r for r in summary if r['corpus'] == corpus and r.get('component', 'state') == component]
        for metric in ('r95', 'r99'):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
            ceiling = max(r['rank_ceiling'] for r in component_rows)
            for ax, matrix in zip(axes, ('w0', 'w1', 'w2')):
                cells = [r for r in component_rows if r['matrix'] == matrix]
                layers = sorted({r['layer'] for r in cells})
                chunks = sorted({r['chunk'] for r in cells})
                grid = np.full((len(layers), len(chunks)), np.nan)
                for r in cells:
                    grid[layers.index(r['layer']), chunks.index(r['chunk'])] = r[metric]['mean']
                im = ax.imshow(grid, vmin=0, vmax=ceiling, aspect='auto', cmap='viridis')
                ax.set_xticks(range(len(chunks)), chunks)
                ax.set_yticks(range(len(layers)), [f'L{x}' for x in layers])
                ax.set_xlabel('State index c (after c updates)')
                ax.set_title(matrix)
            fig.colorbar(im, ax=axes, label=f'Mean {metric} across sequences and heads')
            title = {'state': 'full state', 'adaptation': 'accumulated adaptation',
                     'update': 'preceding write U[c-1]'}[component]
            fig.suptitle(f'{corpus}: {title} energy rank')
            stem = f'{corpus}_{metric}' if component == 'state' else f'{corpus}_{component}_{metric}'
            fig.savefig(out / f'{stem}.png', dpi=180)
            fig.savefig(out / f'{stem}.pdf')
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B', help='tokenizer identifier')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--chunks', type=int, nargs='+', default=[0, 1, 2, 4, 8, 15])
    ap.add_argument('--components', nargs='+', choices=['state', 'adaptation', 'update'],
                    default=['state'], help='full state, accumulated writes, or preceding chunk write')
    ap.add_argument('--svd-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--tokens-file', help='reuse tokens.pt from a previous probe')
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--out', required=True, help='new output directory')
    args = ap.parse_args()
    if args.seq_len < 1 or args.seqs < 1 or min(args.chunks) < 0:
        ap.error('Sequence dimensions must be positive; chunk indices nonnegative')
    if not torch.cuda.is_available():
        ap.error('Checkpoint evaluation requires CUDA (same loader as eval.py)')

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from Training.train import build_model_config
    from eval import load_model, ttt_layers
    from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True))
    config.model.pretrained_model_name_or_path = args.ckpt
    config.model.max_length = args.seq_len
    model_config = build_model_config(config)
    model_config.validate_ttt()
    if model_config.ttt_inner_loss != 'l2':
        ap.error('This probe requires l2 TTT memories')
    chunks = sorted(set([0, *args.chunks]))
    available = math.ceil(args.seq_len / model_config.lact_chunk_size)
    skipped = [c for c in chunks if c >= available]
    chunks = [c for c in chunks if c < available]
    if set(args.components) == {'update'} and not any(c > 0 for c in chunks):
        ap.error('No applied updates at the selected states; include a state index greater than 0')
    if skipped:
        print(f'Skipping unavailable chunk indices {skipped}; last read state is {available - 1}')
    if args.tokens_file:
        saved = torch.load(args.tokens_file, map_location='cpu', weights_only=True)
        if saved['tokenizer'] != args.base:
            raise ValueError('Saved token file uses a different tokenizer identifier')
        corpora = {name: saved['corpora'][name] for name in dict.fromkeys(args.corpora)}
        sources = {name: saved['sources'][name] for name in corpora}
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        corpora, sources = load_corpora(args, tokenizer)
    validate_tokens(corpora, args.seq_len, args.seqs, model_config.vocab_size)
    torch.save({'tokenizer': args.base, 'corpora': corpora, 'sources': sources}, out / 'tokens.pt')

    identity = {name: checkpoint_files(path) for name, path in
                (('ckpt', args.ckpt), ('adapter', args.adapter))}
    metadata = {'arguments': vars(args), 'chunks': chunks, 'sources': sources,
                'model_config': model_config.to_dict(), 'checkpoint_files': identity,
                'complete': False,
                'torch_version': str(torch.__version__),
                'interpretation': 'Per-head state components; no compression applied',
                'component_definitions': {
                    'state': 'W[c], the weights used to read chunk c',
                    'adaptation': 'W[c] - prod(alpha[:c]) * W[0]',
                    'update': 'W[c] - alpha[c-1] * W[c-1], for c >= 1'},
                'subtraction': 'float64 from actual finite-precision states and operator decay; SVD float32',
                'token_hashes': {name: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                                 for name, ids in corpora.items()}}
    source_root = Path(__file__).resolve().parent
    metadata['code_hashes'] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ('diag_rank.py', 'LinearTTT/model/LinearizeLlama/LinearizeLlama.py',
                     'LinearTTT/model/LinearizeLlama/ttt_l2.py')}
    model = load_model(args.ckpt, model_config, args.adapter)
    model.requires_grad_(False)
    use_sdpa_sliding_window(True)
    mods = ttt_layers(model)
    owners = memory_owners(mods)
    metadata['memory_layers'] = [m.layer_idx for m in owners]
    print(f'Independent memories: {metadata["memory_layers"]}; chunks: {chunks}')
    # The language-model head is unnecessary for state spectra and allocates
    # [batch, length, vocabulary] logits. The backbone executes all memory layers.
    backbone = model.model
    if not hasattr(backbone, 'layers'):
        raise ValueError('Expected the LigerGLA decoder backbone at model.model')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    with (out / 'spectra.jsonl').open('w') as spectra, (out / 'ranks.csv').open('w', newline='') as csvfile:
        writer = None
        for corpus, sequences in corpora.items():
            for sequence, ids in enumerate(sequences):
                print(f'{corpus}: sequence {sequence + 1}/{len(sequences)} — collecting states', flush=True)
                snapshots = collect_states(backbone, mods, ids.unsqueeze(0), chunks, args.components)
                for record in rank_records(snapshots, corpus, sequence, args.svd_device):
                    spectra.write(json.dumps(record, allow_nan=False) + '\n')
                    row = {k: v for k, v in record.items() if k != 'singular_values'}
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    rows.append(row)
                spectra.flush()
                csvfile.flush()
                del snapshots
                print(f'{corpus}: sequence {sequence + 1} complete', flush=True)
    summary = summarize(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if not args.no_plots:
        plot_heatmaps(summary, out)
    current = {name: checkpoint_files(path) for name, path in
               (('ckpt', args.ckpt), ('adapter', args.adapter))}
    if current != identity:
        raise RuntimeError('Checkpoint files changed during the run; use fixed checkpoints and rerun')
    metadata['complete'] = True
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote {len(rows)} per-head measurements to {out}')


if __name__ == '__main__':
    main()
