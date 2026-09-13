"""Compare TTT readouts across layers on aligned query positions.

Reports debiased linear CKA, a shuffled-token control, and signed cosine.
History effect holds each observed query/gate fixed and subtracts the initial
memory's readout. This is an observational analysis, not proof of redundancy.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def query_positions(target_mask, chunk_size, count, seed):
    """Masks index targets; layer activations index their prediction queries."""
    positions = torch.where(target_mask.cpu())[0] - 1
    positions = positions[positions >= chunk_size]
    if len(positions) > count:
        generator = torch.Generator().manual_seed(seed)
        positions = positions[torch.randperm(len(positions), generator=generator)[:count]]
    return positions.sort().values


def initial_readout(layer, projected_queries, output_dtype):
    """The initial SwiGLU on these exact observed queries; no state updates."""
    queries, _, _ = layer._ttt_features(projected_queries, projected_queries, projected_queries)
    qi = queries.transpose(1, 2)
    w0, w1, w2 = (getattr(layer, name).float() for name in ('w0', 'w1', 'w2'))
    with torch.autocast(device_type=qi.device.type, enabled=qi.is_cuda, dtype=torch.bfloat16):
        output = torch.bmm(w1, F.silu(torch.bmm(w0, qi)) * torch.bmm(w2, qi)).transpose(1, 2)
    return layer.ttt_norm(output.to(output_dtype))


@torch.no_grad()
def collect_readouts(model, mods, ids, positions):
    """Read-only hooks. Returns layer -> component -> [sampled queries, hidden]."""
    if ids.ndim != 2 or ids.shape[0] != 1 or positions.ndim != 1 or not len(positions):
        raise ValueError('Expected one sequence and a nonempty query-position vector')
    if positions.min() < 0 or positions.max() >= ids.shape[1]:
        raise ValueError('Query positions outside the sequence')
    if any(m.training or m.ttt_inner_loss != 'l2' or m._ablate_ttt or m._ablate_attn for m in mods):
        raise ValueError('Use an intact l2 model in eval mode')
    hooks, pending, result = [], {}, {}
    reference_active = False

    def save(layer_id, kind):
        def hook(module, inputs, output):
            if not reference_active:
                pending.setdefault(layer_id, {})[kind] = output.detach().index_select(
                    1, positions.to(output.device))
        return hook

    def finish(layer, inputs, output):
        nonlocal reference_active
        values = pending.pop(layer.layer_idx)
        normalized = values['normalized']  # [heads, sampled tokens, head dim]
        gate = F.silu(values['gate']).transpose(1, 2).reshape(layer.num_ttt_heads, -1, 1)
        reference_active = True
        try:
            initial = initial_readout(layer, values['query'], normalized.dtype)
        finally:
            reference_active = False

        def contribution(readout):
            gated = readout * gate.to(readout.dtype)
            flattened = gated.transpose(0, 1).reshape(1, len(positions), layer.inner_dim)
            # o_proj is bias-free in this model; the subtraction also supports
            # a bias without counting it as a TTT contribution.
            projected = layer.o_proj(flattened.to(layer.o_proj.weight.dtype))
            if layer.o_proj.bias is not None:
                projected = projected - layer.o_proj.bias
            return projected[0].float()

        current, initial = contribution(normalized), contribution(initial)
        result[layer.layer_idx] = {
            'current': current.cpu(), 'initial': initial.cpu(),
            'history_effect': (current - initial).cpu()}

    try:
        for layer in mods:
            hooks.extend([
                layer.q_proj.register_forward_hook(save(layer.layer_idx, 'query')),
                layer.ttt_norm.register_forward_hook(save(layer.layer_idx, 'normalized')),
                layer.ttt_scale_proj.register_forward_hook(save(layer.layer_idx, 'gate')),
                layer.register_forward_hook(finish)])
        model(input_ids=ids.to(next(model.parameters()).device), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    missing = {m.layer_idx for m in mods} - set(result)
    if missing:
        raise RuntimeError(f'Readout hooks did not run at layers {sorted(missing)}')
    return result


def center_gram(gram, debiased=False):
    """H-centering or U-centering for debiased linear CKA (n > 3)."""
    if gram.ndim != 3 or gram.shape[-1] != gram.shape[-2] or gram.shape[-1] < 4:
        raise ValueError('Expected [layers, n, n] Gram matrices with n >= 4')
    if debiased:
        n = gram.shape[-1]
        gram = gram.clone()
        gram.diagonal(dim1=-2, dim2=-1).zero_()
        rows = gram.sum(-1, keepdim=True)
        centered = gram - rows / (n - 2) - rows.transpose(-1, -2) / (n - 2)
        centered = centered + rows.sum(-2, keepdim=True) / ((n - 1) * (n - 2))
        centered.diagonal(dim1=-2, dim2=-1).zero_()
        return centered
    return gram - gram.mean(-1, keepdim=True) - gram.mean(-2, keepdim=True) + gram.mean((-1, -2), keepdim=True)


def normalize_grams(grams):
    flat = grams.flatten(1)
    norms = flat.norm(dim=1)
    normalized = flat / norms.clamp_min(torch.finfo(flat.dtype).tiny)[:, None]
    return normalized, norms > 0


@torch.no_grad()
def pairwise_similarity(features, device='cpu', seed=0, shuffles=5):
    """Features [layers, aligned tokens, dimensions]; no feature projection."""
    if features.ndim != 3 or features.shape[1] < 4 or shuffles < 1:
        raise ValueError('Need at least four aligned samples and one shuffle')
    x = features.to(device=device, dtype=torch.float64)
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite readout features')
    # Per-layer translation leaves both CKA estimators unchanged and improves
    # numerical stability for a large common mean. Cosine uses uncentered x.
    centered_x = x - x.mean(1, keepdim=True)
    gram = centered_x @ centered_x.transpose(-1, -2)
    standard, valid_standard = normalize_grams(center_gram(gram))
    debiased_gram = center_gram(gram, debiased=True)
    debiased, valid = normalize_grams(debiased_gram)
    cka, debiased_cka = standard @ standard.T, debiased @ debiased.T
    null = torch.zeros_like(debiased_cka)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(shuffles):
        order = torch.randperm(x.shape[1], generator=generator).to(device)
        permuted = debiased_gram[:, order][:, :, order]
        shuffled, _ = normalize_grams(permuted)
        cross = debiased @ shuffled.T
        null += (cross + cross.T) / (2 * shuffles)
    lengths = x.norm(dim=-1)
    units = x / lengths.clamp_min(torch.finfo(x.dtype).tiny).unsqueeze(-1)
    valid_tokens = lengths > 0
    counts = valid_tokens.double() @ valid_tokens.double().T
    cosine = (units.flatten(1) @ units.flatten(1).T) / counts.clamp_min(1)
    result = {'linear_cka': cka, 'debiased_cka': debiased_cka,
              'shuffled_debiased_cka': null, 'mean_cosine': cosine}
    for key, value in result.items():
        if key == 'mean_cosine':
            good = counts > 0
        else:
            validity = valid_standard if key == 'linear_cka' else valid
            good = validity[:, None] & validity[None, :]
        result[key] = value.masked_fill(~good, float('nan')).cpu()
    return result


def aggregate_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('corpus', 'slice', 'component', 'layer_i', 'layer_j'))
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items()):
        item = dict(zip(('corpus', 'slice', 'component', 'layer_i', 'layer_j'), key))
        item['n_sequences'] = len(group)
        for metric in ('linear_cka', 'debiased_cka', 'shuffled_debiased_cka', 'mean_cosine'):
            values = torch.tensor([r[metric] for r in group if r[metric] is not None], dtype=torch.float64)
            item[metric] = {'mean': values.mean().item() if len(values) else None,
                            'n_valid': len(values),
                            'sequence_sd': values.std().item() if len(values) > 1 else None}
        result.append(item)
    return result


def plot_similarity(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    for corpus, slice_name in sorted({(r['corpus'], r['slice']) for r in summary}):
        for component in ('current', 'initial', 'history_effect'):
            rows = [r for r in summary if r['corpus'] == corpus and r['slice'] == slice_name
                    and r['component'] == component]
            if not rows:
                continue
            layers = sorted({r['layer_i'] for r in rows} | {r['layer_j'] for r in rows})
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for ax, metric in zip(axes, ('debiased_cka', 'shuffled_debiased_cka', 'mean_cosine')):
                values = np.full((len(layers), len(layers)), np.nan)
                for row in rows:
                    i, j = layers.index(row['layer_i']), layers.index(row['layer_j'])
                    values[i, j] = values[j, i] = row[metric]['mean'] if row[metric]['mean'] is not None else np.nan
                im = ax.imshow(values, vmin=-1, vmax=1, cmap='coolwarm')
                ticks = range(0, len(layers), max(1, len(layers) // 8))
                ax.set_xticks(list(ticks), [layers[i] for i in ticks])
                ax.set_yticks(list(ticks), [layers[i] for i in ticks])
                ax.set_xlabel('Layer'); ax.set_ylabel('Layer'); ax.set_title(metric.replace('_', ' '))
            fig.colorbar(im, ax=axes, label='Mean across sequences')
            fig.suptitle(f'{corpus} / {slice_name} / {component}')
            stem = out / f'{corpus}_{slice_name}_{component}'
            fig.savefig(str(stem) + '.png', dpi=180)
            fig.savefig(str(stem) + '.pdf')
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter'); ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--tokens-file', help='reuse exact corpus tokens from previous probes')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192); ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--samples', type=int, default=128, help='max query positions per slice per sequence')
    ap.add_argument('--slices', nargs='+', choices=['cross_window', 'other'], default=['cross_window', 'other'])
    ap.add_argument('--layers', nargs='+', type=int, help='default: all TTT layers, including readers')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--shuffles', type=int, default=5)
    ap.add_argument('--analysis-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--save-features', action='store_true'); ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.samples < 4 or args.seqs < 1 or args.seq_len < 5 or args.shuffles < 1:
        ap.error('Need samples >= 4, positive sequence count, length >= 5, and shuffles >= 1')
    if not torch.cuda.is_available():
        ap.error('The checkpoint loader requires CUDA')
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from Training.train import build_model_config
    from eval import ar_masks, load_model, ttt_layers
    from diag_rank import checkpoint_files, load_corpora, validate_tokens
    from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window

    config = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True))
    config.model.pretrained_model_name_or_path = args.ckpt
    config.model.max_length = args.seq_len
    model_config = build_model_config(config)
    if model_config.ttt_inner_loss != 'l2':
        ap.error('Use an l2 checkpoint')
    if args.tokens_file:
        saved = torch.load(args.tokens_file, map_location='cpu', weights_only=True)
        if saved['tokenizer'] != args.base:
            raise ValueError('Token-file tokenizer does not match --base')
        corpora = {name: saved['corpora'][name] for name in dict.fromkeys(args.corpora)}
        sources = {name: saved['sources'][name] for name in corpora}
    else:
        corpora, sources = load_corpora(args, AutoTokenizer.from_pretrained(args.base))
    validate_tokens(corpora, args.seq_len, args.seqs, model_config.vocab_size)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.save({'tokenizer': args.base, 'corpora': corpora, 'sources': sources}, out / 'tokens.pt')
    identity = {name: checkpoint_files(path) for name, path in (('ckpt', args.ckpt), ('adapter', args.adapter))}
    metadata = {'complete': False, 'arguments': vars(args), 'sources': sources,
                'model_config': model_config.to_dict(), 'checkpoint_files': identity,
                'token_hashes': {name: hashlib.sha256(ids.numpy().tobytes()).hexdigest() for name, ids in corpora.items()},
                'definitions': {'current': 'actual gated/projected TTT readout',
                    'initial': 'learned initial weights read with the SAME observed queries/gates',
                    'history_effect': 'current - initial; conditional readout effect, not a whole-model intervention',
                    'cross_window': 'repeated-bigram targets with source beyond the local window',
                    'other': 'non-repeated-bigram targets after the first chunk'},
                'similarity_reference': 'https://proceedings.mlr.press/v97/kornblith19a.html',
                'limitations': 'Similarity is not mutual information, state identity, or proof of causal redundancy.'}
    source_root = Path(__file__).resolve().parent
    metadata['code_hashes'] = {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                             for name in ('diag_similarity.py', 'diag_rank.py', 'eval.py',
                                          'LinearTTT/model/LinearizeLlama/LinearizeLlama.py')}
    model = load_model(args.ckpt, model_config, args.adapter)
    model.requires_grad_(False); use_sdpa_sliding_window(True)
    mods = ttt_layers(model)
    if args.layers is not None:
        wanted = set(args.layers)
        if wanted - {m.layer_idx for m in mods}:
            raise ValueError('Requested layer is not present in model')
        mods = [m for m in mods if m.layer_idx in wanted]
    layers = [m.layer_idx for m in mods]
    metadata['layers'] = layers
    metadata['shared_groups'] = model_config.ttt_share_groups
    metadata['torch_version'] = str(torch.__version__)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, selections, magnitudes = [], [], []
    for corpus, sequences in corpora.items():
        for sequence, ids in enumerate(sequences):
            masks = ar_masks(ids, [model_config.window_size, float('inf')], model_config.lact_chunk_size)
            names = {'cross_window': f'ar_x{model_config.window_size}+', 'other': 'other'}
            chosen = {name: query_positions(masks[names[name]], model_config.lact_chunk_size,
                                            args.samples, args.seed + sequence) for name in dict.fromkeys(args.slices)}
            selections.append({'corpus': corpus, 'sequence': sequence,
                               'positions': {k: v.tolist() for k, v in chosen.items()}})
            chosen = {k: v for k, v in chosen.items() if len(v) >= 4}
            if not chosen:
                print(f'{corpus} {sequence + 1}: no slice with four queries, skipping', flush=True)
                continue
            positions = torch.cat(list(chosen.values())).unique(sorted=True)
            print(f'{corpus} {sequence + 1}/{len(sequences)}: '
                  + ', '.join(f'{k}={len(v)}' for k, v in chosen.items()), flush=True)
            features = collect_readouts(model.model, mods, ids.unsqueeze(0), positions)
            if args.save_features:
                torch.save({'positions': positions, 'features': features}, out / f'{corpus}_{sequence}_features.pt')
            for name, selected in chosen.items():
                indices = torch.searchsorted(positions, selected)
                for component in ('current', 'initial', 'history_effect'):
                    x = torch.stack([features[layer][component][indices] for layer in layers])
                    measurements = pairwise_similarity(x, args.analysis_device, args.seed + sequence, args.shuffles)
                    for i, layer in enumerate(layers):
                        current_norm = features[layer]['current'][indices].double().norm()
                        magnitudes.append({'corpus': corpus, 'sequence': sequence, 'slice': name,
                            'component': component, 'layer': layer,
                            'frobenius_norm': x[i].double().norm().item(),
                            'relative_to_current_norm': (x[i].double().norm() / current_norm).item() if current_norm else None})
                        for j in range(i, len(layers)):
                            row = {'corpus': corpus, 'sequence': sequence, 'slice': name,
                                   'component': component, 'layer_i': layer, 'layer_j': layers[j],
                                   'n_queries': len(selected)}
                            for metric, values in measurements.items():
                                row[metric] = float(values[i, j]) if torch.isfinite(values[i, j]) else None
                            rows.append(row)
                    del x
            del features
            print(f'{corpus} {sequence + 1}: complete', flush=True)
    if not rows:
        raise RuntimeError('No usable similarity measurements')
    with (out / 'pairs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = aggregate_pairs(rows)
    for filename, payload in (('summary.json', summary), ('selections.json', selections), ('magnitudes.json', magnitudes)):
        (out / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    if not args.no_plots:
        plot_similarity(summary, out)
    if identity != {name: checkpoint_files(path) for name, path in (('ckpt', args.ckpt), ('adapter', args.adapter))}:
        raise RuntimeError('Checkpoint changed during the probe; use fixed checkpoints')
    metadata['complete'] = True
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote similarity results to {out}')


if __name__ == '__main__':
    main()
