"""Capture real TTT reader inputs/outputs on held-out prompt examples.

For each selected reader, save actual pre-alignment readouts x from held-out
prompts, the polar-stretch transform P x, and the module's actual output chi x.
The companion plot uses the same saved rotation plane and reports activation
energy captured by that plane.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from Training.train import build_model_config
from eval import load_model, ttt_layers
from analyze_reader_maps import polar_factors
from plot_reader_rotation import angles, load as load_reader_maps, select_example
from measure_flops import set_window_backend


def parse_layers(values):
    layers = {}
    for value in values:
        layer, head = value.split(':', 1)
        layers[int(layer)] = int(head)
    if len(layers) != len(values):
        raise ValueError('Specify each layer at most once')
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg', required=True, help='matching model/config YAML')
    ap.add_argument('--ckpt', required=True, help='base pretrained model checkpoint')
    ap.add_argument('--adapter', required=True, help='trained checkpoint dir with adapters and ttt_params.pt')
    ap.add_argument('--base', help='tokenizer id/path; defaults to --ckpt')
    ap.add_argument('--prompts-jsonl',
                    help='optional held-out JSONL; each row may contain prompt, text, or instruction/input/output; '
                         'by default use the configured validation loader')
    ap.add_argument('--out', required=True, help='output .npz path; writes adjacent .json metadata')
    ap.add_argument('--layer-head', nargs='+', default=None, metavar='LAYER:HEAD',
                    help='select layer/head pairs, e.g. 1:12 29:3 30:4 31:9; default picks median head by rotation')
    ap.add_argument('--max-examples', type=int, default=32)
    ap.add_argument('--positions-per-prompt', type=int, default=32)
    ap.add_argument('--max-length', type=int, default=8192)
    args = ap.parse_args()
    if args.max_examples <= 0 or args.positions_per_prompt <= 0:
        ap.error('--max-examples and --positions-per-prompt must be positive')

    config = OmegaConf.load(args.cfg)
    maps = load_reader_maps(args.adapter)
    selections = parse_layers(args.layer_head or [])
    metadata = {}
    bases, aligneds, readouts, positions, position_fractions, example_ids = {}, {}, {}, {}, {}, {}
    for layer, chi in maps.items():
        if selections and layer not in selections:
            continue
        # Every head, not one. A single median-rotation head per layer made the
        # layers incomparable (L1 head 25 against L29 head 23) and left a lone
        # low number with no way to tell an unlucky head from a quiet reader.
        per_layer = {layer: dict(chi=chi, ang=angles(chi))}
        chosen = select_example(per_layer, layer=layer, head=selections.get(layer))
        from plot_reader_rotation import strongest_plane
        heads, dim = chi.shape[0], chi.shape[-1]
        basis_stack = torch.zeros(heads, dim, 2, dtype=torch.float64)
        head_info = []
        for h in range(heads):
            q_head, _, _ = polar_factors(chi[h])
            # A reflection has no proper rotation plane. The turn between a
            # readout and its image needs no plane, so such a head still
            # reports a turn; only its plane energy is undefined.
            try:
                basis_h, q2 = strongest_plane(q_head)
            except ValueError:
                head_info.append(dict(head=h, has_plane=False,
                                      rotation_angle_degrees=None))
                continue
            basis_stack[h] = basis_h
            head_info.append(dict(head=h, has_plane=True,
                rotation_angle_degrees=float(torch.atan2(q2[1, 0], q2[0, 0]).rad2deg())))
        if not any(info['has_plane'] for info in head_info):
            raise ValueError(f'Layer {layer}: no head has a proper rotation plane')
        bases[layer] = basis_stack
        metadata[str(layer)] = dict(display_head=chosen['head'], heads=heads,
            heads_with_plane=sum(info['has_plane'] for info in head_info),
            per_head=head_info,
            display_rotation_angle_degrees=chosen['angle_degrees'],
            display_stretch_energy_outside_plane=chosen['stretch_out_of_plane_energy_fraction'])
        readouts[layer], aligneds[layer], positions[layer] = [], [], []
        position_fractions[layer], example_ids[layer] = [], []

    if not bases:
        raise ValueError('No reader maps selected')
    if selections and set(selections) != set(bases):
        raise ValueError(f'Requested layers without saved readers: {sorted(set(selections)-set(bases))}')

    torch.manual_seed(0)
    # Match eval.py's supported sliding-window attention path. This avoids
    # recompiling flex attention for the variable prompt lengths in the holdout.
    set_window_backend(True)
    config.model.pretrained_model_name_or_path = args.ckpt
    config.model.max_length = max(int(config.model.max_length), args.max_length)
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    model = load_model(args.ckpt, build_model_config(config), args.adapter,
                       strict_ttt=True)
    mods = ttt_layers(model)
    if not torch.cuda.is_available():
        raise RuntimeError('Activation capture expects a GPU model')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base or args.ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = model.device

    hooks = []
    active = {}
    for layer in bases:
        if layer >= len(mods):
            raise ValueError(f'Layer {layer} outside model with {len(mods)} TTT layers')
        active[layer] = dict(seen=0)
        def capture(module, inputs, output, layer=layer):
            x = inputs[0].detach()
            y = output.detach()
            heads = module.weight.shape[0]
            if x.shape[0] % heads:
                raise ValueError('Reader activation batch dimension is not divisible by head count')
            x = x.reshape(-1, heads, x.shape[1], x.shape[2])[0]      # [heads, tokens, dim]
            y = y.reshape(-1, heads, y.shape[1], y.shape[2])[0]
            n = x.shape[1]
            take = min(args.positions_per_prompt, n)
            idx = torch.linspace(0, n - 1, take, device=x.device).round().long().unique()
            readouts[layer].append(x[:, idx].permute(1, 0, 2).float().cpu())
            aligneds[layer].append(y[:, idx].permute(1, 0, 2).float().cpu())
            positions[layer].append(idx.cpu())
            position_fractions[layer].append((idx.float() / max(1, n - 1)).cpu())
            active[layer]['seen'] += 1
        hooks.append(mods[layer].ttt_reader_alignment.register_forward_hook(capture))

    try:
        if args.prompts_jsonl:
            with Path(args.prompts_jsonl).open() as source:
                rows = [json.loads(line) for line in source]
            def row_text(row):
                for key in ('prompt', 'text'):
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
                fields = [row.get(key) for key in ('instruction', 'input', 'output')
                          if isinstance(row.get(key), str) and row[key].strip()]
                return '\n\n'.join(fields) if fields else None
            rows = [(row, row_text(row)) for row in rows]
            rows = [(row, prompt) for row, prompt in rows if prompt]
            if not rows:
                raise ValueError('No non-empty prompt rows found')
            # Spread examples across the holdout instead of taking its first rows.
            indices = np.linspace(0, len(rows) - 1,
                                  min(args.max_examples, len(rows))).round().astype(int)
            examples = [('jsonl', rows[i][0], rows[i][1]) for i in indices]
        elif 'longalpaca' in str(config.data.name).lower().replace('-', '').replace('_', ''):
            # Mirror the validation subset and ConcatDataset packing used by
            # Training.dataloader without materializing/tokenizing the 12k-row
            # training split just to inspect its held-out windows.
            from datasets import load_dataset
            source = load_dataset(str(config.data.path), split='train', streaming=True)
            chunks, buffer = [], []
            min_chars = int(config.data.get('min_doc_chars', 0))
            target = int(config.data.get('num_val_docs', 200))
            accepted = 0
            for row in source:
                doc = '\n\n'.join(str(row[k]) for k in ('instruction', 'input', 'output')
                                   if row.get(k)) or str(row.get('text', ''))
                if len(doc) < min_chars:
                    continue
                ids = tokenizer.encode(doc, add_special_tokens=False)
                if tokenizer.bos_token_id is not None:
                    ids = [tokenizer.bos_token_id] + ids
                if tokenizer.eos_token_id is not None:
                    ids.append(tokenizer.eos_token_id)
                buffer.extend(ids)
                while len(buffer) > args.max_length:
                    chunks.append(buffer[:args.max_length])
                    buffer = buffer[args.max_length:]
                accepted += 1
                if accepted >= target:
                    break
            if not chunks:
                raise ValueError('Configured LongAlpaca holdout produced no full validation windows')
            indices = np.linspace(0, len(chunks) - 1,
                                  min(args.max_examples, len(chunks))).round().astype(int)
            examples = [('batch', {
                'input_ids': torch.tensor([chunks[i]], dtype=torch.long),
                'attention_mask': torch.ones((1, len(chunks[i])), dtype=torch.long)}, None)
                for i in indices]
        else:
            raise ValueError('Pass --prompts-jsonl for this data config; automatic holdout loading '
                             'currently mirrors the configured LongAlpaca validation windows')
        with torch.inference_mode():
            count = 0
            for kind, row, prompt in examples:
                if kind == 'batch':
                    input_ids = row['input_ids']
                    attention_mask = row.get('attention_mask', torch.ones_like(input_ids))
                else:
                    encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
                    input_ids = encoded.input_ids
                    if tokenizer.eos_token_id is not None and input_ids[0, -1].item() != tokenizer.eos_token_id:
                        input_ids = torch.cat((input_ids, input_ids.new_tensor([[tokenizer.eos_token_id]])), dim=1)
                    attention_mask = torch.ones_like(input_ids)
                if input_ids.shape[1] > args.max_length:
                    raise ValueError(f'Prompt {row.get("id", count)} has {input_ids.shape[1]} tokens; '
                                     f'exceeds max-length={args.max_length}; refusing truncation')
                model(input_ids=input_ids.to(device),
                      attention_mask=attention_mask.to(device), use_cache=False)
                for layer in bases:
                    if active[layer]['seen'] != count + 1:
                        raise RuntimeError(
                            f'Layer {layer} reader fired {active[layer]["seen"]} times over '
                            f'{count + 1} sequences; example ids would misalign with readouts')
                for layer in bases:
                    example_id = (str(row.get('id', count)) if kind == 'jsonl' else f'validation_window_{indices[count]}')
                    example_ids[layer].extend([example_id] * len(positions[layer][-1]))
                count += 1
        if count != len(examples):
            raise RuntimeError(f'Expected {len(examples)} captured sequences, saw {count}')
    finally:
        for hook in hooks:
            hook.remove()

    arrays = {}
    for layer in bases:
        x = torch.cat(readouts[layer]).double()               # [samples, heads, dim]
        y_actual = torch.cat(aligneds[layer]).double()
        chi = maps[layer].double()                            # [heads, dim, dim]
        basis = bases[layer]                                  # [heads, dim, 2]
        stretch = torch.stack([polar_factors(chi[h])[1] for h in range(chi.shape[0])])
        y_stretch = torch.einsum('nhd,hed->nhe', x, stretch)
        # The module's own output is the definitive activation. Checking it
        # against the factorisation catches a convention slip in one number.
        expected = torch.einsum('nhd,hed->nhe', x, chi)
        error = float((y_actual - expected).norm() / expected.norm().clamp(min=1e-30))
        energy = x.square().sum(-1).clamp(min=1e-30)                       # [samples, heads]
        in_plane = torch.einsum('nhd,hdk->nhk', x, basis).square().sum(-1) / energy
        cos = ((x * y_actual).sum(-1)
               / (x.norm(dim=-1) * y_actual.norm(dim=-1)).clamp(min=1e-30))
        turn = cos.clamp(-1, 1).arccos().rad2deg()                         # [samples, heads]
        for info in metadata[str(layer)]['per_head']:
            h = info['head']
            info['median_turn_degrees'] = float(turn[:, h].median())
            info['median_energy_in_plane'] = (
                float(in_plane[:, h].median()) if info['has_plane'] else None)
        planar = [i['median_energy_in_plane'] for i in metadata[str(layer)]['per_head']
                  if i['has_plane']]
        metadata[str(layer)].update(
            samples=int(x.shape[0]), prompts=count,
            actual_reader_relative_reconstruction_error=error,
            median_turn_over_heads=float(np.median(
                [i['median_turn_degrees'] for i in metadata[str(layer)]['per_head']])),
            median_energy_in_plane_over_heads=float(np.median(planar)))
        arrays[f'readout_{layer}'] = x.float().numpy()
        arrays[f'stretched_{layer}'] = y_stretch.float().numpy()
        arrays[f'aligned_{layer}'] = y_actual.float().numpy()
        arrays[f'basis_{layer}'] = basis.float().numpy()
        arrays[f'token_position_{layer}'] = torch.cat(positions[layer]).numpy()
        arrays[f'position_fraction_{layer}'] = torch.cat(position_fractions[layer]).numpy()
        arrays[f'example_id_{layer}'] = np.asarray(example_ids[layer], dtype=str)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    manifest = dict(checkpoint=args.adapter, base=args.ckpt, cfg=args.cfg,
        prompts_jsonl=args.prompts_jsonl, validation_source=(
            'configured validation loader' if not args.prompts_jsonl else args.prompts_jsonl),
        max_examples=count,
        positions_per_prompt=args.positions_per_prompt, layers=metadata)
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for layer, info in metadata.items():
        print(f"layer {layer}: {info['samples']} readouts x {info['heads']} heads "
              f"({info['heads_with_plane']} with a rotation plane); "
              f"median turn {info['median_turn_over_heads']:.1f} deg; "
              f"median plane energy {info['median_energy_in_plane_over_heads']:.1%}; "
              f"reconstruction error {info['actual_reader_relative_reconstruction_error']:.2e}")
    print(f'Wrote actual activations to {output} and {output.with_suffix(".json")}')


if __name__ == '__main__':
    main()
