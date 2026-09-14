"""Paired baseline versus fixed-identity reader evaluation on WikiText and PG19.

No training. Reports token-level perplexity, not lm-eval word perplexity.
The baseline bypasses the maps; the identity arm performs the matrix multiply.
Both arms use the same loaded model, stage-2 weights, and sequence tokens.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def load_corpora(args, tokenizer):
    """Load validation text directly; keep identical tokens for both arms."""
    from datasets import load_dataset
    corpora, sources = {}, {}
    for name in dict.fromkeys(args.corpora):
        if name == 'wikitext':
            data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            ids = tokenizer('\n\n'.join(data['text']), add_special_tokens=False)['input_ids']
            required = args.seqs * args.seq_len
            if len(ids) < required:
                raise ValueError(f'WikiText has only {len(ids)//args.seq_len} full sequences')
            corpora[name] = torch.tensor(ids[:required], dtype=torch.long).reshape(args.seqs,args.seq_len)
            sources[name] = {'dataset':'wikitext', 'config':'wikitext-2-raw-v1',
                             'split':'validation', 'packing':'contiguous joined text',
                             'fingerprint':data._fingerprint}
        else:
            data = load_dataset(args.pg19_path, split='validation', streaming=True)
            sequences, books = [], []
            for index,book in enumerate(data):
                ids = tokenizer(book['text'], add_special_tokens=False,
                                truncation=True, max_length=args.seq_len)['input_ids']
                if len(ids) == args.seq_len:
                    sequences.append(torch.tensor(ids, dtype=torch.long))
                    books.append(index)
                if len(sequences) == args.seqs:
                    break
            if len(sequences) != args.seqs:
                raise ValueError(f'PG19 has only {len(sequences)} sufficiently long validation books')
            corpora[name] = torch.stack(sequences)
            sources[name] = {'dataset':args.pg19_path, 'split':'validation',
                             'packing':'one prefix per qualifying book', 'book_indices':books}
    return corpora, sources


def checkpoint_files(directory):
    if directory is None:
        return None
    root = Path(directory)
    return [{'name':p.name, 'size':p.stat().st_size, 'mtime_ns':p.stat().st_mtime_ns}
            for p in sorted(root.iterdir()) if p.is_file()]


def identity_readers(model):
    from eval import ttt_layers
    readers = [m for m in ttt_layers(model) if m.ttt_reader_alignment is not None]
    if not readers:
        raise ValueError('No reader alignment maps found')
    for reader in readers:
        if reader._share_gid is None or reader._share_leader:
            raise ValueError('Alignment map is on a writer or private layer')
        weight = reader.ttt_reader_alignment.weight
        identity = torch.eye(weight.shape[-1], device=weight.device, dtype=weight.dtype)
        if not torch.equal(weight, identity.expand_as(weight)):
            raise ValueError(f'Layer {reader.layer_idx}: map is not identity; use the original unified checkpoints')
    return readers


@torch.no_grad()
def paired_sequence(model, ids, readers, logit_block=128, atol=1e-3):
    """Score all T-1 targets; compute vocabulary logits in bounded blocks."""
    if model.training or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
        raise ValueError('Use an eval model and one sequence of at least two tokens')
    if logit_block < 1 or not math.isfinite(atol) or atol < 0:
        raise ValueError('Require a positive logit block and finite nonnegative tolerance')
    if not readers:
        raise ValueError('Need reader maps for a paired evaluation')
    ids = ids.to(next(model.parameters()).device)
    previous = [m._identity_reader_alignment for m in readers]
    try:
        for reader in readers:
            reader._identity_reader_alignment = True
        baseline = model.model(input_ids=ids, use_cache=False).last_hidden_state
        for reader in readers:
            reader._identity_reader_alignment = False
        identity = model.model(input_ids=ids, use_cache=False).last_hidden_state
    finally:
        for reader, flag in zip(readers, previous):
            reader._identity_reader_alignment = flag
    if not torch.isfinite(baseline).all() or not torch.isfinite(identity).all():
        raise RuntimeError('Nonfinite decoder output')
    nll_base, nll_identity = 0., 0.
    logit_sum, logit_count, max_logit, max_nll, disagreements = 0., 0, 0., 0., 0
    head = model.get_output_embeddings()
    for start in range(0, ids.shape[1]-1, logit_block):
        stop = min(start+logit_block, ids.shape[1]-1)
        a = head(baseline[:,start:stop].to(head.weight.device, head.weight.dtype)).float()[0]
        b = head(identity[:,start:stop].to(head.weight.device, head.weight.dtype)).float()[0]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('Nonfinite logits')
        labels = ids[0,start+1:stop+1].to(a.device)
        loss_a = F.cross_entropy(a, labels, reduction='none')
        loss_b = F.cross_entropy(b, labels, reduction='none')
        nll_base += loss_a.double().sum().item()
        nll_identity += loss_b.double().sum().item()
        diff = (a-b).abs()
        max_logit = max(max_logit, diff.max().item())
        max_nll = max(max_nll, (loss_a-loss_b).abs().max().item())
        logit_sum += diff.sum(dtype=torch.float64).item()
        logit_count += diff.numel()
        disagreements += int((a.argmax(-1)!=b.argmax(-1)).sum())
    count = ids.shape[1]-1
    return {'n_targets':count, 'baseline_nll_sum':nll_base, 'identity_nll_sum':nll_identity,
            'baseline_ce':nll_base/count, 'identity_ce':nll_identity/count,
            'baseline_token_ppl':math.exp(nll_base/count),
            'identity_token_ppl':math.exp(nll_identity/count),
            'delta_ce':(nll_identity-nll_base)/count,
            'max_abs_logit_delta':max_logit, 'mean_abs_logit_delta':logit_sum/logit_count,
            'max_abs_token_nll_delta':max_nll, 'argmax_disagreements':disagreements,
            'within_tolerance':max_logit<=atol}


def aggregate(rows):
    result = []
    for corpus in sorted({r['corpus'] for r in rows}):
        selected = [r for r in rows if r['corpus']==corpus]
        count = sum(r['n_targets'] for r in selected)
        a = sum(r['baseline_nll_sum'] for r in selected)/count
        b = sum(r['identity_nll_sum'] for r in selected)/count
        result.append({'corpus':corpus, 'n_sequences':len(selected), 'n_targets':count,
            'baseline_ce':a, 'identity_ce':b, 'delta_ce':b-a,
            'baseline_token_ppl':math.exp(a), 'identity_token_ppl':math.exp(b),
            'max_abs_logit_delta':max(r['max_abs_logit_delta'] for r in selected),
            'max_abs_token_nll_delta':max(r['max_abs_token_nll_delta'] for r in selected),
            'argmax_disagreements':sum(r['argmax_disagreements'] for r in selected),
            'within_tolerance':all(r['within_tolerance'] for r in selected)})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cfg', default='Configs/ttt_ar_unified.yml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', help='completed stage-2 adapter; TTT sidecar is required when provided')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--corpora', nargs='+', choices=['pg19','wikitext'], default=['pg19','wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--logit-block', type=int, default=128)
    ap.add_argument('--atol', type=float, default=1e-3, help='absolute logit-difference tolerance')
    ap.add_argument('--out', required=True, help='new output directory')
    args = ap.parse_args()
    if args.seq_len<2 or args.seqs<1 or args.logit_block<1 or not math.isfinite(args.atol) or args.atol<0:
        ap.error('Invalid sequence count/length, block size, or tolerance')
    if not torch.cuda.is_available():
        ap.error('The checkpoint loader requires a CUDA GPU')
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from Training.train import build_model_config, resume_stage2
    from eval import load_model, ttt_layers
    from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window

    cfg = OmegaConf.load(args.cfg)
    cfg.model.pretrained_model_name_or_path = args.ckpt
    cfg.model.max_length = args.seq_len
    cfg.model.ttt_reader_alignment = 'linear'
    model_config = build_model_config(cfg)
    model_config.validate_ttt()
    corpora, sources = load_corpora(args, AutoTokenizer.from_pretrained(args.base))
    for name,ids in corpora.items():
        if tuple(ids.shape)!=(args.seqs,args.seq_len) or int(ids.min())<0 or int(ids.max())>=model_config.vocab_size:
            raise ValueError(f'{name}: invalid token shape or IDs outside model vocabulary')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    paths = {'ckpt':args.ckpt, 'adapter':args.adapter}
    identities = {k:checkpoint_files(v) for k,v in paths.items()}
    metadata = {'complete':False, 'arguments':vars(args), 'sources':sources,
        'checkpoint_files':identities, 'model_config':model_config.to_dict(),
        'metric':'exp(total next-token NLL / total predicted tokens); sampled token perplexity',
        'comparison':'same frozen model: bypass maps versus execute fixed identity maps; fresh memory each forward',
        'token_hashes':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in corpora.items()},
        'torch_version':str(torch.__version__), 'gpu':torch.cuda.get_device_name(0)}
    root = Path(__file__).resolve().parent
    metadata['code_hashes'] = {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in (
        'eval_reader_identity.py','eval.py','Training/train.py',
        'LinearTTT/model/LinearizeLlama/LinearizeLlama.py')}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    model = load_model(args.ckpt, model_config)
    if args.adapter:
        model = resume_stage2(model, args.adapter)
    model.eval().requires_grad_(False)
    if any(m._ablate_ttt or m._ablate_attn for m in ttt_layers(model)):
        raise ValueError('Use an intact model')
    readers = identity_readers(model)
    metadata['reader_layers'] = [m.layer_idx for m in readers]
    use_sdpa_sliding_window(True)
    print(f'Fixed identity on {len(readers)} readers: {metadata["reader_layers"]}', flush=True)
    print('Paired evaluation only; no optimizer or training. PPL is TOKEN perplexity.', flush=True)
    rows = []
    for corpus,sequences in corpora.items():
        for sequence,ids in enumerate(sequences):
            print(f'{corpus} {sequence+1}/{len(sequences)}: baseline + identity', flush=True)
            row = {'corpus':corpus, 'sequence':sequence,
                   **paired_sequence(model, ids.unsqueeze(0), readers, args.logit_block, args.atol)}
            rows.append(row)
            print(f'  token PPL {row["baseline_token_ppl"]:.6f} -> {row["identity_token_ppl"]:.6f}; '
                  f'max |delta logit| {row["max_abs_logit_delta"]:.3g}; '
                  f'{"PASS" if row["within_tolerance"] else "CHECK"}', flush=True)
            (out/'sequences.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    if identities!={k:checkpoint_files(v) for k,v in paths.items()}:
        raise RuntimeError('Source checkpoint changed during evaluation')
    summary = aggregate(rows)
    metadata['complete'] = True
    metadata['all_within_tolerance'] = all(r['within_tolerance'] for r in rows)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (out/'results.csv').open('w',newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    print('\ncorpus       baseline token PPL   identity token PPL   max |delta logit|')
    for row in summary:
        print(f'{row["corpus"]:<12} {row["baseline_token_ppl"]:>18.6f} '
              f'{row["identity_token_ppl"]:>20.6f} {row["max_abs_logit_delta"]:>19.3g}')
    print(f'Wrote {out}/results.json', flush=True)
    if not metadata['all_within_tolerance']:
        raise SystemExit('Identity comparison exceeded tolerance; inspect saved per-sequence differences')


if __name__=='__main__':
    main()
