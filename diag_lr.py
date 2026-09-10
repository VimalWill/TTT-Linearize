"""Are the inner-loop update magnitudes miscalibrated off-distribution?

The one unbounded quantity in the TTT update path. q, k are l2-normalised and v
is too under the l2 loss, but the per-token learning rate is
`softplus(lr_proj(h) + base_lr_inv)` with no ceiling. lr_proj is fitted on the
training corpus; on another corpus the hidden states shift and lr can run hot.
Under Muon that matters doubly -- Newton-Schulz discards the update's magnitude
and eta is reapplied afterwards, so lr alone sets how hard each chunk is written.

That predicts exactly the observed pattern: invisible on short sequences (the
update loop never runs), fine in-domain (calibrated), bad out-of-domain.

    python diag_lr.py --cfg Configs/ttt_ar_unified.yml --ckpt BASE --adapter ADAPTER

Prints per-layer lr and retention stats on the training corpus vs wikitext.
In a shared model only the WRITER's values reach the memory -- readers run no
inner loop -- so the writer row is the one that matters.
"""
import argparse

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import LinearTTT  # noqa: F401
from Training.train import build_model_config
from eval import load_model, ttt_layers


def stats(x):
    x = x.flatten().float()
    q = torch.quantile(x[torch.randperm(x.numel())[:200_000]],
                       torch.tensor([.5, .99], device=x.device))
    return x.mean().item(), q[0].item(), q[1].item(), x.max().item()


@torch.no_grad()
def probe(model, mods, ids):
    """-> {layer: (lr_stats, ret_stats)} for one batch."""
    acc = {}
    hooks = []

    def mk(i, kind):
        def fn(mod, inp, out):
            if kind == 'lr':
                # mirror the forward exactly: softplus AFTER the bias shift
                v = F.softplus(out.float() + mods[i].base_lr_inv)
            else:
                v = out.float()          # retention_proj ends in Sigmoid
            acc.setdefault(i, {})[kind] = stats(v)
        return fn

    for i, m in enumerate(mods):
        hooks.append(m.lr_proj.register_forward_hook(mk(i, 'lr')))
        if hasattr(m, 'retention_proj'):
            hooks.append(m.retention_proj.register_forward_hook(mk(i, 'ret')))
    model(input_ids=ids.to(model.device), use_cache=False)
    for h in hooks:
        h.remove()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=8192)
    args = ap.parse_args()

    config = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True))
    config.model.pretrained_model_name_or_path = args.ckpt
    config.model.max_length = args.seq_len
    model_config = build_model_config(config)
    model = load_model(args.ckpt, model_config, args.adapter)
    mods = ttt_layers(model)

    groups = getattr(model_config, 'ttt_share_groups', None) or []
    writers = {min(g) for g in groups}
    readers = {i for g in groups for i in g} - writers
    if groups:
        print(f'shared: writer(s) {sorted(writers)}, {len(readers)} readers '
              '(readers run NO inner loop -- their lr never reaches the memory)')

    # ---- in-domain batch, straight from the training loader ----
    from Training.dataloader import load_data
    pg = next(iter(load_data(config)['validation']))['input_ids'][:1]

    # ---- out-of-domain batch: wikitext, same length ----
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.base)
    txt = '\n\n'.join(load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    wt = tok(txt, return_tensors='pt').input_ids[:, :args.seq_len]
    if wt.shape[1] < args.seq_len:
        raise SystemExit(f'wikitext gave only {wt.shape[1]} tokens')
    print(f'pg19 {tuple(pg.shape)}, wikitext {tuple(wt.shape)}')

    a, b = probe(model, mods, pg), probe(model, mods, wt)

    print(f'\n{"":6}{"":9}{"--------- in-domain ---------":>30}'
          f'{"--------- wikitext ----------":>30}{"":>10}')
    print(f'{"layer":>6}{"role":>9}' + ''.join(f'{h:>7}' for h in
          ('mean', 'p50', 'p99', 'max')) * 2 + f'{"p99 x":>10}')
    worst = []
    for i in sorted(a):
        role = ('writer' if i in writers else 'reader') if groups else 'private'
        for kind in ('lr', 'ret'):
            if kind not in a[i]:
                continue
            x, y = a[i][kind], b[i][kind]
            ratio = y[2] / x[2] if x[2] else float('nan')
            tag = f'{role}/{kind}'
            print(f'{i:>6}{tag:>9}' + ''.join(f'{v:>7.3f}' for v in x)
                  + ''.join(f'{v:>7.3f}' for v in y) + f'{ratio:>10.2f}')
            if kind == 'lr':
                worst.append((ratio, i, role))
    print('\np99 x  = wikitext p99 / in-domain p99. >1 means the memory is '
          'written harder off-distribution.')
    worst.sort(reverse=True)
    print('largest lr inflation: ' + ', '.join(
        f'L{i}({r:.2f}, {ro})' for r, i, ro in worst[:5]))
    if groups:
        w = [(r, i) for r, i, ro in worst if ro == 'writer']
        print(f'WRITER lr inflation (the only one that reaches the memory): '
              + ', '.join(f'L{i}: {r:.2f}x' for r, i in w))


if __name__ == '__main__':
    main()
