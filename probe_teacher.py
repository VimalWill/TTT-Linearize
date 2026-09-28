"""Is the stage-1 teacher a valid target for this backbone?

Stage 1 regresses the hybrid onto F.scaled_dot_product_attention(..., is_causal=True)
-- FULL causal attention over the whole sequence. For Llama-3.1-8B that is the
backbone's native mechanism. Mistral-7B-v0.1 was pretrained with a 4096 sliding
window, so full causal at 8192 is off-distribution and the teacher may be
regressing the student onto attention the base model never produces.

This measures the base model's own CE at --seq-len under both mechanisms. If
full-causal CE is far worse than native, the teacher is the problem and no
amount of stage-1 training fixes it.

    python probe_teacher.py --base mistralai/Mistral-7B-v0.1 --seq-len 8192
    python probe_teacher.py --base mistralai/Mistral-7B-v0.3 --seq-len 8192
"""
import argparse, itertools, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


@torch.no_grad()
def ce(model, ids):
    out = model(input_ids=ids, use_cache=False)
    lg = out.logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=4)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.base)
    cfg = AutoConfig.from_pretrained(a.base)
    native_win = getattr(cfg, 'sliding_window', None)
    print(f'{a.base}: sliding_window={native_win}, rope_theta={cfg.rope_theta}')

    # same corpus stage 1 trains on
    txt, n = [], 0
    for row in load_dataset('Yukang/LongAlpaca-12k', split='train', streaming=True):
        s = '\n\n'.join(str(row[k]) for k in ('instruction', 'input', 'output') if row.get(k))
        txt.append(s); n += len(s)
        if n > a.seq_len * a.seqs * 5:
            break
    ids = tok('\n\n'.join(txt), return_tensors='pt').input_ids[0]
    batch = torch.stack([ids[i * a.seq_len:(i + 1) * a.seq_len] for i in range(a.seqs)])
    print(f'{tuple(batch.shape)} tokens')

    for label, win in (('native', native_win), ('full causal', None)):
        cfg2 = AutoConfig.from_pretrained(a.base)
        cfg2.sliding_window = win
        m = AutoModelForCausalLM.from_pretrained(
            a.base, config=cfg2, torch_dtype=torch.bfloat16,
            device_map='auto', attn_implementation='sdpa').eval()
        vals = [ce(m, batch[i:i + 1].to(m.device)) for i in range(a.seqs)]
        mean = sum(vals) / len(vals)
        print(f'{label:12s} (sliding_window={win}): CE {mean:.4f}  ppl {torch.tensor(mean).exp():.1f}')
        del m; torch.cuda.empty_cache()

    print('\nIf full-causal CE is far worse than native, the stage-1 teacher is '
          'not this backbone\'s real behaviour and MSE-down/CE-flat is expected.')


if __name__ == '__main__':
    main()
