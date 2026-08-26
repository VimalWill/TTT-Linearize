import os
import shutil
import random
import itertools
from tqdm import tqdm
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
import evaluate
from huggingface_hub import hf_hub_download

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq  


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

def encode_response(response: str, tokenizer) -> list[int]:
    tokens = tokenizer.encode(response.strip(), add_special_tokens=False)
    # For Llama 3 Instruct: tokens.append(tokenizer.get_added_vocab()["<|eot_id|>"])
    tokens.append(tokenizer.eos_token_id)  
    try:  # Llama 3 Instruct
        tokens.append(tokenizer.get_added_vocab()["<|end_of_text|>"])
    except KeyError:
        pass
    return tokens

def load_data(config):
    # Use HF_DATASETS_CACHE env var first, then XDG_CACHE_HOME, then user home directory
    cache_dir = os.environ.get('HF_DATASETS_CACHE') or os.path.join(
        os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 
        'huggingface', 
        'datasets'
    )
    input_len = config.model.max_length
    concat_data = True

    tokenizer_path = config.model.pretrained_model_name_or_path
    tokenizer_name = tokenizer_path.split('/')[-1]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation
        # Get initial data
    ignore_kwargs = ['concat_data', 'chunk_size', 'pose_kwargs']
    data_path = config.data.path
    data_name = config.data.get("name", "alpaca_cleand")
    # select formatter based on dataset
    if any(t in data_name.lower() for t in ("pg19", "books", "longtext")):
        # Long-form continuous text: no prompt template, every token supervised.
        # ConcatDataset packing is the right thing here (unlike LongBench, where
        # only the answer carries a label and packing discards the context).
        formatter = partial(template_and_tokenize_lm, tokenizer=tokenizer)
        n_train_docs = int(config.data.get("num_train_docs", 200))
        n_val_docs = int(config.data.get("num_val_docs", 20))
        max_doc_tokens = config.data.get("max_doc_tokens", None)
        if max_doc_tokens is not None:
            formatter = partial(formatter, max_doc_tokens=int(max_doc_tokens))

        def _take(split, n):
            # stream so we do not pull the full corpus (pg19 is ~11GB / 28k books);
            # materialise the slice because from_generator may re-run the generator
            it = load_dataset(data_path, split=split, streaming=True)
            return convert_to_hf_dataset(list(itertools.islice(it, n)), cache_dir)

        train_set = _take("train", n_train_docs)
        val_set = _take("validation", n_val_docs)
        test_set = _take("test", n_val_docs)
        cols = list(train_set.features)
    elif "longbench" in data_name.lower():
        formatter = partial(template_and_tokenize_longbench, tokenizer=tokenizer)
        # Load each task's JSONL directly (THUDM/LongBench uses a loading script not supported)
        lb_tasks = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "narrativeqa"]
        from huggingface_hub import hf_hub_download
        import json, zipfile, tempfile, pathlib
        all_samples = []
        zip_path = hf_hub_download(
            repo_id="zai-org/LongBench",
            filename="data.zip",
            repo_type="dataset",
        )
        extract_dir = pathlib.Path(zip_path).parent / "longbench_data"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
        for task in lb_tasks:
            # try both data/{task}.jsonl and {task}.jsonl inside zip
            for candidate in [
                extract_dir / "data" / f"{task}.jsonl",
                extract_dir / f"{task}.jsonl",
            ]:
                if candidate.exists():
                    with open(candidate) as f:
                        for line in f:
                            all_samples.append(json.loads(line))
                    break
        import random as _random
        _random.seed(42)
        _random.shuffle(all_samples)
        n_val = min(200, len(all_samples) // 10)
        train_samples = all_samples[n_val:]
        val_samples   = all_samples[:n_val]
        train_set = convert_to_hf_dataset(train_samples, cache_dir)
        val_set   = convert_to_hf_dataset(val_samples, cache_dir)
        test_set  = val_set
        cols = list(train_set.features)
    else:
        formatter = partial(template_and_tokenize, tokenizer=tokenizer)
        dataset = load_dataset(
            **{k: v for k, v in {"path": data_path, "cache_dir": cache_dir}.items()}
        )
        dataset = dataset['train']
        train_set = convert_to_hf_dataset([dataset[ix] for ix in range(200, len(dataset))], cache_dir)
        val_set   = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)
        test_set  = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)
        cols = list(dataset.features)

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(
        partial(formatter, include_label=True),
        remove_columns=cols)
    val_set = val_set.map(
        partial(formatter, include_label=True),
        remove_columns=cols)
    test_set = test_set.map(
        partial(formatter, include_label=False),
        remove_columns=cols)

    # Chunk together train and val sets
    if concat_data:
        train_set = ConcatDataset(train_set, chunk_size=input_len)
        val_set = ConcatDataset(val_set, chunk_size=input_len)

    loader_kwargs = {
        "batch_size": config.data.micro_batch_size,
        "num_workers": 0,
        "drop_last": False,
        "pin_memory": True,
    }

    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, tokenizer, 'train', input_len, **loader_kwargs),
        'validation': get_lm_loader(val_set, tokenizer, 'validation', input_len, **loader_kwargs),
        'test': get_seq2seq_loader(test_set, tokenizer, 'test', **loader_kwargs),
    }
    # Evaluation metric
    try:
        # metric = load_metric(download_metric(), 'gov_report')  # hack but we want rouge
        metric = evaluate.load(download_metric(), 'gov_report') 
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def convert_to_hf_dataset(dataset, cache_dir: str):
    """
    Convert iterable dataset to HuggingFace HFDataset object
    """
    def gen():
        for _, sample in enumerate(dataset):
            yield sample  # dataset[idx]
    return HFDataset.from_generator(gen, cache_dir=cache_dir)

def template_and_tokenize(sample, tokenizer, include_label: bool = True):
    """
    Format dataset context and answers into single-sequence prompts
    """
    if sample.get('input', '') == '':
        prompt = PROMPT_DICT["prompt_no_input"].format_map(sample)
    else:
        prompt = PROMPT_DICT["prompt_input"].format_map(sample)

    prompt = tokenizer.encode(prompt, add_special_tokens=True)
    if include_label:
        answer = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
        target = None
    else:
        answer = []
        target = tokenizer.encode(f'{sample["output"]}{tokenizer.eos_token}', 
                                  add_special_tokens=False)
    input_ids = prompt + answer
    attn_mask = [1] * len(input_ids)

    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
    }
    return sample

def template_and_tokenize_lm(sample, tokenizer, include_label: bool = True,
                             max_doc_tokens: int = None):
    """
    Plain language modelling over continuous text (PG19 and friends).

    No prompt template and no masking: every token is a target, so ConcatDataset
    packs whole documents into fully supervised chunks and its all--100 filter
    never drops anything.
    """
    text = sample.get("text", "")
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    if max_doc_tokens is not None:
        input_ids = input_ids[:max_doc_tokens]
    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + input_ids
    input_ids = input_ids + [tokenizer.eos_token_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": list(input_ids),
    }


def template_and_tokenize_longbench(sample, tokenizer, include_label: bool = True):
    """
    Format LongBench samples: {input, context, answers} → single sequence
    """
    context = sample.get("context", "")
    question = sample.get("input", "")
    answer = sample["answers"][0] if isinstance(sample.get("answers"), list) else sample.get("answers", "")

    prompt = (
        f"Read the following passage and answer the question.\n\n"
        f"Passage:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if include_label:
        answer_ids = tokenizer.encode(f" {answer}{tokenizer.eos_token}", add_special_tokens=False)
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
    else:
        input_ids = prompt_ids
        labels = tokenizer.encode(f" {answer}{tokenizer.eos_token}", add_special_tokens=False)

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def get_lm_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                  split: str, max_length: int = None, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    -> Currently this ends up being the same as get_seq2seq_loader
    """
    # collate_fn = DefaultDataCollator(return_tensors='pt')
    # collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True,
    #                                      max_length=max_length, return_tensors='pt')
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)

def get_seq2seq_loader(dataset: Dataset, tokenizer: AutoTokenizer,
                       split: str, **loader_kwargs: any):
    """
    Get dataloader for seq2seq tasks (evaluation)
    """
    tokenizer.padding_side = 'right'
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)

def download_metric():
    """
    Download ROUGE, F1, and other accuracy metrics included in the SCROLLS dataset
    """
    scrolls_metric_path = hf_hub_download(
        repo_id="tau/scrolls", filename="metrics/scrolls.py", repo_type="dataset"
    )
    updated_scrolls_metric_path = os.path.join(
        os.path.dirname(scrolls_metric_path),
        os.path.basename(scrolls_metric_path).replace(".", "_") + ".py",
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path

class ConcatDataset(Dataset):
    """
    Concatenates or packs samples of a dataset into chunks of size `chunk_size`
    """
    def __init__(self, dataset, chunk_size: int = 1024, seed: int = 42,) -> None:
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        random.seed(seed)
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
        # Slow hack, but filter out any samples without valid labels (all -100)
        self.filtered_samples = []
        for s in self.samples:
            if sum(s['labels']) != chunk_size * -100:
                self.filtered_samples.append(s)
        if len(self.filtered_samples) < len(self.samples):
            print(f'OG dataset: {len(self.samples)} samples -> Filtered dataset: {len(self.filtered_samples)}')
            print(f'-> Filtered out {len(self.samples) - len(self.filtered_samples)} samples')
                
    def __getitem__(self, idx):
        return self.filtered_samples[idx]
    
    def __len__(self):
        return len(self.filtered_samples)
