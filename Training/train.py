import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import fla
import LinearTTT
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import TrainingArguments
from peft import LoraConfig, TaskType, PeftModel, get_peft_model

from Training.trainer import DefaultTrainer, FinetuneTrainer
from Training.utils import get_optimizer_and_scheduler, count_model_params
from Training.dataloader import load_data


# Parameters belonging to the test-time-training branch. These have no
# counterpart in the pretrained checkpoint, so unlike the Liger recipe they must
# actually be trained -- LoRA on q/k/v alone leaves them at their random init.
TTT_PARAM_KEYS = (
    '.w0', '.w1', '.w2',
    'lr_proj', 'ttt_scale_proj', 'ttt_norm',
    'ttt_qk_scale', 'ttt_qk_offset', 'momentum_proj',
)

# yml `model:` keys consumed by the harness rather than by the model config
_HARNESS_ONLY = {'name', 'pretrained_model_name_or_path', 'device_map',
                 'add_eos_token', 'max_length'}


def build_model_config(config):
    """Load the checkpoint's own shape, then apply the yml `model:` overrides.

    Constructing LigerGLAConfig() bare gives Llama-2 defaults (vocab 32000,
    32 kv heads, rope_theta 10000), which silently mismatch a Llama-3 checkpoint.
    """
    from LinearTTT.model.LinearizeLlama import LigerGLAConfig
    model_config = LigerGLAConfig.from_pretrained(
        config.model.pretrained_model_name_or_path
    )
    for k, v in config.model.items():
        if k in _HARNESS_ONLY:
            continue
        # Configs/liger.yml spells it `attn_varient`
        setattr(model_config, 'attn_variant' if k == 'attn_varient' else k, v)
    # the packing width is what the layer actually sees
    model_config.max_position_embeddings = max(
        model_config.max_position_embeddings, int(config.model.max_length)
    )
    return model_config


def set_trainable_params(model, config):
    """Decide the trainable set. Must run *after* get_peft_model, which marks
    every non-adapter parameter frozen."""
    train_ttt = config.model.get('attn_varient', None) == 'ttt' or \
        config.model.get('attn_variant', None) == 'ttt'
    for name, param in model.named_parameters():
        param.requires_grad = bool(
            'lora_' in name
            or (train_ttt and any(k in name for k in TTT_PARAM_KEYS))
        )
    return model


def train(config):

    # stage: 'ttt_at' = attention transfer (per-layer distillation, no LM loss)
    #        'ttt_ar' = autoregressive finetune on the LM loss
    stage = config.model.name
    trainer = DefaultTrainer if stage.endswith('_at') else FinetuneTrainer
    if stage not in ('liger_gla', 'ttt_at', 'ttt_ar'):
        raise NotImplementedError(stage)

    model_config = build_model_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.pretrained_model_name_or_path,
        config=model_config,
        device_map=config.model.get('device_map', 'auto'),
    ).to(torch.bfloat16)


    print("Model config:")
    print(model_config)
    print("Model:")
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    # LoRA finetune. Attention transfer trains only the TTT branch: the
    # pretrained projections have to stay exactly as the teacher sees them, or
    # the regression target moves with the student.
    target_modules = []
    if stage != 'ttt_at':
        if "train_qk" in config.train and config.train.train_qk and config.train.train_qk_lora:
            target_modules.append("self_attn.q_proj")
            target_modules.append("self_attn.k_proj")
        if "train_v" in config.train and config.train.train_v and config.train.train_v_lora:
            target_modules.append("self_attn.v_proj")
        if "train_o" in config.train and config.train.train_o and config.train.train_o_lora:
            target_modules.append("self_attn.o_proj")
    if len(target_modules) != 0:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=target_modules)
        model = get_peft_model(model, peft_config=lora_config)

    set_trainable_params(model, config)

    # print trainable params count
    trainable_params = count_model_params(model, requires_grad=True)
    total_params = count_model_params(model, requires_grad=False)
    print(f"Model trainable params: {trainable_params}")
    print(f"Model total params: {total_params}")
    print(f"trainable%: {trainable_params / total_params}")

    gradient_accumulation_steps = config.data.batch_size // config.data.micro_batch_size

    print("Preparing data...")

    dataloaders  = load_data(config)
    train_loader = dataloaders["train"]
    eval_loader  = dataloaders["validation"]

    print("Building trainer...")    

    training_args = TrainingArguments(
            per_device_train_batch_size=config.data.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=config.train.epochs,
            learning_rate=config.train.lr,
            bf16=True,
            max_grad_norm=config.train.max_grad_norm,
            logging_steps=1,
            optim=config.train.optim,
            eval_strategy="steps" if config.data.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=config.train.get('eval_steps', 200) if config.data.val_set_size > 0 else None,
            save_steps=config.train.get('save_steps', 1000),
            logging_dir=config.train.output_dir,
            output_dir=config.train.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if config.data.val_set_size > 0 else False,
            # default trainer args
            greater_is_better = False,
            metric_for_best_model = 'eval/loss',
            # wandb
            report_to="none" # wandb off "wandb"
        )
    
    trainer = trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        args=training_args,
        optimizers=get_optimizer_and_scheduler(model, config),
        tokenizer=tokenizer,
        config=config
    )

    print("Train start")
    best_model = trainer.train()
    save_path = trainer.save_path + '/best'
    best_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'\n-> Saved best model checkpoint to: {save_path}!')

    print("Train over")