import sys
import os
import math
from collections.abc import Mapping
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
import os
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments

from peft import PeftModel
from Training.utils import model_autocast

def save_checkpoint(model, tokenizer, save_path):
    """Save a checkpoint, including TTT parameters under PEFT.

    PeftModel.save_pretrained writes only adapter weights. The TTT parameters
    (w0/w1/w2, lr_proj, ttt_scale_proj, ...) are plain base-model tensors that
    training also updates, so they would be silently dropped -- measured once as
    an 18.9MB stage-2 checkpoint containing nothing but LoRA.

    Shared TTT memories need the follower copies dropped by hand. safetensors
    refuses to write tensors that alias each other, and declaring them in
    _tied_weights_keys was not enough -- transformers' own dedup pass let them
    through and the failure surfaced deeper, inside safe_save_file. Removing
    them from the state dict is explicit and version-independent. Loading is
    unaffected: the follower keys come back as missing and tie_weights()
    re-aliases them to their leader, which is where the values live.
    """
    sd = None
    groups = getattr(getattr(model, 'config', None), 'ttt_share_groups', None)
    if groups:
        # match on suffix so this works whether or not a PeftModel wrapper has
        # prefixed every key with base_model.model.
        tails = tuple(f'layers.{li}.self_attn.{w}'
                      for g in groups for li in sorted(g)[1:]
                      for w in ('w0', 'w1', 'w2'))
        sd = {k: v for k, v in model.state_dict().items() if not k.endswith(tails)}
        print(f'-> dropped {len(model.state_dict()) - len(sd)} aliased '
              f'fast-weight tensors from the checkpoint')
    model.save_pretrained(save_path, state_dict=sd)
    tokenizer.save_pretrained(save_path)
    if isinstance(model, PeftModel):
        ttt = {n: p.detach().cpu() for n, p in model.named_parameters()
               if p.requires_grad and 'lora_' not in n}
        if ttt:
            torch.save(ttt, os.path.join(save_path, 'ttt_params.pt'))
            print(f'-> Saved {len(ttt)} TTT tensors alongside the adapters')


class DefaultTrainer():
    # code is modified from: https://github.com/HazyResearch/lolcats/blob/main/src/trainer/default_lm.py
    def __init__(self, model, train_loader, eval_loader, args, optimizers, tokenizer, config):
        super().__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.config = config
        self.type = 'default'

        self.step = 0  # Total steps taken
        self.grad_step = 0  # Total gradient updates
        train_options = getattr(config, 'train', {})
        if not hasattr(train_options, 'get'):
            train_options = vars(train_options)
        self.train_options = train_options
        self.input_tokens = 0
        self.pending_input_tokens = 0
        self.pending_exposure = {}
        self.training_exposure = {}
        self._last_saved_grad_step = None
        self.max_input_tokens = int(train_options.get('max_input_tokens', 0))
        self.phase_transition_tokens = int(train_options.get('phase_transition_tokens', 0))
        self.final_max_length = int(train_options.get('final_max_length', 16384))
        if self.max_input_tokens < 0 or self.phase_transition_tokens < 0:
            raise ValueError('Token budgets must be nonnegative')
        if self.max_input_tokens and self.phase_transition_tokens >= self.max_input_tokens:
            raise ValueError('Phase transition must precede the total token budget')
        self.compute_loss_backprop = False  # Whether we backprop in self.compute_loss

        self.optimizer, self.scheduler = optimizers
        # Plateau schedules consume validation metrics; step schedules advance
        # only after a successful optimizer update, never on evaluation.
        self.scheduler_step_after_epoch = isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        # Dataloaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = model.device
        wandb = None
        self.wandb = wandb

        # args
        self.metric_for_best_model = self.args.metric_for_best_model
        self.num_train_epochs = self.args.num_train_epochs
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.eval_strategy = self.args.eval_strategy
        self.greater_is_better = self.args.greater_is_better
        self.is_better = (lambda x, y: x > y if self.args.greater_is_better else x < y)
        self.load_best_model_at_end = self.args.load_best_model_at_end
        self.logging_steps = self.args.logging_steps
        self.max_steps = self.args.max_steps
        self.eval_steps = self.args.eval_steps

        max_eval_batches = -1
        print_samples = False
        initial_eval = True
        self.max_eval_batches = max_eval_batches
        self.print_samples = print_samples
        self.initial_eval = initial_eval
        self.save_total_limit = self.args.save_total_limit 
        self.save_steps = self.args.save_steps # num_save_ckpt_steps

        # Saving metrics
        self.train_metrics = {'train/loss': None, 
                              'train/epoch': None, 
                              'train/step': None}
        self.eval_metrics = {self.metric_for_best_model: None}
        self.eval_metrics_by_step = {'eval_step': []}  # save all eval metrics
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
            
        save_results = True
        save_checkpoints = True
        
        self.save_results = save_results
        self.results_path = None
        self.best_val_metric = float('-inf') if self.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        if save_checkpoints:  # Also initializes best_val_metrics
            self.init_checkpointing(config=config)

    def train(self) -> nn.Module:
        """
        Entire training run
        """
        model = self.model
        pbar = tqdm(range(self.num_train_epochs), leave=False, colour='white', desc='Training')
        for ix, epoch in enumerate(pbar):
            model, early_stopping = self.train_step(model, epoch)
            if self.eval_strategy == 'epoch':
                _eval_metrics = self.eval_step(model, step=self.grad_step)
                print(f'Epoch {ix} metrics:', _eval_metrics)
            if early_stopping:
                break
                
        if self.max_input_tokens:
            if self.input_tokens < self.max_input_tokens:
                raise RuntimeError(f'Training ended before token budget: {self.input_tokens}/{self.max_input_tokens}')
            # The token limit rarely lands on an eval_steps boundary. Validate
            # the final update before restoring the selected checkpoint.
            if self.eval_metrics_by_step['eval_step'][-1:] != [self.grad_step]:
                self.eval_step(model, step=self.grad_step)
            # Keep the actual final state even if no checkpoint beat the
            # initial retrieval score. It is never silently called the best.
            self.save_latest_checkpoint(model)

        if self.load_best_model_at_end:  # Return best checkpoint
            try:
                import os
                ckpt_path = os.path.abspath(self.best_val_checkpoint_path)
                if isinstance(model, PeftModel):
                    from peft import set_peft_model_state_dict
                    import torch
                    adapter_loaded = False
                    for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
                        fpath = os.path.join(ckpt_path, fname)
                        if os.path.exists(fpath):
                            if fname.endswith('.safetensors'):
                                from safetensors.torch import load_file
                                weights = load_file(fpath, device="cpu")
                            else:
                                weights = torch.load(fpath, map_location="cpu")
                            result = set_peft_model_state_dict(model, weights)
                            if self.max_input_tokens and (result.unexpected_keys or
                                    any('lora_' in key for key in result.missing_keys)):
                                raise RuntimeError('Selected checkpoint has incomplete adapter weights')
                            adapter_loaded = True
                            break
                    if self.max_input_tokens and not adapter_loaded:
                        raise FileNotFoundError(f'Missing selected adapter weights in {ckpt_path}')
                    # The adapters are only half the checkpoint -- the TTT
                    # tensors are saved separately by save_checkpoint. Without
                    # this the model carries the LAST step's memory with the
                    # BEST step's adapters.
                    ttt = os.path.join(ckpt_path, 'ttt_params.pt')
                    if os.path.exists(ttt):
                        sd = torch.load(ttt, map_location='cpu')
                        if self.max_input_tokens:
                            expected = {n for n, p in model.named_parameters()
                                        if p.requires_grad and 'lora_' not in n}
                            if expected - sd.keys():
                                raise RuntimeError('Selected checkpoint has incomplete TTT weights')
                        unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
                        if self.max_input_tokens and unexpected:
                            raise RuntimeError('Selected checkpoint contains unmatched TTT weights')
                        matched = len(sd) - len(unexpected)
                        if matched == 0:
                            raise RuntimeError(
                                f'{ttt} has {len(sd)} tensors but none matched; '
                                f'first key {next(iter(sd))}'
                            )
                        print(f'-> Restored {matched}/{len(sd)} TTT tensors')
                    elif self.max_input_tokens:
                        raise FileNotFoundError(f'Missing selected TTT weights: {ttt}')
                else:
                    model = model.from_pretrained(
                        ckpt_path,
                        torch_dtype=model.dtype,
                        device_map=getattr(model, 'hf_device_map', {'': model.device}),
                    )
                    self.model = model
                print(f'-> Loading best checkpoint from {ckpt_path}')
            except Exception as e:
                if self.max_input_tokens:
                    raise RuntimeError('Could not restore the selected long-context checkpoint') from e
                print(e)
                print('-> Returning most recent model instead')
        return model            
    
    def _optimizer_step(self, accumulated_batches, accumulation_steps):
        """Normalize an accumulation window, check/clip gradients, then update."""
        if self.max_input_tokens and self.pending_input_tokens <= 0:
            raise RuntimeError('Token-budget update has no counted input tokens; refusing to step the optimizer')
        params = [p for group in self.optimizer.param_groups for p in group['params']
                  if p.grad is not None]
        if not params:
            self.pending_input_tokens = 0
            self.pending_exposure = {}
            return False
        # Each loss was divided by accumulation_steps before backward. A short
        # final window must instead average over the batches actually present.
        if accumulated_batches != accumulation_steps:
            scale = accumulation_steps / accumulated_batches
            for param in params:
                param.grad.mul_(scale)
        max_norm = self.args.max_grad_norm
        norm = torch.nn.utils.clip_grad_norm_(
            params, max_norm if max_norm is not None and max_norm > 0 else float('inf'),
        )
        if not torch.isfinite(norm):
            if self.train_options.get('fail_on_nonfinite', False):
                raise FloatingPointError('Nonfinite gradient; stopping the matched-budget run')
            print('\n-> Nonfinite gradient norm, skipping accumulated update')
            self.pending_input_tokens = 0
            self.pending_exposure = {}
            self.optimizer.zero_grad()
            return False
        self.optimizer.step()
        self.input_tokens += self.pending_input_tokens
        for cell, exposure in self.pending_exposure.items():
            total = self.training_exposure.setdefault(cell, dict(examples=0, tokens=0))
            total['examples'] += exposure['examples']
            total['tokens'] += exposure['tokens']
        self.pending_exposure = {}
        self.pending_input_tokens = 0
        sampler = getattr(self.train_loader, 'sampler', None)
        if self.phase_transition_tokens and self.input_tokens >= self.phase_transition_tokens:
            if hasattr(sampler, 'set_max_length'):
                sampler.set_max_length(self.final_max_length)
        if not self.scheduler_step_after_epoch and self.scheduler is not None:
            if hasattr(self.scheduler, 'step_tokens'):
                self.scheduler.step_tokens(self.input_tokens)
            else:
                self.scheduler.step()
        self.optimizer.zero_grad()
        self.grad_step += 1
        return True

    def train_step(self, model, epoch) -> nn.Module:
        if self.gradient_accumulation_steps is None:
            accum_iter = 1
        else:
            accum_iter = self.gradient_accumulation_steps
        if not isinstance(accum_iter, int) or isinstance(accum_iter, bool) or accum_iter < 1:
            raise ValueError('gradient_accumulation_steps must be a positive integer')

        model.train()
        model.zero_grad()        
        accumulated_batches = 0
        self.pending_input_tokens = 0
        self.pending_exposure = {}
        num_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, leave=False, colour='blue', desc=f'-> Training (epoch {epoch} / {self.args.num_train_epochs})')
        total_loss = 0
        successful_batches = 0
        eval_for_step = False

        # Initial eval
        if self.initial_eval:
            print('')
            print('-> Initial eval')
            # Register and save the starting model too. Logging alone leaves
            # best_val_metric at its sentinel, so the first trained checkpoint
            # is called "best" even when it is worse than initialization.
            self.eval_step(model, step=self.grad_step)
            self.initial_eval = False
            # compute_eval_metrics sets model.eval() and does not restore it.
            # Without this the whole training loop runs in eval mode, which
            # disables the `self.gradient_checkpointing and self.training`
            # branch in LigerGLAModel.forward.
            model.train()
        
        # model.to(self.device)
        for ix, data in enumerate(pbar):
            # DataCollatorForSeq2Seq returns BatchEncoding (a Mapping), not dict.
            # Reject uncountable budget batches before doing a costly forward.
            batch_tokens = 0
            if isinstance(data, Mapping) and 'input_ids' in data:
                mask = data.get('attention_mask')
                batch_tokens = int(mask.sum().item() if mask is not None else data['input_ids'].numel())
            if self.max_input_tokens and batch_tokens <= 0:
                raise ValueError('Token-budget batches require input_ids and a positive count of unmasked input tokens')
            loss, train_metrics = self.compute_loss(model, data, return_outputs=True)
            if torch.isnan(loss) or torch.isinf(loss):
                if self.train_options.get('fail_on_nonfinite', False):
                    raise FloatingPointError('Nonfinite training loss')
                print(f'\n-> NaN/Inf loss at step {ix}, skipping batch')
                self.optimizer.zero_grad()
                accumulated_batches = 0
                self.pending_input_tokens = 0
                self.pending_exposure = {}
                self.step += 1
                continue
            raw_loss = loss.detach().item()
            loss = loss / accum_iter
            if not self.compute_loss_backprop:
                # loss.backward() did not occur in compute_loss
                try:
                    loss.backward()
                except Exception as e:
                    if self.train_options.get('fail_on_backward_error', False):
                        raise
                    self.pending_input_tokens = 0
                    self.pending_exposure = {}
                    print(f'\n-> Backward error at step {ix}: {e}, skipping')
                    self.optimizer.zero_grad()
                    accumulated_batches = 0
                    self.step += 1
                    continue
            accumulated_batches += 1
            if isinstance(data, Mapping) and 'input_ids' in data:
                self.pending_input_tokens += batch_tokens
                if 'context_tasks' in data:
                    cell = f"{data['context_tasks'][0]}/{data['context_lengths'][0]}"
                    exposure = self.pending_exposure.setdefault(cell, dict(examples=0, tokens=0))
                    exposure['examples'] += 1
                    exposure['tokens'] += batch_tokens
            optimizer_stepped = False
            if accumulated_batches == accum_iter or ix + 1 == num_batches:
                optimizer_stepped = self._optimizer_step(accumulated_batches, accum_iter)
                accumulated_batches = 0
            
            self.step += 1
            total_loss += raw_loss
            successful_batches += 1
            mean_loss = total_loss / successful_batches
            desc = f"Training epoch {epoch} | loss_mean: {mean_loss:.3f} | loss_total: {raw_loss:.3f} | lr: {self.optimizer.param_groups[0]['lr']:.3e}"
            desc += f' | gradient step: {self.grad_step}'
            if self.max_input_tokens:
                desc += f' | input_tokens: {self.input_tokens}/{self.max_input_tokens}'
            for k, v in train_metrics.items():
                desc += f' | {k}: {v:.3f}'
            pbar.set_description(desc)

            # Logging
            if optimizer_stepped and self.grad_step % self.logging_steps == 0:
                self.train_metrics['train/loss'] = raw_loss
                self.train_metrics['train/loss_mean'] = mean_loss
                self.train_metrics['train/epoch'] = epoch
                self.train_metrics['train/step'] = self.grad_step
                self.train_metrics['train/input_tokens'] = self.input_tokens
                self.train_metrics['train/lr'] = self.optimizer.param_groups[0]['lr']
                for k, v in train_metrics.items():
                    self.train_metrics[f'train/{k}'] = v
                
                if self.wandb is not None:
                    self.wandb.log(self.train_metrics, step=self.grad_step)

            if self.eval_strategy == 'steps':
                if (self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and not eval_for_step):
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                elif self.grad_step == 0 and self.save_steps < 1000 and not eval_for_step:  # hack for micros
                    _eval_metrics = self.eval_step(model, step=self.grad_step)
                    print(f'Grad Step {self.grad_step} eval metrics:', _eval_metrics)
                    eval_for_step = True
                    model.train()  # Need to set back to train mode
                    
                elif self.grad_step % self.eval_steps == 0 and self.grad_step > 0 and eval_for_step:
                    pass
                else:
                    if self.grad_step > 0:
                        eval_for_step = False
            if ((self.max_steps > 0 and self.grad_step >= self.max_steps)
                    or (self.max_input_tokens and self.input_tokens >= self.max_input_tokens)):
                early_stopping = True
                return model, early_stopping
        
        early_stopping = False
        return model, early_stopping

    
    def save_latest_checkpoint(self, model):
        """Bounded weight snapshot; continuation uses a fresh optimizer."""
        if self._last_saved_grad_step == self.grad_step:
            return
        save_path = self.save_path + '/last_ckpt'
        save_checkpoint(model, self.tokenizer, save_path)
        import json
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)
        Path(save_path, 'training_progress.json').write_text(json.dumps(dict(
            input_tokens=self.input_tokens, optimizer_steps=self.grad_step,
            max_input_tokens=self.max_input_tokens, exposure=self.training_exposure,
            optimizer_state_saved=False), indent=2) + '\n')
        self._last_saved_grad_step = self.grad_step

    def eval_step(self, model: nn.Module, step: int = None, **kwargs: any) -> dict[any]:
        """
        Evaluation loop over one epoch
        """
        step = self.grad_step if step is None else step
        # Preserve trained weights even when accuracy stays tied at zero or
        # the allocation ends during lengthy generated validation.
        if self.max_input_tokens and self.grad_step > 0:
            self.save_latest_checkpoint(model)
        with torch.no_grad():
            self.eval_metrics = self.compute_eval_metrics(model, step=step, **kwargs)
            if self.metric_for_best_model not in self.eval_metrics:
                raise KeyError(
                    f'metric_for_best_model={self.metric_for_best_model!r} not in '
                    f'eval metrics {sorted(self.eval_metrics)}'
                )
            val_metric = self.eval_metrics[self.metric_for_best_model]
            if not math.isfinite(val_metric):
                raise ValueError(
                    f'Nonfinite checkpoint metric {self.metric_for_best_model}: '
                    f'{val_metric} at step {step}'
                )

            if self.max_input_tokens:
                import json
                from pathlib import Path
                progress = dict(input_tokens=self.input_tokens, max_input_tokens=self.max_input_tokens,
                                optimizer_steps=self.grad_step, exposure=self.training_exposure,
                                phase_max_length=getattr(getattr(self.train_loader, 'sampler', None),
                                                         'max_length', None))
                Path(self.save_path, 'training_progress.json').write_text(json.dumps(progress, indent=2))
            # Save results
            if self.wandb is not None:  # log to WandB
                self.wandb.log(self.eval_metrics, step=self.grad_step)

            if self.results_path is not None:  # log to local file
                self.eval_metrics_by_step['eval_step'].append(step)
                for k, v in self.eval_metrics.items():
                    if k not in self.eval_metrics_by_step:
                        self.eval_metrics_by_step[k] = [v]
                    else:
                        self.eval_metrics_by_step[k].append(v)
                # Inefficient, but log for experiments results
                pd.DataFrame(self.eval_metrics_by_step).to_csv(self.results_path)

            # Save best metric and checkpoint
            # Every evaluation is eligible, including step 0 and epoch-end
            # evaluations that do not coincide with eval_steps.
            if self.is_better(val_metric, self.best_val_metric):
                # Reuse one directory: full checkpoints are large.
                save_path = self.save_path + '/best_ckpt'
                save_checkpoint(model, self.tokenizer, save_path)
                self.best_val_checkpoint_path = save_path
                self.best_val_metric = val_metric
                self.best_val_metric_step = step
                print(f'\n-> Saved best model checkpoint to: {save_path} '
                      f'({self.metric_for_best_model}={val_metric:.6f}, step {step})!')

            if self.grad_step % self.save_steps == 0 and step > 0:

                save_path = self.save_path + '/' + self.type + '_' + str(step)
                save_checkpoint(model, self.tokenizer, save_path)
                print(f'\n-> Saved model checkpoint to: {save_path}!')
            
            if self.scheduler_step_after_epoch and self.scheduler is not None:
                self.scheduler.step(val_metric)
            return self.eval_metrics

    def compute_eval_metrics(self, 
                            model: nn.Module, step: int,
                            max_batches: int = None,
                            dataloader: DataLoader = None,
                            **kwargs: any) -> dict[any]:
        """
        One evaluation loop over a validation dataset
        """
        max_batches = (self.max_eval_batches if max_batches is None else max_batches)
        dataloader = self.eval_loader if dataloader is None else dataloader
        pbar = tqdm(dataloader, leave=False, colour='green', desc=f'Evaluating at step {step}')

        model.eval()
        step_loss = 0
        step_eval_metrics = {}
        with torch.no_grad():
            for ix, data in enumerate(pbar):
                loss, eval_metrics = self.compute_loss(model, data, return_outputs=True)
                if not self.compute_loss_backprop:
                    loss = loss.item()  # otherwise already float
                # The total loss goes under its own key. Filed under
                # metric_for_best_model it collided with stage 1's
                # 'eval/loss_ce', averaging CE with (1000*MSE + CE).
                for k, v in [('loss_total', loss), *eval_metrics.items()]:
                    step_eval_metrics.setdefault(f'eval/{k}', []).append(v)
                if data.get('context_tasks') == ['instruction'] and 'loss_ce' in eval_metrics:
                    step_eval_metrics.setdefault('eval/instruction_loss_ce', []).append(eval_metrics['loss_ce'])
                elif data.get('context_tasks') and 'loss_ce' in eval_metrics:
                    step_eval_metrics.setdefault('eval/synthetic_loss_ce', []).append(eval_metrics['loss_ce'])
                
                step_loss += loss
                desc = f"Evaluating at step {step} | loss: {step_loss / (ix + 1):.3f}"
                if self.optimizer is not None:
                    desc += f" | lr: {self.optimizer.param_groups[0]['lr']:.3e}"
                pbar.set_description(desc)
                if ix == max_batches:
                    break

            if self.train_options.get('generation_validation', False):
                from Training.long_context import retrieval_validation
                rows = dataloader.dataset.records
                print(f'\n-> Generated retrieval validation: step={step}, input_tokens={self.input_tokens}',
                      flush=True)
                step_eval_metrics.update({k: [v] for k, v in retrieval_validation(
                    model, self.tokenizer, rows,
                    int(self.train_options.get('validation_per_cell', 4)),
                    int(self.train_options.get('validation_max_new_tokens', 256)),
                    samples_path=os.path.join(self.save_path, 'validation_retrieval_samples.jsonl'),
                    step=step, input_tokens=self.input_tokens).items()})
            if 'accuracy' in str(self.metric_for_best_model) and self.metric_for_best_model not in step_eval_metrics:
                raise KeyError('Requested accuracy was not computed; refusing to select on a loss fallback')
            # Average over batches
            for k, v in step_eval_metrics.items():
                step_eval_metrics[k] = sum(v) / len(v)
            # Stage 2 selects on 'eval/loss', which compute_loss never returns
            # -- it relied on the total loss being filed here. Preserve that,
            # but only for metrics compute_loss does not already return.
            if (self.metric_for_best_model is not None
                    and self.metric_for_best_model not in step_eval_metrics):
                step_eval_metrics[self.metric_for_best_model] = \
                    step_eval_metrics['eval/loss_total']
            # ppl is averaged per batch above (mean of exp), which upper-bounds
            # exp(mean CE); report the consistent one alongside it.
            if 'eval/loss_ce' in step_eval_metrics:
                step_eval_metrics['eval/ppl_from_mean_ce'] = float(
                    torch.exp(torch.tensor(step_eval_metrics['eval/loss_ce'])))
            print(f'Eval step {step}:', step_eval_metrics)
            del loss
            torch.cuda.empty_cache()
        return step_eval_metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Attention transfer, with an LM term.

        Per-layer MSE alone is not a well-posed objective here: the teacher is
        computed from the student's own hidden states, so the target moves with
        the student. Both can converge somewhere far off the pretrained manifold
        while agreeing perfectly -- measured once as MSE down 941x while
        cross-entropy went from 6.4 (untrained) to 15.7 (worse than uniform).
        The cross-entropy term anchors it.
        """
        input_keys = {'input_ids', 'attention_mask'}
        data = {k: v.to(model.device) for k, v in inputs.items() if k in input_keys}
        outputs = model(**data, output_attentions=True)

        self.mse_factor = self.config.train.get('mse_factor', 1000)
        self.lm_weight = self.config.train.get('lm_loss_weight', 1.0)
        self.criterion_mse = nn.MSELoss(reduction='mean')

        loss_mse = 0
        n_layers = 0  # Number of layers to distill
        for layer_idx, attns in enumerate(outputs.attentions):
            if attns is not None:
                loss_mse += self.criterion_mse(attns[0].float(), attns[1].float())
                n_layers += 1
        if n_layers > 0:
            loss_mse = loss_mse / n_layers * self.mse_factor

        loss_ce = torch.tensor(0.0, device=model.device)
        if self.lm_weight > 0 and 'labels' in inputs:
            logits = outputs.get('logits')[..., :-1, :].contiguous()
            targets = inputs['labels'][..., 1:].contiguous().to(logits.device)
            loss_ce = self.criterion(
                logits.view(-1, logits.shape[-1]).float(), targets.view(-1)
            )

        loss = loss_mse + self.lm_weight * loss_ce
        metrics = {'loss_mse': loss_mse.item() if n_layers > 0 else 0.0,
                   'loss_ce': loss_ce.item(),
                   'ppl': torch.exp(loss_ce).item(),
                   'mse_factor': self.mse_factor}

        return (loss, metrics) if return_outputs else loss

    def init_checkpointing(self, config) -> None:
        self.save_path = config.train.output_dir
        self.best_val_checkpoint_path = config.train.output_dir
        if self.max_input_tokens:
            os.makedirs(self.save_path, exist_ok=True)
            self.results_path = os.path.join(self.save_path, 'eval_metrics.csv')

        # Best metric setup
        self.best_val_metric = float('-inf') if self.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        self.best_train_metric = 0 if self.greater_is_better else 1e10
        self.best_train_metric_epoch = 0
        self.best_train_metric_step = 0
        self.metric_for_best_model = self.metric_for_best_model
        if self.metric_for_best_model is not None:
            if 'eval' not in self.metric_for_best_model:
                self.metric_for_best_model = f'eval/{self.metric_for_best_model}'

class FinetuneTrainer(DefaultTrainer):
    def __init__(self, model, train_loader, eval_loader, args, optimizers, tokenizer, config):
        super().__init__(model, train_loader, eval_loader, args, optimizers, tokenizer, config)
        self.type = 'finetune'

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_keys = {'input_ids', 'attention_mask'}
        data = {k: v.to(model.device) for k, v in inputs.items() if k in input_keys}  
        labels = inputs.get('labels')
        if self.config.get('data', {}).get('name') == 'long_context':
            import inspect
            if labels.shape[0] != 1:
                raise ValueError('Answer-only long-context loss requires micro_batch_size=1')
            supervised = int((labels[0] != -100).sum())
            if supervised < 1 or (labels[0, -supervised:] == -100).any():
                raise ValueError('Expected one contiguous supervised answer suffix')
            base = model.get_base_model() if hasattr(model, 'get_base_model') else model
            parameters = inspect.signature(base.forward).parameters
            key = next((k for k in ('logits_to_keep', 'num_logits_to_keep') if k in parameters), None)
            if key is None:
                raise ValueError('Model lacks suffix-only LM-head support')
            with model_autocast(model):
                outputs = model(**data, output_attentions=False, use_cache=False,
                                **{key: supervised + 1})
            targets = labels[..., -supervised:].contiguous()
        else:
            # Multi-turn answer masks need full logits, but FP32 trainable
            # parameters still need autocast with the BF16 backbone.
            with model_autocast(model):
                outputs = model(**data, output_attentions=False, use_cache=False)
            targets = labels[..., 1:].contiguous()
        outputs = outputs.get('logits')[..., :-1, :].contiguous()
        # Flatten and compute cross-entropy loss
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1).to(outputs.device)
        loss = self.criterion(outputs.float(), targets)
        
        targets = targets.cpu()
        # 'loss_ce' so compute_eval_metrics also reports ppl_from_mean_ce: the
        # 'ppl' below is a mean of per-batch exp(CE), which Jensen-inflates above
        # exp(mean CE) -- and exp(mean CE) is what the eval reports, i.e. what
        # the recorded dot-path numbers (31, 18.1) are on.
        outputs = {'loss_ce': loss.item(), 'ppl': torch.exp(loss).item(),
                   'seq_len': data['input_ids'].shape[-1]}
        return (loss, outputs) if return_outputs else loss
