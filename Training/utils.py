import numpy as np
import torch
import torch.optim

def get_optimizer_and_scheduler(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.train.lr,
                                 weight_decay=config.train.get('weight_decay', 0.01),
                                 fused=torch.cuda.is_available())
    schedule = config.train.get('lr_scheduler', 'plateau')
    if schedule == 'token_linear':
        scheduler = TokenLinearScheduler(optimizer, int(config.train.max_input_tokens),
                                         int(config.train.get('warmup_tokens', 0)),
                                         float(config.train.get('min_lr', 0)))
    elif schedule == 'linear':
        max_steps = int(config.train.get('max_steps', -1))
        if max_steps <= 0:
            raise ValueError('lr_scheduler: linear requires positive max_steps')
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / max_steps))
    elif schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            min_lr=0.00001
        )
    else:
        raise ValueError(f'Unknown lr_scheduler: {schedule!r}')
    return optimizer, scheduler

def count_model_params(model, requires_grad: bool = True):
    # code form lolcats
    """
    Return total # of trainable parameters
    """
    if requires_grad:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    try:
        return sum([np.prod(p.size()) for p in model_parameters]).item()
    except:
        return sum([np.prod(p.size()) for p in model_parameters])


class TokenLinearScheduler:
    """Warmup/decay by tokens in successful optimizer updates, not examples."""
    def __init__(self, optimizer, total_tokens, warmup_tokens=0, min_lr=0):
        if total_tokens < 1 or not 0 <= warmup_tokens < total_tokens:
            raise ValueError('Invalid token schedule budget/warmup')
        self.optimizer, self.total_tokens, self.warmup_tokens = optimizer, total_tokens, warmup_tokens
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        if not 0 <= min_lr <= min(self.base_lrs):
            raise ValueError('min_lr must lie between zero and the initial learning rate')
        self.min_lr, self.tokens = min_lr, 0
        self.step_tokens(0)

    def step_tokens(self, tokens):
        self.tokens = tokens
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if self.warmup_tokens and tokens < self.warmup_tokens:
                lr = base * tokens / self.warmup_tokens
            else:
                remaining = max(0, self.total_tokens - tokens)
                lr = self.min_lr + (base - self.min_lr) * remaining / (self.total_tokens - self.warmup_tokens)
            group['lr'] = lr

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return dict(tokens=self.tokens)

    def load_state_dict(self, state):
        self.step_tokens(state['tokens'])
