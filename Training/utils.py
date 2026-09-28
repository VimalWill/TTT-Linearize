import numpy as np
import torch
import torch.optim

def get_optimizer_and_scheduler(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.train.lr,
                                 weight_decay=config.train.get('weight_decay', 0.01),
                                 fused=torch.cuda.is_available())
    schedule = config.train.get('lr_scheduler', 'plateau')
    if schedule == 'linear':
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
