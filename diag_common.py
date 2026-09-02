"""Shared loading for the diagnostics.

This lived as a copy-pasted `load_model` in diag_ce.py and diag_gate.py, and the
same silent bug was therefore present in both: `ttt_params.pt` is written from a
PeftModel, so its keys carry peft's wrapper prefix, and loading it into the
still-unwrapped model matches nothing while `strict=False` hides that. Any
measurement of a stage-2 checkpoint was really measuring stage-1 TTT weights
with stage-2 adapters.
"""

import os
import re

import torch
from transformers import AutoModelForCausalLM

_PEFT_PREFIX = re.compile(r'^base_model\.model\.')


def load_ttt_params(model, adapter, verbose=True):
    """Overlay `adapter/ttt_params.pt` onto an *unwrapped* model.

    Raises if nothing matched, which is the failure `strict=False` would
    otherwise swallow. Returns the number of tensors applied.
    """
    ttt = os.path.join(adapter, 'ttt_params.pt')
    if not os.path.exists(ttt):
        if verbose:
            print('  NOTE: no ttt_params.pt -- stage-2 TTT weights were never saved')
        return 0

    sd = torch.load(ttt, map_location='cpu')
    sd = {_PEFT_PREFIX.sub('', k): v for k, v in sd.items()}
    unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
    matched = len(sd) - len(unexpected)
    if matched == 0:
        raise RuntimeError(
            f'{ttt} holds {len(sd)} tensors but none match this model. '
            f'First saved key after prefix strip: {next(iter(sd))}'
        )
    if verbose:
        print(f'  overlaid {matched}/{len(sd)} saved TTT tensors')
        if unexpected:
            print(f'  WARNING: {len(unexpected)} unmatched, e.g. {unexpected[0]}')
    return matched


def load_model(path, model_config, adapter=None, verbose=True):
    """Full checkpoint, optionally with PEFT adapters and saved TTT params on top."""
    model = AutoModelForCausalLM.from_pretrained(
        path, config=model_config, device_map={'': 0}
    ).to(torch.bfloat16)
    if adapter:
        from peft import PeftModel
        load_ttt_params(model, adapter, verbose=verbose)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval()


__all__ = ['load_model', 'load_ttt_params']
