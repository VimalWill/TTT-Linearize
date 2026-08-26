# -*- coding: utf-8 -*-
"""Loader for the LaCT test-time-training operator.

`third_party/LaCT` is a git submodule with no package `__init__.py` chain, so it
cannot be imported by name. `lact_llm/lact_model/ttt_operation.py` only depends
on torch (unlike `minimal_implementations/`, which imports flash_attn at module
scope), so we load that file directly by path.
"""

import importlib.util
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TTT_OPERATION = os.path.abspath(
    os.path.join(_HERE, '..', '..', '..',
                 'third_party', 'LaCT', 'lact_llm', 'lact_model', 'ttt_operation.py')
)

if not os.path.exists(_TTT_OPERATION):
    raise ImportError(
        f'LaCT operator not found at {_TTT_OPERATION}. '
        'Run `git submodule update --init third_party/LaCT`.'
    )

_spec = importlib.util.spec_from_file_location('lact_ttt_operation', _TTT_OPERATION)
_mod = importlib.util.module_from_spec(_spec)
# Register before exec: the operator is torch.compile'd, and when Dynamo traces
# it, LOAD_GLOBAL resolves the function's __module__ through
# importlib.import_module(). A module loaded only by path is not importable by
# name, so tracing fails with ModuleNotFoundError.
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

block_causal_lact_swiglu = _mod.block_causal_lact_swiglu
prenorm_block_causal_lact_swiglu = _mod.prenorm_block_causal_lact_swiglu
l2_norm = _mod.l2_norm


def inv_softplus(x):
    """Inverse of softplus, used to initialise the per-token learning rate."""
    if hasattr(x, 'log'):
        return x + (-(-x).expm1()).log()
    return x + math.log(-math.expm1(-x))


__all__ = [
    'block_causal_lact_swiglu',
    'prenorm_block_causal_lact_swiglu',
    'l2_norm',
    'inv_softplus',
]
