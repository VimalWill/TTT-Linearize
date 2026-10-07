"""Make the TTT memory available to MAD without editing the submodule.

mad.registry is a plain dict, so registration is an insertion. The config
path is absolute, which os.path.join in mad.configs passes through unchanged,
so our defaults live in this tree rather than in third_party.

Sharing needs a second step. MAD builds layers independently -- it has no
notion of one layer reading another's state -- but LanguageModel runs them in
order inside a single forward, so a writer placed before its readers has
already left its trajectory by the time they run. `link_shared_memories`
walks the built model and wires that up.
"""
import os
import sys
import types

import torch.nn as nn

from LinearTTT.mad.attention import SDPAAttention
from LinearTTT.mad.ttt_block import TTTBlock

_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ttt.yml')


def stub_flash_attn():
    """Let `import mad` succeed on a machine without flash-attn.

    mad/model/layers/__init__.py does `from flash_attn.modules.mha import MHA`
    at import time, so the registry is unreachable without it -- and
    flash-attn has no aarch64 wheels, so it cannot be installed on the GH200
    nodes at all. The stub only has to be subclassable, because
    register_ttt() replaces every attention entry with SDPAAttention before
    anything is constructed.
    """
    if 'flash_attn' in sys.modules:
        return False
    for name in ('flash_attn', 'flash_attn.modules', 'flash_attn.modules.mha'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules['flash_attn.modules.mha'].MHA = nn.Module
    return True


def register_ttt(name='ttt', shorthand='T', replace_attention=True):
    """Insert the TTT block into MAD's layer registry. Idempotent.

    Also swaps MAD's flash-attn Attention for the SDPA one by default. The
    two are mathematically the same attention; only the kernel differs, and
    MAD's models are two blocks at sequence lengths up to 1280, where the
    kernel choice is not worth a dependency that will not install.
    """
    stub_flash_attn()
    from mad.registry import layer_registry
    layer_registry[name] = {'module': TTTBlock, 'cfg': _CFG, 'shorthand': shorthand}
    if replace_attention:
        for key, entry in layer_registry.items():
            if 'attention' in key and 'linear' not in key:
                entry['module'] = SDPAAttention
    return layer_registry


def ttt_blocks(model):
    """Every TTTBlock in a built MAD model, in execution order."""
    return [m for m in model.modules() if isinstance(m, TTTBlock)]


def link_shared_memories(model, groups):
    """Tie TTT blocks into writer/reader groups.

    `groups` indexes into the TTT blocks only, not into all layers, so a
    striped stack of [ttt, mlp, ttt, mlp, ttt, mlp, attention, mlp] has TTT
    blocks 0, 1, 2 and the group [[0, 1, 2]] makes block 0 the writer.

    The first index in each group writes; the rest read its trajectory. A
    reader must come after its writer, since the trajectory only exists once
    the writer has run.
    """
    blocks = ttt_blocks(model)
    seen = set()
    for group in groups:
        if len(group) < 2:
            raise ValueError('A share group needs a writer and at least one reader')
        if min(group) != group[0]:
            raise ValueError(f'The writer must come first in {group}; readers run after it')
        for index in group:
            if index in seen:
                raise ValueError(f'TTT block {index} is in more than one group')
            if not 0 <= index < len(blocks):
                raise ValueError(f'No TTT block {index}; the model has {len(blocks)}')
            seen.add(index)
        writer = blocks[group[0]]
        for index in group[1:]:
            blocks[index].share_with(writer)
    return blocks


def total_state_dim(model):
    """Total fixed state across the stack -- the quantity MAD normalises.

    The paper holds architectures to a common total state dimension (4096) so
    a comparison is not confounded by state size. Readers contribute nothing.
    """
    return sum(b.state_dim() for b in ttt_blocks(model))
