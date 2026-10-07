"""A TTT memory block with no transformers dependency, for MAD.

`LinearTTTAttention` carries rotary embeddings, a KV cache, PEFT hooks and the
HF decoder-layer contract, none of which MAD's trainer provides. The operator
underneath does not need any of it: `block_causal_lact_swiglu_l2` takes plain
tensors and returns [b, n, d]. This module is the thin wrapper that reaches it.

It reproduces the paths that matter for a recall study and deliberately drops
the rest:

  kept     q/k/v projections, the silu + scale/offset + l2 feature map, the
           per-token inner learning rate, momentum, Atlas retention, Muon, the
           sigmoid output gate, cross-block memory sharing
  dropped  rotary (MAD tasks are position-agnostic and its own blocks have no
           RoPE), the KV cache and decode path (MAD scores a single forward),
           the sliding-window attention branch -- a striped topology gets its
           locality from MAD's own attention blocks instead

Sharing follows the deployed architecture: one writer fits the memory, the
others read the trajectory it leaves, never the converged state, so a reader
cannot see tokens that come after it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from LinearTTT.model.LinearizeLlama.ttt_l2 import (
    block_causal_lact_swiglu_l2, read_lact_swiglu_l2,
)
from LinearTTT.model.LinearizeLlama.ttt_ops import inv_softplus, l2_norm


class TTTBlock(nn.Module):
    """[b, n, d] -> [b, n, d].

    Args mirror the yml keys so a MAD sweep and a Llama config describe the
    same memory: `dim` is the residual width, `heads` splits it into memories
    of `dim // heads`, `inter_multi` scales the SwiGLU hidden width, and
    `chunk_size` is how many tokens a single inner update sees.
    """

    def __init__(self, dim, heads=4, inter_multi=1.0, chunk_size=64,
                 base_lr=0.018, retention_init_bias=4.0, scale_init_bias=-3.0,
                 init_gain=1.0, use_muon=True, use_momentum=True):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must divide evenly into heads')
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.chunk_size = chunk_size
        self.use_muon, self.use_momentum = use_muon, use_momentum

        d_h = max(1, int(self.head_dim * inter_multi))
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.w0 = nn.Parameter(torch.randn(heads, d_h, self.head_dim)
                               / math.sqrt(self.head_dim) * init_gain)
        self.w1 = nn.Parameter(torch.randn(heads, self.head_dim, d_h)
                               / math.sqrt(d_h) * init_gain)
        self.w2 = nn.Parameter(torch.randn(heads, d_h, self.head_dim)
                               / math.sqrt(self.head_dim) * init_gain)

        self.lr_proj = nn.Linear(dim, 3 * heads)
        self.base_lr_inv = inv_softplus(base_lr)
        self.momentum_proj = nn.Sequential(nn.Linear(dim, heads), nn.Sigmoid())
        self.retention_proj = nn.Sequential(nn.Linear(dim, heads), nn.Sigmoid())
        self.gate_proj = nn.Linear(dim, dim)
        self.norm = nn.RMSNorm(self.head_dim) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(self.head_dim)

        self.qk_scale = nn.Parameter(torch.ones(2, dim))
        self.qk_offset = nn.Parameter(torch.zeros(2, dim))

        # The gate starts nearly closed so the residual stream survives step 0,
        # and retention near 1 so the memory is not erased between chunks:
        # sigmoid(4.0)^16 retains 0.75, sigmoid(0.0)^16 retains 1.5e-5.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, scale_init_bias)
        nn.init.zeros_(self.retention_proj[0].weight)
        nn.init.constant_(self.retention_proj[0].bias, retention_init_bias)

        self._share_src = None   # set by share_with(); a reader holds its writer

    def share_with(self, writer):
        """Read `writer`'s memory instead of fitting one.

        The reader keeps its own q, gate, norm and output projection -- only
        the fast weights are shared, which is what makes the state cheap.
        """
        if writer is self:
            raise ValueError('A block cannot share with itself')
        self._share_src = [writer]            # list, to stay out of _modules
        return self

    def _features(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = q * self.qk_scale[0] + self.qk_offset[0]
        k = k * self.qk_scale[1] + self.qk_offset[1]
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads)
                   for t in (q, k, v))
        return l2_norm(q), l2_norm(k), l2_norm(v)

    def _coefficients(self, x):
        lr = F.softplus(self.lr_proj(x) + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(lr, 'b n (h lrs d) -> lrs (b h) n d',
                                  lrs=3, h=self.heads, d=1)
        shape = lambda p: rearrange(p(x).float(), 'b n (h d) -> (b h) n d', h=self.heads)
        momentum = shape(self.momentum_proj) if self.use_momentum else None
        return lr0, lr1, lr2, momentum, shape(self.retention_proj)

    def forward(self, x, **_):
        b = x.shape[0]
        q, k, v = self._features(x)
        lr0, lr1, lr2, momentum, retention = self._coefficients(x)

        if self._share_src is None:
            readout, self._trajectory = self._write(
                q, k, v, lr0, lr1, lr2, momentum, retention)
        else:
            # A reader uses its OWN query against the writer's trajectory. The
            # trajectory holds the state BEFORE each chunk was folded in, so a
            # reader never sees tokens that follow its own position.
            writer = self._share_src[0]
            if getattr(writer, '_trajectory', None) is None:
                raise RuntimeError('The writer must run before its readers')
            readout = read_lact_swiglu_l2(writer._trajectory, q, self.chunk_size)

        readout = self.norm(readout)
        readout = rearrange(readout, '(b h) n d -> b n (h d)', b=b)
        return self.out(readout * torch.sigmoid(self.gate_proj(x)))

    def _write(self, q, k, v, lr0, lr1, lr2, momentum, retention):
        # The operator fits one memory per (batch element, head), so the
        # parameters -- stored once per head -- are repeated to [b*h, ...] and
        # promoted to FP32, matching LinearizeLlama.py:811. Every sequence in
        # the batch starts from the same learned initial memory.
        b = q.shape[0] // self.heads
        w0, w1, w2 = (w.repeat(b, 1, 1).float() for w in (self.w0, self.w1, self.w2))
        out = block_causal_lact_swiglu_l2(
            w0, w1, w2, q, k, v, lr0, lr1, lr2,
            chunk_size=self.chunk_size, use_muon=self.use_muon,
            momentum=momentum, retention=retention, return_trajectory=True,
        )
        return out[0], out[-1]
