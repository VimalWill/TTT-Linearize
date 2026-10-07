"""SDPA attention with MAD's interface, so MAD imports without flash-attn.

mad/model/layers/__init__.py does `from flash_attn.modules.mha import MHA` at
import time, so the layer registry cannot be reached at all on a machine
without flash-attn -- which has no aarch64 wheels and is absent from the
GH200 nodes this project runs on.

This is a drop-in replacement with the same constructor keys and the same
[b, n, d] -> [b, n, d] contract. MAD's models are two blocks at sequence
lengths up to 1280, so a dense mask and torch's SDPA cost nothing worth
optimising, and SDPA is exact against flash-attn rather than an
approximation.

Supported from their signature: dim, causal, n_heads, rotary_emb_dim (partial
rotary, as their configs use 8 of 128), dropout, window_size as flash-attn's
(left, right) pair. The rest are accepted and ignored, which is what their
own wrapper effectively does for a model this size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class SDPAAttention(nn.Module):
    def __init__(self, dim: int, causal: bool = True, n_heads: int = 16,
                 rotary_emb_dim: float = 0.0, dropout: float = 0.0,
                 window_size=(-1, -1), rotary_emb_base: float = 10000.0,
                 qkv_proj_bias: bool = True, out_proj_bias: bool = True,
                 softmax_scale: float = None, **kwargs):
        super().__init__()
        if dim % n_heads:
            raise ValueError('dim must divide evenly into n_heads')
        self.dim, self.n_heads = dim, n_heads
        self.head_dim = dim // n_heads
        self.causal, self.dropout = causal, dropout
        self.scale = softmax_scale
        self.window = tuple(window_size)

        # Partial rotary, as their configs use (8 of 128 dimensions). The
        # rotated block must be even for the half-split to pair up.
        self.rotary_dim = int(rotary_emb_dim)
        if self.rotary_dim % 2:
            raise ValueError('rotary_emb_dim must be even')
        if self.rotary_dim > self.head_dim:
            raise ValueError('rotary_emb_dim cannot exceed the head dimension')
        if self.rotary_dim:
            inv = 1.0 / (rotary_emb_base ** (
                torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
            self.register_buffer('inv_freq', inv, persistent=False)

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_proj_bias)
        self.out_proj = nn.Linear(dim, dim, bias=out_proj_bias)

    def _apply_rotary(self, x, positions):
        """Rotate the leading `rotary_dim` channels, pass the rest through."""
        if not self.rotary_dim:
            return x
        rot, keep = x[..., :self.rotary_dim], x[..., self.rotary_dim:]
        freqs = torch.outer(positions.float(), self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        rot = rot * cos + _rotate_half(rot) * sin
        return torch.cat((rot, keep), dim=-1)

    def _mask(self, n, device):
        """None when plain causal SDPA already expresses the constraint.

        window_size is flash-attn's (left, right): a query attends to keys
        from i-left to i+right, and -1 means unbounded on that side.
        """
        left, right = self.window
        unbounded = left < 0 and (right < 0 or self.causal)
        if unbounded:
            return None
        i = torch.arange(n, device=device)
        delta = i[:, None] - i[None, :]
        allowed = torch.ones(n, n, dtype=torch.bool, device=device)
        if self.causal or right >= 0:
            allowed &= delta >= 0 if self.causal else allowed
        if left >= 0:
            allowed &= delta <= left
        if right >= 0 and not self.causal:
            allowed &= delta >= -right
        return allowed

    def forward(self, x, **_):
        b, n, _ = x.shape
        qkv = self.Wqkv(x).view(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [b, h, n, d]

        positions = torch.arange(n, device=x.device)
        q, k = self._apply_rotary(q, positions), self._apply_rotary(k, positions)

        mask = self._mask(n, x.device)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if mask is None else mask[None, None],
            is_causal=self.causal and mask is None,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        return self.out_proj(out.transpose(1, 2).reshape(b, n, self.dim))
