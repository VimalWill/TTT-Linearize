"""Fused local attention and TTT, plus conversion of a trained MAD stack."""
import torch
import torch.nn.functional as F

from .attention import SDPAAttention
from .ttt_block import TTTBlock


class TTTHybridBlock(TTTBlock):
    def __init__(self, dim, num_heads=4, window_size=512, rotary_emb_dim=0,
                 rotary_emb_base=10000., softmax_scale=None, **kwargs):
        super().__init__(dim, num_heads=num_heads, **kwargs)
        if type(window_size) is not int or window_size < self.chunk_size:
            raise ValueError('Hybrid window_size must be an integer >= chunk_size')
        self.attention = SDPAAttention(
            dim, n_heads=num_heads, rotary_emb_dim=rotary_emb_dim,
            rotary_emb_base=rotary_emb_base, softmax_scale=softmax_scale,
            window_size=(window_size, 0), qkv_proj_bias=False, out_proj_bias=False)
        self.attention.Wqkv = self.qkv
        self.attention.out_proj = self.out
        self.collect_distillation = False
        self.distillation_loss = None

    def forward(self, x, **_):
        self.distillation_loss = None
        projected = self.qkv(x).chunk(3, dim=-1)
        local = self.attention.attention_output(projected)
        combined = local + self._memory_output(x, projected)
        if self.collect_distillation:
            # Stage-1 teacher: full causal attention on the same hidden states
            # and frozen projections, matching the deployed attention transfer.
            with torch.no_grad():
                teacher = self.attention.attention_output(
                    tuple(t.detach() for t in projected), window=(-1, -1))
            self.distillation_loss = F.mse_loss(combined.float(), teacher.float())
        return self.out(combined)


def hybridize_mad_model(model, hybrid_window=512, swa_window=4096,
                        chunk_size=64, **ttt_kwargs):
    """Convert a task-trained SDPA MAD model to three hybrid + one SWA.

    Attention slots, rather than intervening MLPs, define the four-layer cycle.
    Call on a fresh/deep-copied teacher model before constructing the optimizer.
    The returned model retains the teacher's embeddings, MLPs and projections.
    """
    if swa_window < 1:
        raise ValueError('swa_window must be positive')
    if any(isinstance(m, TTTBlock) for m in model.modules()):
        raise ValueError('Convert a pure attention model, not an existing TTT stack')
    slots = [(name, module) for name, module in model.named_modules()
             if isinstance(module, SDPAAttention)]
    if not slots or len(slots) % 4:
        raise ValueError('The number of attention slots must be a positive multiple of four')
    for _, attention in slots:
        if not attention.causal or attention.dropout:
            raise ValueError('Conversion requires causal attention with zero dropout')
    writer = None
    for index, (name, attention) in enumerate(slots):
        if index % 4 == 3:
            attention.window = (swa_window, 0)
            continue
        block = TTTHybridBlock(
            attention.dim, num_heads=attention.n_heads, chunk_size=chunk_size,
            window_size=hybrid_window, rotary_emb_dim=attention.rotary_dim,
            softmax_scale=attention.scale, **ttt_kwargs)
        block.to(device=attention.Wqkv.weight.device, dtype=attention.Wqkv.weight.dtype)
        block.qkv, block.out = attention.Wqkv, attention.out_proj
        # Preserve the original rotary frequencies as well as projections.
        block.attention = attention
        attention.window = (hybrid_window, 0)
        if index % 4 == 0:
            writer = block
        else:
            block.share_with(writer)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, block)
    return model
