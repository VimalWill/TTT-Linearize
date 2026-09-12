import math
import warnings
import copy
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
)

from transformers.utils import logging, add_start_docstrings_to_model_forward
from transformers.utils import is_flash_attn_2_available

# packages to integrate test-time training
from .ttt_ops import (
    block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu,
    l2_norm,
    inv_softplus,
)
from .ttt_l2 import block_causal_lact_swiglu_l2, read_lact_swiglu_l2

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    print("flash_attn_2 is not available")

from .cache import TTTCache

from .Configuration import LigerGLAConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

logger = logging.get_logger(__name__)

# Must be compiled: called bare, flex_attention falls back to `math_attention`,
# which materialises the whole [B, H, Q, KV] score matrix and masks afterwards.
_flex_attention_compiled = None


def _compiled_flex_attention():
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled


# Fallback for varying sequence lengths (lm-eval), where dynamo stops lowering
# the call and flex drops to a dense [B, H, Q, KV]. Peak B*H*C*(C+W).
_force_sdpa_window = False


def use_sdpa_sliding_window(enable: bool = True):
    """Route sliding_window_attention through the chunked SDPA path."""
    global _force_sdpa_window
    _force_sdpa_window = enable


def _sdpa_sliding_window(q, k, v, window_size, causal, scale, chunk=1024):
    """Chunked softmax attention over a sliding window. [B, H, T, D] in and out.

    matmul + softmax rather than SDPA, whose GQA validation rejects these
    slices. Peak is B*H*C*(C+W).
    """
    B, H, T, D = q.shape
    if k.shape[1] != H:                    # GQA: broadcast kv heads up to q heads
        k = repeat_kv(k, H // k.shape[1])
        v = repeat_kv(v, H // v.shape[1])
    if scale is None:
        scale = D ** -0.5
    out = torch.empty_like(q)
    neg = torch.finfo(q.dtype).min
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lo = max(0, s - window_size)
        hi = e if causal else min(T, e + window_size)
        qi = torch.arange(s, e, device=q.device).unsqueeze(1)
        ki = torch.arange(lo, hi, device=q.device).unsqueeze(0)
        if causal:
            m = (qi >= ki) & (qi - ki <= window_size)
        else:
            m = (qi - ki).abs() <= window_size
        att = torch.matmul(q[:, :, s:e], k[:, :, lo:hi].transpose(-1, -2)) * scale
        att = att.masked_fill(~m.view(1, 1, e - s, hi - lo), neg)
        att = torch.softmax(att.float(), dim=-1).to(q.dtype)
        out[:, :, s:e] = torch.matmul(att, v[:, :, lo:hi])
    return out


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 1024,
    causal: bool = True,
    scale: float = None,
    block_mask_cache: dict = None
) -> torch.Tensor:

    B, H, Q_len, D = q.shape
    KV_len = k.shape[2]
    device = q.device

    # Non-square (decode with rolling KV buffer): all KV tokens are already
    # causally valid and within the window (trimmed before calling), so no mask needed.
    if Q_len != KV_len:
        return _compiled_flex_attention()(q, k, v, scale=scale)

    # apply_rotary_pos_emb BROADCASTS: under generate() q/k can expand to the
    # prefix length while v does not, and Q_len == KV_len still passes.
    if v.shape[2] != KV_len:
        raise RuntimeError(
            f'v has {v.shape[2]} positions but k has {KV_len}: q/k were '
            'broadcast by the rotary embedding while v was not. This model has '
            'no incremental-decode path -- run generation with use_cache=False.'
        )

    if _force_sdpa_window or q.device.type != 'cuda':
        return _sdpa_sliding_window(q, k, v, window_size, causal, scale)

    if causal:
        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= window_size
            return causal_mask & window_mask
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= window_size

    cache_key = (Q_len, window_size, causal, device)
    if block_mask_cache is not None and cache_key in block_mask_cache:
        block_mask = block_mask_cache[cache_key]
    else:
        # B=H=None keeps the mask [1, 1, Q, Q]; passing them materialises the
        # dense per-head mask, 256 GiB at 32k.
        try:
            block_mask = create_block_mask(
                mask_mod, None, None, Q_len, Q_len, device=device, _compile=True)
        except TypeError:  # older torch without _compile
            block_mask = create_block_mask(
                mask_mod, None, None, Q_len, Q_len, device=device)
        if block_mask_cache is not None:
            block_mask_cache[cache_key] = block_mask

    output = _compiled_flex_attention()(q, k, v, block_mask=block_mask, scale=scale)
    return output

class LinearTTTAttention(nn.Module):

    def __init__(
        self,
        config: LigerGLAConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.inner_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.num_ttt_heads = getattr(config, 'num_ttt_heads', None) or self.num_heads
        if self.inner_dim % self.num_ttt_heads != 0:
            raise ValueError(
                f'inner_dim {self.inner_dim} is not divisible by num_ttt_heads {self.num_ttt_heads}'
            )
        self.ttt_head_dim = self.inner_dim // self.num_ttt_heads
        if self.num_key_value_groups > 1 and self.num_ttt_heads < self.num_heads:
          
            logger.warning_once(
                f'num_ttt_heads={self.num_ttt_heads} < num_attention_heads={self.num_heads} under GQA '
                f'(groups={self.num_key_value_groups}): TTT keys/values will contain duplicated channels.'
            )

        self.lact_chunk_size = getattr(config, 'lact_chunk_size', 512)
        self.window_size = getattr(config, 'window_size', self.lact_chunk_size)
        if self.window_size < self.lact_chunk_size:
            raise ValueError(
                f'window_size ({self.window_size}) must be >= lact_chunk_size ({self.lact_chunk_size}); '
                'the fast weights are frozen within a chunk, so the attention window has to span it.'
            )
        self.ttt_use_muon = getattr(config, 'ttt_use_muon', False)
        self.ttt_use_momentum = getattr(config, 'ttt_use_momentum', True)
        self.ttt_prenorm = getattr(config, 'ttt_prenorm', False)

        # 'dot' = LaCT Eq. 7 + Eq. 8 renorm; 'l2' = Atlas Eq. 9 + Eq. 32
        # retention gate. The pairings are not interchangeable -- see ttt_l2.py.
        self.ttt_inner_loss = getattr(config, 'ttt_inner_loss', 'dot')
        if self.ttt_inner_loss not in ('dot', 'l2'):
            raise ValueError(
                f"ttt_inner_loss must be 'dot' or 'l2', got {self.ttt_inner_loss!r}"
            )
        if self.ttt_inner_loss == 'l2' and self.ttt_prenorm:
            raise NotImplementedError(
                "ttt_inner_loss='l2' has no prenorm variant; set ttt_prenorm=False."
            )

        # State is proportional to ttt_inter_multi -- the knob for allocating
        # capacity across layers. Scalar (uniform) or a per-layer list.
        inter = getattr(config, 'ttt_inter_multi', 1.0)
        if isinstance(inter, (list, tuple)):
            if layer_idx is None:
                raise ValueError('per-layer ttt_inter_multi needs a layer_idx')
            if len(inter) != config.num_hidden_layers:
                raise ValueError(
                    f'ttt_inter_multi has {len(inter)} entries but the model has '
                    f'{config.num_hidden_layers} layers'
                )
            inter = inter[layer_idx]
        self.ttt_inter_multi = float(inter)

        d_in = d_out = self.ttt_head_dim
        d_h = int(self.ttt_head_dim * self.ttt_inter_multi)
        if d_h < 1:
            raise ValueError(
                f'ttt_inter_multi={self.ttt_inter_multi} gives d_h={d_h} at '
                f'ttt_head_dim={self.ttt_head_dim}'
            )
        gain = getattr(config, 'fw_init_gain', 0.5)
       
        self.w0 = nn.Parameter(torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in) * gain)
        self.w1 = nn.Parameter(torch.randn(self.num_ttt_heads, d_out, d_h) / math.sqrt(d_h) * gain)
        self.w2 = nn.Parameter(torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in) * gain)

        # per-token, per-head inner-loop learning rate (one scalar per fast weight)
        self.lr_proj = nn.Linear(self.hidden_size, 3 * self.num_ttt_heads)
        self.base_lr_inv = inv_softplus(getattr(config, 'ttt_base_lr', 1e-2))

        if self.ttt_use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_ttt_heads),
                nn.Sigmoid(),
            )

        # Atlas Eq. 32's alpha_t. Only the l2 path uses it; the dot path keeps
        # Eq. 8's renormalisation, which this would double up on.
        self.ttt_retention_init_bias = getattr(config, 'ttt_retention_init_bias', 4.0)
        if self.ttt_inner_loss == 'l2':
            self.retention_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.num_ttt_heads),
                nn.Sigmoid(),
            )

       
        self.ttt_qk_scale = nn.Parameter(torch.ones(2, self.inner_dim))
        self.ttt_qk_offset = nn.Parameter(torch.zeros(2, self.inner_dim))

        self.ttt_norm = LlamaRMSNorm(self.ttt_head_dim, eps=config.rms_norm_eps)
        self.ttt_scale_proj = nn.Linear(self.hidden_size, self.num_ttt_heads)
        self.ttt_scale_init_bias = getattr(config, 'ttt_scale_init_bias', 0.1)
        self.fw_init_gain = gain

        # Incremental decode cache: the converged fast weights plus a rolling
        # k/v window, set at the end of prefill. None means "not decoding".
        self._share_gid = None
        self._share_leader = True    # meaningless unless _share_gid is set
        self._share_size = 1
        self._share_leader_idx = None
        # Shared-memory group, one writer and N readers. The leader runs the
        # inner loop and publishes its per-chunk trajectory; the other members
        # read block c with their own q, which is causal because block c is the
        # state before chunk c was folded in. Sharing the FINAL state instead
        # would hand every reader a memory fitted on the whole sequence.
        groups = getattr(config, 'ttt_share_groups', None) or []
        for gid, g in enumerate(groups):
            if layer_idx in g:
                if self.ttt_inner_loss != 'l2':
                    raise NotImplementedError(
                        'ttt_share_groups needs the l2 operator: upstream\'s '
                        'dot-product operator returns only the output, not the '
                        'converged fast weights, so there is no state to hand to '
                        "the next layer. Set ttt_inner_loss: 'l2'."
                    )
                if self.ttt_prenorm:
                    raise NotImplementedError('ttt_share_groups with ttt_prenorm')
                self._share_gid = gid
                self._share_leader = (layer_idx == min(g))
                self._share_leader_idx = min(g)
                self._share_size = len(g)
                break

        # reset_ttt_parameters reads _share_gid/_share_leader, so the share
        # group must be resolved BEFORE it runs.
        self.reset_ttt_parameters()

        self._block_mask_cache = {}

        # Branch ablation for the diagnostics; the branches are summed in
        # forward, so a module hook cannot separate them.
        self._ablate_ttt = False
        self._ablate_attn = False

    def reset_ttt_parameters(self):
        """Initialise the parameters with no counterpart in the checkpoint.

        Re-run from _init_weights: post_init() re-randomises every nn.Linear,
        including the output gate that must start nearly closed.
        """
        d_in, d_h = self.ttt_head_dim, self.w0.shape[1]
        # A non-leader's w0/w1/w2 alias its leader's, and follower keys come
        # back missing from a shared checkpoint so _init_weights runs on them.
        # Without this guard it overwrites the leader's loaded values.
        own_fast_weights = self._share_gid is None or self._share_leader
        with torch.no_grad():
            if own_fast_weights:
                self.w0.normal_(0, 1 / math.sqrt(d_in)).mul_(self.fw_init_gain)
                self.w1.normal_(0, 1 / math.sqrt(d_h)).mul_(self.fw_init_gain)
                self.w2.normal_(0, 1 / math.sqrt(d_in)).mul_(self.fw_init_gain)
            self.ttt_qk_scale.fill_(1.0)
            self.ttt_qk_offset.zero_()
            # No counterpart in the Llama checkpoint, and _init_weights handles
            # Linear/Embedding only -- uninitialised under the meta-device path.
            self.ttt_norm.weight.fill_(1.0)
        # Nearly closed so the host's residual stream survives step 0, but not
        # fully: silu(0) == 0 would cut the gradient to lr_proj and the weights.
        nn.init.zeros_(self.ttt_scale_proj.weight)
        nn.init.constant_(self.ttt_scale_proj.bias, self.ttt_scale_init_bias)
        # Near 1 so the memory survives between chunks: alpha^16 is 0.75 at
        # sigmoid(4.0), 1.5e-5 at sigmoid(0.0). PRIVATE exponent -- see above.
        if hasattr(self, 'retention_proj'):
            nn.init.zeros_(self.retention_proj[0].weight)
            nn.init.constant_(self.retention_proj[0].bias,
                              self.ttt_retention_init_bias)

    def _ttt_features(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """[b, n, inner_dim] -> [b * num_ttt_heads, n, ttt_head_dim]."""
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = q * self.ttt_qk_scale[0] + self.ttt_qk_offset[0]
        k = k * self.ttt_qk_scale[1] + self.ttt_qk_offset[1]
        q, k, v = (
            rearrange(x, 'b n (h d) -> (b h) n d', h=self.num_ttt_heads) for x in (q, k, v)
        )
        if self.ttt_inner_loss == 'l2':
            # A regression bias fits ||v|| itself, which varies ~30x across
            # heads, so one inner lr cannot serve them all. Output magnitude is
            # set downstream by ttt_norm + ttt_scale_proj anyway.
            v = l2_norm(v)
        return l2_norm(q), l2_norm(k), v

    def _decode_step(self, hidden_states, position_ids, position_embeddings,
                     past_key_value, bsz, q_len):
        """One generated token, reusing the state parked by prefill.

        Returns (output, aux, cache, trajectory); decode needs no trajectory.
        """
        st = past_key_value.states[self.layer_idx]
        if q_len != 1:
            raise NotImplementedError(
                f'decode expects one token per step, got {q_len}. Greedy and '
                'sampling both feed one; beam search and prompt chunking do not.'
            )

        # A reader carries no memory: it borrows the leader's, which is the
        # decode-time equivalent of reading its trajectory. The leader is the
        # lowest index in the group, so it has already stepped this token --
        # including any deferred chunk update -- by the time we get here.
        reader = self._share_gid is not None and not self._share_leader
        if reader:
            src = past_key_value.states.get(self._share_leader_idx)
            if src is None:
                raise RuntimeError(
                    f'layer {self.layer_idx} is decoding but its share-group '
                    'leader has no parked state; prefill both together.'
                )
        else:
            src = st
            # Apply first: chunk i reads weights fitted on chunks 0..i-1. The
            # buffer can arrive full out of prefill, so checking BEFORE the
            # readout is what matches a full-sequence forward.
            if (st['k_buf'] is not None
                    and st['k_buf'].shape[1] >= self.lact_chunk_size):
                self._apply_deferred_chunk(st)

        q = rearrange(self.q_proj(hidden_states), 'b n (h d) -> b h n d',
                      h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b n (h d) -> b h n d',
                      h=self.num_key_value_heads)
        v = rearrange(self.v_proj(hidden_states), 'b n (h d) -> b h n d',
                      h=self.num_key_value_heads)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # ---- local branch: attend over the rolling window ----
        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        # generate() can hand down embeddings spanning the whole prefix; take
        # the tail so the rotary phase matches the token being decoded.
        if cos.shape[-2] != q_len:
            cos, sin = cos[..., -q_len:, :], sin[..., -q_len:, :]
        aq, ak = apply_rotary_pos_emb(q, k, cos, sin)

        # window_size + 1: a query sees itself plus window_size of history, so
        # every retained key is in-window and no mask is needed.
        keep = self.window_size + 1
        st['k'] = torch.cat([st['k'], ak], dim=2)[:, :, -keep:]
        st['v'] = torch.cat([st['v'], v], dim=2)[:, :, -keep:]
        attn_out = F.scaled_dot_product_attention(aq, st['k'], st['v'])
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')

        # ---- global branch: read the frozen memory ----
        # ttt_l2's tail-chunk readout: w1 @ (silu(w0 @ q) * (w2 @ q)).
        ttt_q, _, _ = self._ttt_features(
            *(rearrange(x, 'b h n d -> b n (h d)') for x in (q, k, v))
        )
        w0, w1, w2 = (src[name].to(ttt_q.device) for name in ('w0', 'w1', 'w2'))
        # Use the operator's CUDA autocast policy in both prefill and decode.
        qi = ttt_q.transpose(1, 2).to(w0.dtype)
        with torch.autocast(device_type=qi.device.type, enabled=qi.is_cuda,
                            dtype=torch.bfloat16):
            gate = F.silu(torch.bmm(w0, qi))
            ttt_out = torch.bmm(w1, gate * torch.bmm(w2, qi)).transpose(1, 2)

        ttt_out = self.ttt_norm(ttt_out.to(hidden_states.dtype))
        ttt_scale = rearrange(
            F.silu(self.ttt_scale_proj(hidden_states)),
            'b n (h d) -> (b h) n d', h=self.num_ttt_heads,
        )
        ttt_out = ttt_out * ttt_scale.to(ttt_out.dtype)
        ttt_out = rearrange(ttt_out, '(b h) n d -> b n (h d)',
                            b=bsz, h=self.num_ttt_heads)

        if self._ablate_attn:
            attn_out = torch.zeros_like(attn_out)
        if self._ablate_ttt:
            ttt_out = torch.zeros_like(ttt_out)
        o = attn_out.to(ttt_out.dtype) + ttt_out
        out = self.o_proj(o.to(self.o_proj.weight.dtype))

        if reader:
            return (out, None, past_key_value, None)

        _, ttt_k_new, ttt_v_new = self._ttt_features(
            *(rearrange(x, 'b h n d -> b n (h d)') for x in (q, k, v))
        )
        lr = self._decode_lr(hidden_states)
        mom = self._decode_momentum(hidden_states)
        ret = self._decode_retention(hidden_states)
        app = lambda a, b: b if a is None else torch.cat([a, b], dim=1)
        st['k_buf'] = app(st['k_buf'], ttt_k_new)
        st['v_buf'] = app(st['v_buf'], ttt_v_new)
        st['lr_buf'] = [app(st['lr_buf'][i], lr[i]) for i in range(3)]
        st['mom_buf'] = None if mom is None else app(st['mom_buf'], mom)
        st['ret_buf'] = None if ret is None else app(st['ret_buf'], ret)
        return (out, None, past_key_value, None)

    # ---- helpers shared by prefill and decode, so the two cannot drift ----

    def _decode_lr(self, hidden_states):
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            lr = F.linear(hidden_states.float(), self.lr_proj.weight.float(),
                          self.lr_proj.bias.float())
        lr = F.softplus(lr + self.base_lr_inv)
        return rearrange(lr, 'b n (h lrs d) -> lrs (b h) n d',
                         lrs=3, h=self.num_ttt_heads, d=1)

    def _decode_momentum(self, hidden_states):
        if not self.ttt_use_momentum:
            return None
        return rearrange(self.momentum_proj(hidden_states).float(),
                         'b n (h d) -> (b h) n d', h=self.num_ttt_heads)

    def _decode_retention(self, hidden_states):
        if self.ttt_inner_loss != 'l2':
            return None
        return rearrange(self.retention_proj(hidden_states).float(),
                         'b n (h d) -> (b h) n d', h=self.num_ttt_heads)

    def _apply_deferred_chunk(self, st):
        """Run the one chunk update the buffered tokens are now due.

        Reuses the operator: its loop means chunk_size + 1 tokens give exactly
        one update over [0, chunk_size) plus a discarded tail readout.
        """
        C = self.lact_chunk_size
        pad = lambda x: None if x is None else torch.cat([x[:, :C], x[:, C - 1:C]], dim=1)
        k, v = pad(st['k_buf']), pad(st['v_buf'])
        lr0, lr1, lr2 = (pad(b) for b in st['lr_buf'])
        _, w0, w1, w2, mom = block_causal_lact_swiglu_l2(
            st['w0'], st['w1'], st['w2'], k, k, v, lr0, lr1, lr2,
            chunk_size=C, use_muon=self.ttt_use_muon,
            momentum=pad(st['mom_buf']), retention=pad(st['ret_buf']),
            init_momentum=st['mom_state'], return_momentum=True,
        )
        st['w0'], st['w1'], st['w2'] = w0.detach(), w1.detach(), w2.detach()
        st['mom_state'] = None if mom is None else tuple(x.detach() for x in mom)
        keep = lambda x: None if x is None else x[:, C:]
        st['k_buf'], st['v_buf'] = keep(st['k_buf']), keep(st['v_buf'])
        st['lr_buf'] = [keep(b) for b in st['lr_buf']]
        st['mom_buf'], st['ret_buf'] = keep(st['mom_buf']), keep(st['ret_buf'])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[TTTCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ttt_trajectory: Optional[Tuple[torch.Tensor, ...]] = None,
        **kwargs,
    ) -> Tuple:
        bsz, q_len, _ = hidden_states.size()

        if use_cache and self.layer_idx in past_key_value.states:
            return self._decode_step(hidden_states, position_ids,
                                     position_embeddings, past_key_value,
                                     bsz, q_len)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None].to(v.dtype)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_key_value_heads)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # ---------------- local branch: sliding-window attention ----------------
        if position_embeddings is None:
            cos, sin = self.rotary_emb(v, position_ids)
        else:
            cos, sin = position_embeddings
        aq, ak = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = sliding_window_attention(
            aq, ak, v,
            window_size=self.window_size,
            causal=True,
            block_mask_cache=self._block_mask_cache,
        )
        attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')

        # ---------------- global branch: test-time training ----------------
        if q_len <= self.lact_chunk_size:
            # The operator's loop is empty here: no update, so the branch
            # degenerates to a static MLP over the whole sequence.
            logger.warning_once(
                f'seq_len ({q_len}) <= lact_chunk_size ({self.lact_chunk_size}): '
                'the TTT fast weights receive zero updates for this batch.'
            )

        ttt_q, ttt_k, ttt_v = self._ttt_features(
            *(rearrange(x, 'b h n d -> b n (h d)') for x in (q, k, v))
        )

        shared = self._share_gid is not None
        reader = shared and not self._share_leader

        if reader:
            # No inner loop at all: read the leader's per-chunk trajectory with
            # this layer's own q. The leader is min(group) and layers run in
            # index order, so its trajectory is always already published.
            traj = ttt_trajectory
            if traj is None:
                raise RuntimeError(
                    f'layer {self.layer_idx} shares group {self._share_gid} but '
                    'its leader has not published a trajectory this forward. '
                    'The leader must be the lowest layer index in the group.'
                )
            ttt_out = read_lact_swiglu_l2(traj, ttt_q, self.lact_chunk_size)
            if use_cache:
                # A reader owns no memory, so it parks only its attention
                # window; decode pulls the weights from the leader.
                keep = self.window_size + 1
                past_key_value.states[self.layer_idx] = {
                    'k': ak[:, :, -keep:].detach(),
                    'v': v[:, :, -keep:].detach(),
                }
            return self._merge_ttt(ttt_out, attn_out, hidden_states, bsz,
                                   aq, ak, v, output_attentions, past_key_value)

        # Inner-loop lr in fp32; upcast BOTH sides or bf16 projection weights
        # give a mat1/mat2 dtype mismatch.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            lr = F.linear(
                hidden_states.float(),
                self.lr_proj.weight.float(),
                self.lr_proj.bias.float(),
            )
        lr = F.softplus(lr + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, 'b n (h lrs d) -> lrs (b h) n d', lrs=3, h=self.num_ttt_heads, d=1
        )

        if self.ttt_use_momentum:
            momentum = rearrange(
                self.momentum_proj(hidden_states).float(),
                'b n (h d) -> (b h) n d', h=self.num_ttt_heads,
            )
        else:
            momentum = None

        # [num_ttt_heads, ...] -> [b * num_ttt_heads, ...]; a fresh copy of the
        # initial state for every sequence in the batch.
        w0 = self.w0.repeat(bsz, 1, 1).float()
        w1 = self.w1.repeat(bsz, 1, 1).float()
        w2 = self.w2.repeat(bsz, 1, 1).float()

        ttt_kwargs = {}
        if self.ttt_inner_loss == 'l2':
            ttt_op = block_causal_lact_swiglu_l2
            retention = rearrange(
                self.retention_proj(hidden_states).float(),
                'b n (h d) -> (b h) n d', h=self.num_ttt_heads,
            )
            ttt_kwargs['retention'] = retention
        elif self.ttt_prenorm:
            ttt_op = prenorm_block_causal_lact_swiglu
        else:
            ttt_op = block_causal_lact_swiglu

        # Optional inference diagnostic: expose the actual per-chunk states
        # for private memories as well. The callback runs outside the compiled
        # operator and is never saved in a checkpoint/config.
        observer = getattr(self, '_ttt_state_observer', None)
        collect_trajectory = shared or observer is not None
        if observer is not None and (self.training or self.ttt_inner_loss != 'l2'):
            raise RuntimeError('TTT state observation requires eval mode and l2 memory')
        if collect_trajectory:
            ttt_kwargs['return_trajectory'] = True

        if use_cache:
            if self.ttt_inner_loss != 'l2':
                raise NotImplementedError(
                    'incremental decode needs the l2 operator: the dot operator '
                    'returns only the output, not the converged fast weights, so '
                    "there is nothing to carry across steps. Set ttt_inner_loss: 'l2' "
                    'or run generation with use_cache=False.'
                )
            ttt_kwargs['return_state'] = True
            ttt_kwargs['return_momentum'] = True

        ttt_out = ttt_op(
            w0, w1, w2,
            ttt_q, ttt_k, ttt_v,
            lr0, lr1, lr2,
            chunk_size=self.lact_chunk_size,
            use_muon=self.ttt_use_muon,
            momentum=momentum,
            **ttt_kwargs,
        )
        nmom = None
        traj = None
        if use_cache and collect_trajectory:
            ttt_out, nw0, nw1, nw2, nmom, traj = ttt_out
        elif use_cache:
            ttt_out, nw0, nw1, nw2, nmom = ttt_out
        elif collect_trajectory:
            ttt_out, nw0, nw1, nw2, traj = ttt_out
        if observer is not None:
            observer(self, traj)
        if not shared:
            traj = None
        if use_cache:
            keep = self.window_size + 1
            C = self.lact_chunk_size
            n_upd = math.ceil((q_len - C) / C) if q_len > C else 0
            r = q_len - C * n_upd
            tail = (lambda x: None if x is None or r == 0 else x[:, -r:].detach())
            past_key_value.states[self.layer_idx] = {
                'w0': nw0.detach(), 'w1': nw1.detach(), 'w2': nw2.detach(),
                'k': ak[:, :, -keep:].detach(), 'v': v[:, :, -keep:].detach(),
                'k_buf': tail(ttt_k), 'v_buf': tail(ttt_v),
                'lr_buf': [tail(lr0), tail(lr1), tail(lr2)],
                'mom_buf': tail(momentum),
                'ret_buf': tail(ttt_kwargs.get('retention')),
                'mom_state': None if nmom is None else tuple(x.detach() for x in nmom),
            }

        return self._merge_ttt(ttt_out, attn_out, hidden_states, bsz,
                               aq, ak, v, output_attentions, past_key_value, traj)

    def _merge_ttt(self, ttt_out, attn_out, hidden_states, bsz,
                   aq, ak, v, output_attentions, past_key_value, trajectory=None):
        """Gate the TTT readout, sum the branches, project out.

        Shared by the writer and reader paths of forward() so the two cannot
        drift; `ttt_out` arrives as [b*h, n, d] from either.
        """
        ttt_out = self.ttt_norm(ttt_out)
        ttt_scale = rearrange(
            F.silu(self.ttt_scale_proj(hidden_states)),
            'b n (h d) -> (b h) n d', h=self.num_ttt_heads,
        )
        ttt_out = ttt_out * ttt_scale.to(ttt_out.dtype)
        ttt_out = rearrange(ttt_out, '(b h) n d -> b n (h d)', b=bsz, h=self.num_ttt_heads)

        if self._ablate_attn:
            attn_out = torch.zeros_like(attn_out)
        if self._ablate_ttt:
            ttt_out = torch.zeros_like(ttt_out)
        o = attn_out.to(ttt_out.dtype) + ttt_out

        # Attention transfer (LoLCATS stage 1): the softmax output the trainer
        # regresses onto, per layer. Detached -- frozen pretrained behaviour.
        aux = None
        if output_attentions:
            with torch.no_grad():
                teacher = F.scaled_dot_product_attention(aq, ak, v, is_causal=True)
            teacher = rearrange(teacher, 'b h n d -> b n (h d)').detach()
            aux = torch.stack([o, teacher.to(o.dtype)])

        o = self.o_proj(o.to(self.o_proj.weight.dtype))
        return o, aux, past_key_value, trajectory

class LigerGLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LigerGLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        # checkpoints carry the `attn_varient` misspelling; accept both
        variant = getattr(config, 'attn_variant', None) or getattr(config, 'attn_varient', 'ttt')
        if variant != 'ttt':
            raise NotImplementedError(f'unknown attn_variant: {variant}')
        self.self_attn = LinearTTTAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, ttt_trajectory=None,
                **kwargs):
        # Trajectories are ordinary tensor inputs/outputs, including through
        # checkpoint recomputation and device-placement hooks.
        residual = hidden_states
        hidden_states, attn, cache, trajectory = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings,
            ttt_trajectory=ttt_trajectory,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn,)
        if use_cache:
            outputs += (cache,)
        return outputs + (trajectory,)

class LigerGLAPreTrainedModel(LlamaPreTrainedModel):

    config_class = LigerGLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ['LigerGLADecoderLayer']
    _skip_keys_device_placement = "past_key_values"
    _supports_static_cache = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = dict(gradient_checkpointing_kwargs or {})
        if getattr(self.config, 'ttt_share_groups', None):
            if kwargs.get('use_reentrant', False):
                raise ValueError('Shared trajectories require use_reentrant=False for checkpointing')
            # Reentrant checkpointing does not track nested trajectory outputs.
            kwargs['use_reentrant'] = False
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def _init_weights(self, module):
        super()._init_weights(module)
        # `apply` visits children before parents, so by the time we reach the
        # attention module its Linears have already been re-randomised.
        if isinstance(module, LinearTTTAttention):
            module.reset_ttt_parameters()

class LigerGLAModel(LlamaModel, LigerGLAPreTrainedModel):

    # LlamaModel precedes LigerGLAPreTrainedModel in the MRO, so without this
    # `config_class` resolves to LlamaConfig and AutoModel.register rejects it.
    config_class = LigerGLAConfig

    def __init__(self, config: LigerGLAConfig):
        config.validate_ttt()
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LigerGLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Shared initial weights; per-forward trajectories are passed explicitly.
        self._tie_ttt_memories()

        # Initialize weights and apply final processing once during construction.
        self.post_init()

    def _tie_ttt_memories(self):
        """Give every member of a share group its leader's Parameter objects.

        Must be re-applied after loading: device_map REPLACES Parameter objects
        and undoes __init__'s aliasing, leaving followers with private weights
        they never read and never train.
        """
        n = 0
        for g in (getattr(self.config, 'ttt_share_groups', None) or []):
            g = sorted(g)
            lead = self.layers[g[0]].self_attn
            # Some transformers versions reach tie_weights() from the base
            # LlamaModel.__init__, before this class has swapped its own
            # attention in. Nothing to tie yet; the post-load call does the work.
            if not hasattr(lead, 'w0'):
                return 0
            for li in g[1:]:
                a = self.layers[li].self_attn
                if a.w0.shape != lead.w0.shape:
                    raise ValueError(
                        f'layers {g[0]} and {li} share a memory but have '
                        f'different fast-weight shapes {tuple(lead.w0.shape)} vs '
                        f'{tuple(a.w0.shape)} -- ttt_inter_multi must match '
                        'within a share group'
                    )
                # count only genuine re-ties -- from_pretrained calls
                # tie_weights() several times and re-tying is a no-op
                if a.w0 is not lead.w0:
                    n += 1
                a.w0, a.w1, a.w2 = lead.w0, lead.w1, lead.w2
        if n:
            print(f'-> tied {n} TTT memories into '
                  f'{len(self.config.ttt_share_groups)} shared group(s)')
        return n

    def tie_weights(self):
        super().tie_weights()
        self._tie_ttt_memories()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        

        if use_cache and not isinstance(past_key_values, TTTCache):
            # generate() supplies an empty DynamicCache on its first call.
            if past_key_values is None or (
                isinstance(past_key_values, DynamicCache)
                and past_key_values.get_seq_length() == 0
            ):
                past_key_values = TTTCache()
            else:
                raise ValueError('Continue decoding with the TTTCache returned by this model.')
        elif not use_cache:
            past_key_values = None

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = attention_mask

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Tensor trajectories live only in this forward's computation graph.
        trajectories = {}

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            gid = decoder_layer.self_attn._share_gid
            trajectory = trajectories.get(gid)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    trajectory,
                )

            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    ttt_trajectory=trajectory,
                )

            hidden_states = layer_outputs[0]
            if gid is not None and layer_outputs[-1] is not None:
                trajectories[gid] = layer_outputs[-1]

            if use_cache:
                # LlamaDecoderLayer repacks the tuple by flag: the cache is at
                # index 2 only when attentions are also returned.
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if next_cache is not None:
            next_cache._seen_tokens += inputs_embeds.shape[1]

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
class LigerGLAForCausalLM(LlamaForCausalLM, LigerGLAPreTrainedModel, GenerationMixin):
    config_class = LigerGLAConfig    # see LigerGLAModel
    _tied_weights_keys = ["lm_head.weight"]

    def tie_weights(self):
        # from_pretrained calls tie_weights() on the TOP-LEVEL model after
        # loading, so the shared memories must be re-tied from here too. Guarded
        # because some transformers versions run post_init() -> tie_weights()
        # from the base __init__, before `self.model` has been replaced.
        super().tie_weights()
        inner = getattr(self, 'model', None)
        if hasattr(inner, '_tie_ttt_memories'):
            inner._tie_ttt_memories()

    def __init__(self, config):
        config.validate_ttt()
        super().__init__(config)
        self.model = LigerGLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # safe_serialization refuses to write storage-sharing tensors unless
        # they are declared here. Every follower aliases its leader's w0/w1/w2.
        extra = [f'model.layers.{li}.self_attn.{w}'
                 for g in (getattr(config, 'ttt_share_groups', None) or [])
                 for li in sorted(g)[1:]
                 for w in ('w0', 'w1', 'w2')]
        if extra:
            self._tied_weights_keys = list(self._tied_weights_keys or []) + extra

        # Initialize weights and apply final processing
        self.post_init()
