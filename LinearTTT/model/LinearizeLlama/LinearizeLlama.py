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
from .ttt_l2 import block_causal_lact_swiglu_l2

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    print("flash_attn_2 is not available")

from fla.models.utils import Cache as FlaCache
from fla.ops.gla import fused_chunk_gla, fused_recurrent_gla

from .Configuration import LigerGLAConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

logger = logging.get_logger(__name__)

# flex_attention only reaches its fused Triton kernel when the call is compiled.
# Called bare it falls back to `math_attention`, which materialises the whole
# [B, H, Q, KV] score matrix -- 8 GiB at 8192 tokens / 32 heads -- and applies
# the mask afterwards, so the window buys nothing.
_flex_attention_compiled = None


def _compiled_flex_attention():
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    return _flex_attention_compiled


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

    if causal:
        def mask_mod(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= window_size
            return causal_mask & window_mask
    else:
        def mask_mod(b, h, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= window_size

    cache_key = (B, H, Q_len, window_size, causal, device)
    if block_mask_cache is not None and cache_key in block_mask_cache:
        block_mask = block_mask_cache[cache_key]
    else:
        block_mask = create_block_mask(mask_mod, B, H, Q_len, Q_len, device=device)
        if block_mask_cache is not None:
            block_mask_cache[cache_key] = block_mask

    output = _compiled_flex_attention()(q, k, v, block_mask=block_mask, scale=scale)
    return output

class LigerGatedLinearAttention(nn.Module):
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
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.pool_g = nn.AdaptiveAvgPool1d(output_size=self.head_dim * self.num_key_value_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        last_state = None
        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            last_state = past_key_value[self.layer_idx]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        g = self.pool_g(k)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_key_value_heads)
        g = rearrange(g, 'b n (h m) -> b h n m', h=self.num_key_value_heads)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        g = repeat_kv(g, self.num_key_value_groups)

        sq, sk, sv = q, k, v

        # norm
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-1)
        
        gate_logit_normalizer = 16
        g = F.logsigmoid(g) / gate_logit_normalizer # (b, h, n, m)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        offsets = kwargs.get('offsets', None)
        scale = 1 
        q, k, v, g = (x.to(torch.float32).contiguous() for x in (q, k, v, g))

        if self.training or q.shape[-2] > 1:
            o_, recurrent_state = fused_chunk_gla(q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)
        else:
            o_, recurrent_state = fused_recurrent_gla(q, k, v, g, scale=scale, initial_state=recurrent_state, output_final_state=True)

        if past_key_value is not None:
            past_key_value.update(
                recurrent_state=recurrent_state,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )
        
        q_len = hidden_states.size(-2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(sv, position_ids)
        else:
            cos, sin = position_embeddings
        sq, sk = apply_rotary_pos_emb(sq, sk, cos, sin)

        input_dtype = sq.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            sq = sq.to(target_dtype)
            sk = sk.to(target_dtype)
            sv = sv.to(target_dtype)

        window_size = 64
        if attention_mask is not None and 0.0 in attention_mask:
            pass
        else:
            attention_mask = None

        y = _flash_attention_forward( # Reashape to the expected shape for Flash Attention
            sq.transpose(1, 2),
            sk.transpose(1, 2),
            sv.transpose(1, 2),
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=0.0,
            sliding_window=window_size,
            use_top_left_mask=False,
            is_causal=True,
            target_dtype=torch.float32,
        ).transpose(1, 2)
        o_ = 0.5 * y + 0.5 * o_ 
        o = rearrange(o_.bfloat16(), 'b h n d -> b n (h d)')
        o = self.o_proj(o)

        return o, None, past_key_value

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

        # 'dot': the upstream Hebbian bias, L = -f(k)^T v (LaCT Eq. 7).
        # 'l2' : regression bias, L = 1/2||f(k) - v||^2, whose gradient carries
        #        the residual, so a key that already reads out correctly is left
        #        alone. See ttt_l2.py.
        self.ttt_inner_loss = getattr(config, 'ttt_inner_loss', 'dot')
        if self.ttt_inner_loss not in ('dot', 'l2'):
            raise ValueError(
                f"ttt_inner_loss must be 'dot' or 'l2', got {self.ttt_inner_loss!r}"
            )
        if self.ttt_inner_loss == 'l2' and self.ttt_prenorm:
            raise NotImplementedError(
                "ttt_inner_loss='l2' has no prenorm variant; set ttt_prenorm=False."
            )

        d_in = d_out = self.ttt_head_dim
        d_h = int(self.ttt_head_dim * getattr(config, 'ttt_inter_multi', 1.0))
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

       
        self.ttt_qk_scale = nn.Parameter(torch.ones(2, self.inner_dim))
        self.ttt_qk_offset = nn.Parameter(torch.zeros(2, self.inner_dim))

        self.ttt_norm = LlamaRMSNorm(self.ttt_head_dim, eps=config.rms_norm_eps)
        self.ttt_scale_proj = nn.Linear(self.hidden_size, self.num_ttt_heads)
        self.ttt_scale_init_bias = getattr(config, 'ttt_scale_init_bias', 0.1)
        self.fw_init_gain = gain
        self.reset_ttt_parameters()

        self._block_mask_cache = {}

    def reset_ttt_parameters(self):
        """Initialise the parameters that have no counterpart in the checkpoint.

        Called again from LigerGLAPreTrainedModel._init_weights, because
        PreTrainedModel.post_init() re-randomises every nn.Linear it finds --
        including the output gate, whose whole job is to start nearly closed.
        """
        d_in, d_h = self.ttt_head_dim, self.w0.shape[1]
        with torch.no_grad():
            self.w0.normal_(0, 1 / math.sqrt(d_in)).mul_(self.fw_init_gain)
            self.w1.normal_(0, 1 / math.sqrt(d_h)).mul_(self.fw_init_gain)
            self.w2.normal_(0, 1 / math.sqrt(d_in)).mul_(self.fw_init_gain)
            self.ttt_qk_scale.fill_(1.0)
            self.ttt_qk_offset.zero_()
            # ttt_norm is the only RMSNorm in this model with no counterpart in
            # the Llama checkpoint, and LlamaPreTrainedModel._init_weights only
            # handles Linear/Embedding. Under from_pretrained's meta-device path
            # it would otherwise be materialised from uninitialised memory.
            self.ttt_norm.weight.fill_(1.0)
        # Start the TTT branch nearly closed so the frozen host's residual
        # stream survives step 0, but not fully closed: silu(0) == 0 would cut
        # the gradient to lr_proj and the fast weights entirely.
        nn.init.zeros_(self.ttt_scale_proj.weight)
        nn.init.constant_(self.ttt_scale_proj.bias, self.ttt_scale_init_bias)

    def _ttt_features(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """[b, n, inner_dim] -> [b * num_ttt_heads, n, ttt_head_dim]."""
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        q = q * self.ttt_qk_scale[0] + self.ttt_qk_offset[0]
        k = k * self.ttt_qk_scale[1] + self.ttt_qk_offset[1]
        q, k, v = (
            rearrange(x, 'b n (h d) -> (b h) n d', h=self.num_ttt_heads) for x in (q, k, v)
        )
        return l2_norm(q), l2_norm(k), v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[FlaCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if past_key_value is not None and len(past_key_value) > self.layer_idx:
            # Incremental decoding needs the converged fast weights and a rolling
            # window of k/v carried across steps. `block_causal_lact_swiglu`
            # returns only the output, not the final w0/w1/w2, so state carry
            # needs a variant of the operator that also returns them.
            raise NotImplementedError(
                'LinearTTTAttention has no incremental-decode path yet: the LaCT operator '
                'does not return the final fast weights, so they cannot be cached across steps.'
            )

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
            # range(0, seq_len - chunk_size, chunk_size) is empty: the fast
            # weights are never updated and the branch degenerates to a static
            # MLP over the whole sequence.
            logger.warning_once(
                f'seq_len ({q_len}) <= lact_chunk_size ({self.lact_chunk_size}): '
                'the TTT fast weights receive zero updates for this batch.'
            )

        ttt_q, ttt_k, ttt_v = self._ttt_features(
            *(rearrange(x, 'b h n d -> b n (h d)') for x in (q, k, v))
        )

        # LaCT wants the inner-loop learning rate in fp32; under bf16 training the
        # projection weights are bf16, so upcast both sides rather than the input
        # alone (mat1/mat2 dtype mismatch otherwise).
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

        if self.ttt_inner_loss == 'l2':
            ttt_op = block_causal_lact_swiglu_l2
        elif self.ttt_prenorm:
            ttt_op = prenorm_block_causal_lact_swiglu
        else:
            ttt_op = block_causal_lact_swiglu
        ttt_out = ttt_op(
            w0, w1, w2,
            ttt_q, ttt_k, ttt_v,
            lr0, lr1, lr2,
            chunk_size=self.lact_chunk_size,
            use_muon=self.ttt_use_muon,
            momentum=momentum,
        )

        ttt_out = self.ttt_norm(ttt_out)
        ttt_scale = rearrange(
            F.silu(self.ttt_scale_proj(hidden_states)),
            'b n (h d) -> (b h) n d', h=self.num_ttt_heads,
        )
        ttt_out = ttt_out * ttt_scale.to(ttt_out.dtype)
        ttt_out = rearrange(ttt_out, '(b h) n d -> b n (h d)', b=bsz, h=self.num_ttt_heads)

        # ---------------- merge ----------------
        o = attn_out.to(ttt_out.dtype) + ttt_out

        # Attention transfer (LoLCATS stage 1): alongside the hybrid output,
        # compute what full causal softmax attention would have produced from the
        # same q/k/v, so the trainer can regress one onto the other per layer.
        # The teacher is detached -- it is the frozen pretrained behaviour, not
        # something to optimise.
        aux = None
        if output_attentions:
            with torch.no_grad():
                teacher = F.scaled_dot_product_attention(aq, ak, v, is_causal=True)
            teacher = rearrange(teacher, 'b h n d -> b n (h d)').detach()
            aux = torch.stack([o, teacher.to(o.dtype)])

        o = self.o_proj(o.to(self.o_proj.weight.dtype))

        return o, aux, past_key_value

class LigerGLADecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LigerGLAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        # `attn_varient` is the spelling used in Configs/liger.yml; accept both.
        variant = getattr(config, 'attn_variant', None) or getattr(config, 'attn_varient', 'liger')
        if variant == 'liger':
            self.self_attn = LigerGatedLinearAttention(config=config, layer_idx=layer_idx)
        elif variant == 'ttt':
            self.self_attn = LinearTTTAttention(config=config, layer_idx=layer_idx)
        else:
            raise NotImplementedError(f'unknown attn_variant: {variant}')
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

class LigerGLAPreTrainedModel(LlamaPreTrainedModel):

    config_class = LigerGLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ['LigerGLADecoderLayer']
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        super()._init_weights(module)
        # `apply` visits children before parents, so by the time we reach the
        # attention module its Linears have already been re-randomised.
        if isinstance(module, LinearTTTAttention):
            module.reset_ttt_parameters()

class LigerGLAModel(LlamaModel, LigerGLAPreTrainedModel):

    def __init__(self, config: LigerGLAConfig):
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

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Tuple, FlaCache, List[torch.FloatTensor]]] = None,
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
        

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, FlaCache):
            past_key_values = FlaCache.from_legacy_cache(past_key_values)

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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if output_attentions:
            all_softmax_hidden_states = () 

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                if all_softmax_hidden_states is not None:
                    all_softmax_hidden_states += (hidden_states,)

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
                )

            hidden_states = layer_outputs[0]

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

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
class LigerGLAForCausalLM(LlamaForCausalLM, LigerGLAPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LigerGLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()