# -*- coding: utf-8 -*-

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

class LigerGLAConfig(LlamaConfig, PretrainedConfig):
    model_type = 'liger_gla'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # llama config
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        # --- test-time training (LaCT) branch ---
        num_ttt_heads=None,       # None -> num_attention_heads (no k/v duplication under GQA)
        ttt_inter_multi=1.0,      # SwiGLU fast-weight hidden expansion
        lact_chunk_size=512,      # tokens per fast-weight update
        window_size=512,          # sliding-window attention span; must be >= lact_chunk_size
        ttt_base_lr=1e-2,         # base inner-loop learning rate
        # 'dot' = LaCT Eq. 7 (Hebbian) with Eq. 8's fixed-norm renorm.
        # 'l2'  = Atlas Eq. 9 regression with Eq. 32's retention gate instead.
        # These are the two coherent pairings; do not cross them.
        ttt_inner_loss='dot',
        ttt_retention_init_bias=4.0,   # sigmoid(4.0) ~ 0.982 decay per chunk
        # Layer groups that SHARE one running fast-weight memory, GQA-style:
        # e.g. [[2,3,...,24]] gives the whole interior a single state, read by
        # each member's q and written by each member's k/v in depth order.
        # One parameter set and one live state per group instead of per layer.
        ttt_share_groups=None,
        ttt_use_muon=False,       # Newton-Schulz orthogonalisation of the fast-weight update
        ttt_use_momentum=True,
        ttt_prenorm=False,        # use the prenorm variant of the TTT operator
        fw_init_gain=0.5,         # scale of the initial fast weights
        ttt_scale_init_bias=0.1,  # opens the output gate slightly at init
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            mlp_bias=mlp_bias,
            head_dim=head_dim,
            **kwargs,
        )
        self.num_ttt_heads = num_ttt_heads if num_ttt_heads is not None else self.num_attention_heads
        self.ttt_inter_multi = ttt_inter_multi
        self.lact_chunk_size = lact_chunk_size
        self.window_size = window_size
        self.ttt_base_lr = ttt_base_lr
        self.ttt_inner_loss = ttt_inner_loss
        self.ttt_retention_init_bias = ttt_retention_init_bias
        self.ttt_share_groups = ttt_share_groups
        self.ttt_use_muon = ttt_use_muon
        self.ttt_use_momentum = ttt_use_momentum
        self.ttt_prenorm = ttt_prenorm
        self.fw_init_gain = fw_init_gain
        self.ttt_scale_init_bias = ttt_scale_init_bias