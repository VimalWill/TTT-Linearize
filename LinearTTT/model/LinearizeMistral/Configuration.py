# -*- coding: utf-8 -*-

import math

from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral.configuration_mistral import MistralConfig

class LigerMistralGLAConfig(MistralConfig, PretrainedConfig):
    model_type = 'liger_mistral_gla'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        # mistral config
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        # --- test-time training (LaCT) branch ---
        num_ttt_heads=None,       # None -> num_attention_heads (no k/v duplication under GQA)
        ttt_inter_multi=1.0,      # SwiGLU fast-weight hidden expansion
        lact_chunk_size=512,      # tokens per fast-weight update
        window_size=512,          # scalar or one span per layer; each >= lact_chunk_size
        ttt_base_lr=1e-2,         # base inner-loop learning rate
        # 'dot' = LaCT Eq. 7 (Hebbian) with Eq. 8's fixed-norm renorm.
        # 'l2'  = Atlas Eq. 9 regression with Eq. 32's retention gate instead.
        # These are the two coherent pairings; do not cross them.
        ttt_inner_loss='dot',
        ttt_retention_init_bias=4.0,   # sigmoid(4.0) ~ 0.982 decay per chunk
        # Layer groups that SHARE one running fast-weight memory, GQA-style:
        # the lowest index writes, the rest read its per-chunk trajectory with
        # their own q. One parameter set and one live state per group.
        ttt_share_groups=None,
        ttt_layer_indices=None,    # layers retaining the TTT branch; None means all
        ttt_reader_alignment='none',  # 'linear': per-reader, per-head output map
        ttt_gate='silu',          # output gate: 'silu' (legacy) or 'sigmoid' (gamma in [0,1])
        ttt_swa_dropout=0.0,      # p(zero the SWA branch per sequence) at TRAIN time
        ttt_feature_map='none',   # 'taylor2': degree-2 polynomial lift of q and k
        ttt_feature_dim=32,       # project to this BEFORE lifting; d' -> 1+d'+d'^2
        ttt_feature_layers=None,  # layers that get the lift; None means all TTT layers
        ttt_idle_zero=False,      # zero the readout when the inner loop never ran
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
            sliding_window=sliding_window,
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
        self.ttt_layer_indices = ttt_layer_indices
        self.ttt_reader_alignment = ttt_reader_alignment
        self.ttt_gate = ttt_gate
        self.ttt_swa_dropout = ttt_swa_dropout
        self.ttt_feature_map = ttt_feature_map
        self.ttt_feature_dim = ttt_feature_dim
        self.ttt_feature_layers = ttt_feature_layers
        self.ttt_idle_zero = ttt_idle_zero
        self.ttt_use_muon = ttt_use_muon
        self.ttt_use_momentum = ttt_use_momentum
        self.ttt_prenorm = ttt_prenorm
        self.fw_init_gain = fw_init_gain
        self.ttt_scale_init_bias = ttt_scale_init_bias
        self.validate_ttt()

    def validate_ttt(self):
        """Also called at model construction after harness config overrides."""
        if not isinstance(self.lact_chunk_size, int) or self.lact_chunk_size < 1:
            raise ValueError('lact_chunk_size must be a positive integer')
        windows = self.window_size
        if isinstance(windows, (list, tuple)):
            if len(windows) != self.num_hidden_layers:
                raise ValueError('window_size must contain one value per layer')
        else:
            windows = [windows]
        if any(type(w) is not int or w < self.lact_chunk_size for w in windows):
            raise ValueError('Every window_size must be an integer >= lact_chunk_size')
        groups = self.ttt_share_groups or []
        active = (set(range(self.num_hidden_layers)) if self.ttt_layer_indices is None
                  else set(self.ttt_layer_indices))
        if self.ttt_layer_indices is not None and any(type(i) is not int or not 0 <= i < self.num_hidden_layers for i in active):
            raise ValueError('ttt_layer_indices must contain valid layer indices')
        if self.ttt_reader_alignment not in ('none', 'linear'):
            raise ValueError('ttt_reader_alignment must be "none" or "linear"')
        if self.ttt_gate not in ('silu', 'sigmoid'):
            raise ValueError('ttt_gate must be "silu" or "sigmoid"')
        if not 0.0 <= float(self.ttt_swa_dropout) < 1.0:
            raise ValueError('ttt_swa_dropout must be in [0, 1); 1.0 would remove\n'
                             'the attention branch entirely rather than drop it')
        if self.ttt_feature_map not in ('none', 'taylor2'):
            raise ValueError('ttt_feature_map must be "none" or "taylor2"')
        if self.ttt_feature_map == 'taylor2':
            if not isinstance(self.ttt_feature_dim, int) or self.ttt_feature_dim < 1:
                raise ValueError('ttt_feature_dim must be a positive integer')
            # 1 + d' + d'^2 replaces ttt_head_dim in w0/w2, so the state grows by
            # roughly (1+d'+d'^2)/ttt_head_dim on those two matrices.
            if self.ttt_feature_dim > 64:
                raise ValueError(
                    f'ttt_feature_dim={self.ttt_feature_dim} lifts to '
                    f'{1 + self.ttt_feature_dim + self.ttt_feature_dim**2} dims; '
                    'Based uses 16 and this project sizes for 16-32')
            if self.ttt_feature_layers is not None:
                bad = [i for i in self.ttt_feature_layers
                       if type(i) is not int or not 0 <= i < self.num_hidden_layers]
                if bad:
                    raise ValueError(f'ttt_feature_layers out of range: {bad}')
                if active and not set(self.ttt_feature_layers) <= active:
                    raise ValueError(
                        'ttt_feature_layers must be a subset of the layers that '
                        'have a TTT branch; lifting a layer with no memory does '
                        'nothing and silently misreports the state size')
        # silu(0.1)=0.052 leaves the branch nearly shut so the pretrained residual
        # stream survives step 0. sigmoid(0.1)=0.525 would open it halfway and
        # wreck the model at init; sigmoid(-3.0)=0.047 matches the silu default.
        if self.ttt_gate == 'sigmoid' and self.ttt_scale_init_bias > -1.0:
            raise ValueError(
                f'ttt_gate="sigmoid" with ttt_scale_init_bias='
                f'{self.ttt_scale_init_bias} opens the gate to '
                f'{1/(1+math.exp(-self.ttt_scale_init_bias)):.3f} at init. '
                'Use about -3.0 (sigmoid -> 0.047, matching silu(0.1)=0.052).')
        if not isinstance(groups, (list, tuple)):
            raise ValueError('ttt_share_groups must be a list of layer groups')
        if groups and self.ttt_inner_loss != 'l2':
            raise ValueError('Shared trajectories require ttt_inner_loss="l2"')
        seen = set()
        for group in groups:
            if not isinstance(group, (list, tuple)) or not group:
                raise ValueError('Each ttt_share_groups entry must be a nonempty list of layer indices')
            for index in group:
                if type(index) is not int or not 0 <= index < self.num_hidden_layers:
                    raise ValueError(f'Invalid shared layer index: {index!r}')
                if index in seen:
                    raise ValueError(f'Layer {index} occurs more than once in ttt_share_groups')
                seen.add(index)
                if index not in active:
                    raise ValueError('Every shared-memory layer must retain the TTT branch')
            if isinstance(self.ttt_inter_multi, (list, tuple)):
                if len(self.ttt_inter_multi) != self.num_hidden_layers:
                    raise ValueError('ttt_inter_multi must contain one value per layer')
                if len({self.ttt_inter_multi[i] for i in group}) != 1:
                    raise ValueError('ttt_inter_multi must match within each shared group')
        if self.ttt_reader_alignment != 'none' and not any(len(g) > 1 for g in groups):
            raise ValueError('Reader alignment requires at least one shared-memory reader')

    def window_size_for_layer(self, layer_idx):
        """Resolve the same per-layer span for prefill and decode caches."""
        if isinstance(self.window_size, (list, tuple)):
            if type(layer_idx) is not int or not 0 <= layer_idx < self.num_hidden_layers:
                raise ValueError('Per-layer window_size requires a valid layer_idx')
            return self.window_size[layer_idx]
        return self.window_size
