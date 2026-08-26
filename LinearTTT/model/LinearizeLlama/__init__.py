# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .Configuration import LigerGLAConfig
from .LinearizeLlama import (
    LigerGatedLinearAttention,
    LigerGLADecoderLayer,
    LigerGLAForCausalLM,
    LigerGLAModel,
    LigerGLAPreTrainedModel,
)

AutoConfig.register(LigerGLAConfig.model_type, LigerGLAConfig)
AutoModel.register(LigerGLAConfig, LigerGLAModel)
AutoModelForCausalLM.register(LigerGLAConfig, LigerGLAForCausalLM)

__all__ = [
    'LigerGLAConfig',
    'LigerGatedLinearAttention',
    'LigerGLADecoderLayer',
    'LigerGLAForCausalLM',
    'LigerGLAModel',
    'LigerGLAPreTrainedModel',
]
