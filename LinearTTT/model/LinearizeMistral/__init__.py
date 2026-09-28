# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .Configuration import LigerMistralGLAConfig
from .LinearizeMistral import (
    LigerMistralGLADecoderLayer,
    LigerMistralGLAForCausalLM,
    LigerMistralGLAModel,
    LigerMistralGLAPreTrainedModel,
)

AutoConfig.register(LigerMistralGLAConfig.model_type, LigerMistralGLAConfig)
AutoModel.register(LigerMistralGLAConfig, LigerMistralGLAModel)
AutoModelForCausalLM.register(LigerMistralGLAConfig, LigerMistralGLAForCausalLM)

__all__ = [
    'LigerMistralGLAConfig',
    'LigerMistralGLADecoderLayer',
    'LigerMistralGLAForCausalLM',
    'LigerMistralGLAModel',
    'LigerMistralGLAPreTrainedModel',
]
