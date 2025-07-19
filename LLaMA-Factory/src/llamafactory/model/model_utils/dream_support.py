# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import TYPE_CHECKING, Optional, Any, Dict
from transformers import LlamaConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel

from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = get_logger(__name__)


def is_dream_config(config: "PretrainedConfig") -> bool:
    """Check if the config is a DreamConfig."""
    return getattr(config, "model_type", None) == "Dream"


def convert_dream_config_to_llama(config: "PretrainedConfig") -> LlamaConfig:
    """Convert DreamConfig to LlamaConfig for compatibility."""
    if not is_dream_config(config):
        return config
    
    logger.info("Converting DreamConfig to LlamaConfig for compatibility...")
    
    # Create LlamaConfig with DreamConfig parameters
    llama_config = LlamaConfig(
        vocab_size=getattr(config, "vocab_size", 152064),
        hidden_size=getattr(config, "hidden_size", 3584),
        intermediate_size=getattr(config, "intermediate_size", 18944),
        num_hidden_layers=getattr(config, "num_hidden_layers", 28),
        num_attention_heads=getattr(config, "num_attention_heads", 28),
        num_key_value_heads=getattr(config, "num_key_value_heads", 4),
        hidden_act=getattr(config, "hidden_act", "silu"),
        max_position_embeddings=getattr(config, "max_position_embeddings", 131072),
        initializer_range=getattr(config, "initializer_range", 0.02),
        rms_norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        use_cache=getattr(config, "use_cache", True),
        tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        rope_theta=getattr(config, "rope_theta", 1000000.0),
        rope_scaling=getattr(config, "rope_scaling", None),
        attention_dropout=getattr(config, "attention_dropout", 0.0),
        pad_token_id=getattr(config, "pad_token_id", 151643),
        bos_token_id=getattr(config, "bos_token_id", 151643),
        eos_token_id=getattr(config, "eos_token_id", 151643),
    )
    
    # Copy additional attributes from original config
    for attr_name in dir(config):
        if not attr_name.startswith('_') and not hasattr(llama_config, attr_name):
            try:
                setattr(llama_config, attr_name, getattr(config, attr_name))
            except:
                pass
    
    return llama_config


def load_dream_model(model_args, init_kwargs: Dict[str, Any]):
    """Load Dream model using Llama architecture for compatibility."""
    original_config = init_kwargs["config"]
    
    if not is_dream_config(original_config):
        return None
    
    logger.info("Loading Dream model using Llama architecture for compatibility...")
    
    # Convert config
    llama_config = convert_dream_config_to_llama(original_config)
    init_kwargs["config"] = llama_config
    
    # Load model using AutoModelForCausalLM
    if model_args.train_from_scratch:
        model = AutoModelForCausalLM.from_config(llama_config)
    else:
        try:
            # Try to load with trust_remote_code first
            model = AutoModelForCausalLM.from_pretrained(**init_kwargs)
        except Exception as e:
            logger.warning(f"Failed to load Dream model directly: {e}")
            logger.info("Creating new Llama model with Dream config...")
            model = AutoModelForCausalLM.from_config(llama_config)
    
    # Store original config for reference
    model.config._original_dream_config = original_config
    
    return model