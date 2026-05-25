import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from .base import BaseLLM

logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    # Lazy imports to avoid pulling optional deps unless needed.
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        from .bedrock_llm import BedrockLLM
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        from .transformers_llm import TransformersLLM
        return TransformersLLM(config)

    from .openai_gpt import CacheOpenAI
    return CacheOpenAI.from_experiment_config(config)


__all__ = ["BaseLLM", "_get_llm_class"]
    
