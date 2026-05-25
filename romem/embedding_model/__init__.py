from .base import EmbeddingConfig, BaseEmbeddingModel

# NOTE: Do not import heavy/optional embedding backends at module import time.
# Many providers (e.g. FlagEmbedding/transformers/scipy) pull in native deps that
# may not be available in all environments. Use lazy imports in
# `_get_embedding_model_class` instead.


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    # Check explicit prefix-based backends first
    if embedding_model_name.startswith("Transformers/"):
        from .Transformers import TransformersEmbeddingModel
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        from .VLLM import VLLMEmbeddingModel
        return VLLMEmbeddingModel
    elif "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        return NVEmbedV2EmbeddingModel
    elif "bge-m3" in embedding_model_name.lower():
        from .BGEM3 import BGEM3EmbeddingModel
        return BGEM3EmbeddingModel
    elif "contriever" in embedding_model_name:
        from .Contriever import ContrieverModel
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        from .Cohere import CohereEmbeddingModel
        return CohereEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"


__all__ = [
    "EmbeddingConfig",
    "BaseEmbeddingModel",
    "_get_embedding_model_class",
]
