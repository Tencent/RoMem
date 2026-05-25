from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


class BGEM3EmbeddingModel(BaseEmbeddingModel):
    """
    BGE-M3 embedding using FlagEmbedding.BGEM3FlagModel (dense vectors).
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}"
            )

        self._init_embedding_config()
        self.model = BGEM3FlagModel(self.embedding_model_name, use_fp16=True)

    def _init_embedding_config(self) -> None:
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "batch_size": self.global_config.embedding_batch_size,
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        outputs = self.model.encode(
            texts,
            batch_size=self.embedding_config.encode_params.get("batch_size", 16),
            max_length=self.embedding_config.encode_params.get("max_length", 8192),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        dense = outputs["dense_vecs"]
        return np.array(dense if isinstance(dense, list) else dense)

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)
        max_length = params.pop("max_length", 8192)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            num_batches = (len(texts) + batch_size - 1) // batch_size
            pbar = tqdm(total=len(texts), desc="Batch Encoding", disable=num_batches <= 2)
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_out = self.model.encode(
                    batch,
                    batch_size=batch_size,
                    max_length=max_length,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )["dense_vecs"]
                results.append(batch_out)
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
