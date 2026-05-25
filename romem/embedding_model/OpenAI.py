from copy import deepcopy
from typing import List, Optional

import random
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from openai import AzureOpenAI
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        if self.global_config.azure_embedding_endpoint is None:
            self.client = OpenAI(
                base_url=self.global_config.embedding_base_url
            )
        else:
            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint)


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str]):
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        max_attempts = int(getattr(self.global_config, "max_retry_attempts", 5))
        max_attempts = max(1, max_attempts)
        base_sleep = 1.0
        max_sleep = 10.0
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
                last_exc = None
                break
            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as exc:
                last_exc = exc
                if attempt == max_attempts:
                    raise
                sleep_for = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                sleep_for += random.random() * 0.25 * sleep_for
                logger.warning(
                    "Embedding request failed (attempt %d/%d). Retrying in %.2fs: %s",
                    attempt,
                    max_attempts,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)
        if last_exc is not None:
            raise last_exc
        results = np.array([v.embedding for v in response.data])

        return results

    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "
            # del params["instruction"]

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            num_batches = (len(texts) + batch_size - 1) // batch_size
            pbar = tqdm(total=len(texts), desc="Batch Encoding", disable=num_batches <= 2)
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.append(self.encode(batch))
                pbar.update(batch_size)
            pbar.close()
            results = np.concatenate(results)

        # `openai` embeddings return Python lists; keep everything as numpy arrays.
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
