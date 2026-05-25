from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class EntityLinker:
    """
    Text-space entity linker.

    This bridges query text embeddings to the entity embedding store (text embeddings).
    """

    top_k: int = 5

    def link(
        self,
        query_embedding: np.ndarray,
        entity_texts: List[str],
        entity_embeddings: np.ndarray,
    ) -> List[Tuple[str, float]]:
        if entity_embeddings.size == 0 or not entity_texts:
            return []

        q = np.asarray(query_embedding).reshape(-1)
        e = np.asarray(entity_embeddings)
        if e.ndim != 2:
            return []

        # cosine similarity with safe normalization
        qn = np.linalg.norm(q) + 1e-12
        en = np.linalg.norm(e, axis=1) + 1e-12
        sims = (e @ q) / (en * qn)
        idx = np.argsort(sims)[::-1][: min(self.top_k, len(entity_texts))]
        return [(entity_texts[int(i)], float(sims[int(i)])) for i in idx]
