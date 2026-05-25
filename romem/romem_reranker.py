"""
RoMemReranker — Standalone temporal KGE scoring module.

Provides temporal reranking for any retrieval pipeline. Researchers can
plug this into their own memory systems to add temporal awareness without
adopting the full RoMem pipeline.

Usage:
    from romem import RoMemReranker

    reranker = RoMemReranker(embedding="text-embedding-3-small")
    reranker.fit([
        ("Obama", "president_of", "USA", "2009-01-20"),
        ("Biden", "president_of", "USA", "2021-01-20"),
    ])
    scores = reranker.score(
        candidates=[("Obama", "president_of", "USA"), ("Biden", "president_of", "USA")],
        query_time="2019-06-01",
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np

from dataclasses import dataclass as _dataclass

from .kge.config import TKGEConfig
from .kge.retriever import TKGERetriever
from .kge.time_utils import QueryTime
from .checkpoints import get_gate_path


@_dataclass
class _Fact:
    """Lightweight carrier so the encoder's _to_triple / _to_triple_time find attributes."""
    triple: tuple
    happen_time: str = ""
    system_time: str = ""

logger = logging.getLogger(__name__)

# Type aliases
Triple = Tuple[str, str, str]
TimedTriple = Tuple[str, str, str, str | None]


class RoMemReranker:
    """Temporal KGE reranker using continuous phase rotation and a pretrained semantic speed gate.

    Args:
        embedding: Name of the text embedding model for relation representations.
            Bundled gate checkpoints are available for ``text-embedding-3-small``
            and ``BAAI/bge-m3``. Other models require a custom ``gate_checkpoint``.
        gate_checkpoint: Path to a pretrained speed gate checkpoint. If ``None``,
            the bundled checkpoint matching ``embedding`` is loaded automatically.
        backbone: TKGE backbone architecture (``"chronor"`` or ``"distmult"``).
        device: PyTorch device string (e.g., ``"cpu"``, ``"cuda"``).
        tkge_config: Optional dict of overrides for :class:`TKGEConfig` fields.
    """

    def __init__(
        self,
        embedding: str = "text-embedding-3-small",
        gate_checkpoint: str | None = None,
        backbone: str = "chronor",
        device: str | None = None,
        tkge_config: dict | None = None,
    ):
        self.embedding_model = embedding
        self.backbone = backbone
        self._device = device

        # Resolve gate checkpoint
        if gate_checkpoint is not None:
            self._gate_path = Path(gate_checkpoint)
            if not self._gate_path.exists():
                raise FileNotFoundError(f"Gate checkpoint not found: {gate_checkpoint}")
        else:
            self._gate_path = get_gate_path(embedding)
            if self._gate_path is None:
                raise ValueError(
                    f"No bundled gate checkpoint for embedding '{embedding}'. "
                    f"Use romem.pretrain_gate() to train one, or pass gate_checkpoint=."
                )

        # Build TKGE config
        cfg_overrides = tkge_config or {}
        defaults = dict(
            temporal_mode="romem",
            temporal_backbone=backbone,
            use_time_contrastive=True,
            time_contrastive_weight=0.5,
            time_loss_type="listwise",
            num_time_negatives=8,
            time_sigma_years=0.25,
            time_neg_jitter_years=0.02,
            time_neg_far_days=365,
            time_neg_min_days_start=90,
            time_neg_min_days_end=3,
            time_sigma_years_start=0.5,
            time_sigma_years_end=0.02,
        )
        defaults.update({k: v for k, v in cfg_overrides.items() if hasattr(TKGEConfig, k)})
        self._config = TKGEConfig(**defaults)

        self._retriever: TKGERetriever | None = None
        self._relation_embedder = None
        self._fitted = False

    def _get_relation_embedder(self):
        """Lazily initialize the text embedding model for relation representations."""
        if self._relation_embedder is None:
            from .embedding_model import _get_embedding_model_class
            from .utils.config_utils import BaseConfig

            base_cfg = BaseConfig()
            base_cfg.embedding_model_name = self.embedding_model
            EmbCls = _get_embedding_model_class(self.embedding_model)
            emb_model = EmbCls(global_config=base_cfg, embedding_model_name=self.embedding_model)
            self._relation_embedder = lambda texts: emb_model.batch_encode(texts)
        return self._relation_embedder

    def _ensure_retriever(self) -> TKGERetriever:
        if self._retriever is None:
            self._retriever = TKGERetriever(
                config=self._config,
                verbose=0,
                relation_embedder=self._get_relation_embedder(),
            )
        return self._retriever

    def fit(
        self,
        triples: Sequence[TimedTriple],
        epochs: int | None = None,
        seed: int = 0,
    ) -> "RoMemReranker":
        """Train the TKGE model on temporal triples.

        Args:
            triples: List of ``(head, relation, tail, timestamp)`` tuples.
                Timestamps should be ISO-format strings (``YYYY-MM-DD``) or ``None`` for static facts.
            epochs: Override the number of training epochs.
            seed: Random seed for reproducibility.

        Returns:
            self, for method chaining.
        """
        from .kge.utils import set_seeds
        set_seeds(seed)

        retriever = self._ensure_retriever()

        self._config.steps_per_update = epochs if epochs is not None else 500

        # Convert to the format expected by the encoder (_to_triple needs .triple attribute)
        fact_objects = []
        for h, r, t, *rest in triples:
            ts = rest[0] if rest else None
            fact_objects.append(_Fact(
                triple=(h, r, t),
                happen_time=str(ts) if ts else "",
            ))

        # Register facts (entities, relations, time) without training
        retriever.update(fact_objects, train=False)

        # Load pretrained gate (needs relation text embeddings, available after update)
        if self._gate_path is not None:
            retriever.load_time_gate_checkpoint(str(self._gate_path), strict_dim=False)
            logger.info("Loaded pretrained gate from %s", self._gate_path)

        # Train the KGE model (spectrum + embeddings, gate frozen)
        retriever.encoder._train()
        self._fitted = True
        return self

    def score(
        self,
        candidates: Sequence[Triple],
        query_time: str | None = None,
    ) -> np.ndarray:
        """Score candidate triples at a given query time.

        Args:
            candidates: List of ``(head, relation, tail)`` triples to score.
            query_time: ISO-format timestamp (``YYYY-MM-DD``). Defaults to now.

        Returns:
            Array of float scores, one per candidate. Higher = more temporally relevant.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .score()")

        if query_time:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(query_time).replace(tzinfo=timezone.utc)
            qt = QueryTime(unix_seconds=float(dt.timestamp()))
        else:
            qt = QueryTime.now()
        return self._retriever.score(list(candidates), query_time=qt)

    def rerank(
        self,
        candidates: Sequence[Triple],
        query_time: str | None = None,
        semantic_scores: np.ndarray | Sequence[float] | None = None,
        alpha: float = 0.3,
    ) -> list[dict]:
        """Rerank candidates by fusing semantic and temporal scores.

        Implements: ``S_final = S_sem * (1 + alpha * S_kge)``

        Args:
            candidates: List of ``(head, relation, tail)`` triples.
            query_time: ISO-format timestamp. Defaults to now.
            semantic_scores: Optional semantic similarity scores to fuse with.
                If ``None``, ranking is purely temporal.
            alpha: Temporal weight for multiplicative gating.

        Returns:
            List of dicts ``{"triple": (h,r,t), "score": float, "temporal_score": float}``
            sorted by descending final score.
        """
        temporal = self.score(candidates, query_time=query_time)

        if semantic_scores is not None:
            sem = np.asarray(semantic_scores, dtype=np.float32)
            final = sem * (1.0 + alpha * temporal)
        else:
            final = temporal

        results = []
        for i, (h, r, t) in enumerate(candidates):
            results.append({
                "triple": (h, r, t),
                "score": float(final[i]),
                "temporal_score": float(temporal[i]),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def get_alpha(self, relation: str) -> float:
        """Inspect the learned speed gate value for a relation string.

        Args:
            relation: Relation text (e.g., ``"president_of"``, ``"born_in"``).

        Returns:
            Gate value in ``(0, 1)``. Near 0 = static, near 1 = dynamic.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .get_alpha()")

        import torch
        encoder = self._retriever.encoder
        if encoder is None or encoder.model is None:
            return 0.0
        if not hasattr(encoder.model, 'speed_mlp') or encoder.model.speed_mlp is None:
            return 0.0

        # Compute embedding for the query relation and pass through the gate MLP
        embedder = self._get_relation_embedder()
        emb = embedder([relation])
        if emb is None or len(emb) == 0:
            return 0.0

        with torch.no_grad():
            t = torch.tensor(emb, dtype=torch.float32)
            alpha = torch.sigmoid(encoder.model.speed_mlp(t)).item()
        return alpha

    def save(self, path: str) -> None:
        """Save the trained TKGE model to disk."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .save()")
        self._retriever.save_full_checkpoint(path)
        logger.info("Saved RoMemReranker checkpoint to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        embedding: str = "text-embedding-3-small",
        backbone: str = "chronor",
    ) -> "RoMemReranker":
        """Load a previously saved RoMemReranker.

        Args:
            path: Path to the saved checkpoint.
            embedding: Embedding model name (must match the one used during training).
            backbone: TKGE backbone (must match the one used during training).
        """
        instance = cls(
            embedding=embedding,
            backbone=backbone,
            gate_checkpoint=None,  # gate is inside the full checkpoint
        )
        retriever = instance._ensure_retriever()
        retriever.load_full_checkpoint(path)
        instance._fitted = True
        logger.info("Loaded RoMemReranker from %s", path)
        return instance
