"""
RoMemLLM — Memory-augmented LLM with temporal awareness.

Wraps an LLM with RoMem's full pipeline: ingestion (OpenIE + graph),
temporal KGE reranking, and answer generation.

Usage:
    from romem import RoMemLLM

    llm = RoMemLLM(llm="meta-llama/Llama-3.1-8B-Instruct")
    llm.add("Obama was president from 2009 to 2017.")
    llm.add("Biden became president in January 2021.")
    answer = llm.ask("Who was president in 2019?")  # → Obama
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from .checkpoints import get_gate_path

logger = logging.getLogger(__name__)


class RoMemLLM:
    """Memory-augmented LLM with temporal awareness via continuous phase rotation.

    Args:
        llm: Model name for extraction and answer generation. Supports OpenAI model
            names (``"gpt-4o-mini"``) or HuggingFace model IDs
            (``"meta-llama/Llama-3.1-8B-Instruct"``).
        embedding: Text embedding model for graph construction and retrieval.
            Bundled gate checkpoints are available for ``text-embedding-3-small``
            and ``BAAI/bge-m3``.
        save_dir: Directory for persistent storage (graph, TKGE checkpoint, embeddings).
        gate_checkpoint: Path to a custom pretrained speed gate. If ``None``,
            the bundled checkpoint matching ``embedding`` is loaded automatically.
        answer_llm: Optional separate model for answer generation. If ``None``,
            uses ``llm``.
        api_key: API key for the LLM provider. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Custom API base URL (e.g., for vLLM-served local models).
        config: Configuration overrides. Accepts a dict, or a path to a JSON
            config file (e.g., ``"romem/configs/locomo_openai.json"``).
            Keys are applied as attributes on :class:`BaseConfig`.
    """

    def __init__(
        self,
        llm: str = "gpt-4o-mini",
        embedding: str = "text-embedding-3-small",
        save_dir: str = "./romem_store",
        gate_checkpoint: str | None = None,
        answer_llm: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        config: dict | str | None = None,
    ):
        # Load config from file if a path is given
        if isinstance(config, str):
            import json
            config_path = Path(config)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config_path, "r") as f:
                config = json.load(f)

        self.llm_model = llm
        self.embedding_model = embedding
        self.save_dir = Path(save_dir)
        self.answer_llm_model = answer_llm or llm
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._user_config = config or {}

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

        self._romem = None  # lazy init

    def _ensure_initialized(self):
        """Lazily initialize the full RoMem pipeline."""
        if self._romem is not None:
            return

        from .RoMem import RoMem
        from .utils.config_utils import BaseConfig

        # Build BaseConfig with production-tuned temporal defaults.
        # These match the settings used in our paper experiments.
        cfg = BaseConfig()

        # Core temporal pipeline
        cfg.enable_tkge_tunnel = True
        cfg.tkge_temporal_mode = "romem"
        cfg.tkge_temporal_backbone = "chronor"
        cfg.tkge_time_source = "happen"
        cfg.openie_mode = "online"
        cfg.use_llm_fact_filter = True

        # Time contrastive loss
        cfg.tkge_time_contrastive_weight = 0.5
        cfg.tkge_time_loss_type = "listwise"
        cfg.tkge_num_time_negatives = 8
        cfg.tkge_time_sigma_years = 0.25
        cfg.tkge_time_sigma_years_start = 0.5
        cfg.tkge_time_sigma_years_end = 0.02
        cfg.tkge_time_sigma_decay_epochs = 60

        # Negative sampling curriculum
        cfg.tkge_time_neg_jitter_years = 0.02
        cfg.tkge_time_neg_far_days = 365
        cfg.tkge_time_neg_min_days_start = 90
        cfg.tkge_time_neg_min_days_end = 3
        cfg.tkge_time_neg_min_days_decay_epochs = 60

        # Training schedule
        cfg.tkge_temporal_warmup_epochs = 50
        cfg.tkge_checkpoint_start_epoch = 60

        # Gate checkpoint
        if self._gate_path is not None:
            cfg.tkge_time_gate_checkpoint = str(self._gate_path)

        # Apply user overrides (from dict or config file)
        for k, v in self._user_config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        self._romem = RoMem(
            global_config=cfg,
            save_dir=str(self.save_dir),
            llm_model_name=self.llm_model,
            embedding_model_name=self.embedding_model,
            llm_base_url=self.base_url,
        )

    def add(
        self,
        text: str,
        timestamp: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Ingest a piece of text into memory.

        Runs temporal OpenIE to extract facts, builds the knowledge graph,
        and incrementally updates the TKGE model.

        Args:
            text: The text content to ingest.
            timestamp: Optional ISO-format timestamp (``YYYY-MM-DD``) for when
                this information was valid or observed.
            metadata: Optional metadata dict attached to the episode.
        """
        self._ensure_initialized()
        self._romem.index(
            docs=[text],
            observed_time=timestamp,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        time: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant facts with temporal awareness.

        Args:
            query: The question or search query.
            top_k: Number of facts to return.
            time: Optional ISO-format timestamp to query at. Defaults to now.

        Returns:
            List of dicts with keys ``"fact"``, ``"score"``, ``"timestamp"``.
        """
        self._ensure_initialized()
        solutions = self._romem.retrieve(
            queries=[query],
            num_to_retrieve=top_k,
            query_time_overrides=[time] if time else None,
        )
        if not solutions:
            return []

        sol = solutions[0] if isinstance(solutions, list) else solutions
        docs = getattr(sol, 'docs', []) or []
        scores = getattr(sol, 'doc_scores', None)
        results = []
        for i, doc in enumerate(docs[:top_k]):
            score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            results.append({"fact": str(doc), "score": score})
        return results

    def ask(
        self,
        question: str,
        top_k: int = 5,
        time: str | None = None,
    ) -> str:
        """Retrieve relevant facts and generate an answer.

        Args:
            question: The question to answer.
            top_k: Number of facts to use as context.
            time: Optional ISO-format timestamp. Defaults to now.

        Returns:
            The generated answer string.
        """
        context = self.retrieve(question, top_k=top_k, time=time)
        if not context:
            return "I don't have enough information to answer this question."

        facts_text = "\n".join(f"- {c['fact']}" for c in context)
        prompt = (
            f"Answer the following question using only the provided facts.\n\n"
            f"Facts:\n{facts_text}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """Generate a response using the answer LLM."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.answer_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            return f"Error generating answer: {e}"
