"""RoMem backend for MultiTQ temporal KGQA evaluation.

Combines triple ingestion with retrieval-augmented QA evaluation.
Answer verification uses MemoTime-consistent rule-based multi-strategy matching.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import (
    initialize_answer_llm,
    parse_bool,
    parse_optional_bool,
)
from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.types import TemporalTriple, MultiTQQuestion
from benchmarks.runners.romem_utils import apply_romem_config

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


def _safe_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "multitq")




class MultiTQRoMemBackend:
    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
        exp_name: str | None = None,
    ) -> None:
        self.search_limit = search_limit or int(os.getenv("MULTITQ_SEARCH_LIMIT", "50"))

        ctx_values: list[int] = []
        ctx_env = os.getenv("MULTITQ_CONTEXT_K")
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        elif ctx_env:
            ctx_values = [int(p) for p in ctx_env.split(",") if p.strip().isdigit()]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({k for k in ctx_values if k > 0})

        self.answer_llm_client = initialize_answer_llm(
            answer_llm_model or os.getenv("MULTITQ_ANSWER_LLM_MODEL"),
            answer_llm_api_key or os.getenv("MULTITQ_ANSWER_LLM_API_KEY"),
            answer_llm_base_url or os.getenv("MULTITQ_ANSWER_LLM_BASE_URL"),
        )

        self._romem_root = Path(
            romem_save_dir or os.getenv("TEMPUS_SAVE_DIR") or "outputs/romem_multitq"
        )
        self._romem_llm_model = romem_llm_model or os.getenv("TEMPUS_LLM_MODEL")
        self._romem_embedding_model = romem_embedding_model or os.getenv("TEMPUS_EMBED_MODEL")
        self._romem_openie_mode = romem_openie_mode or os.getenv("TEMPUS_OPENIE_MODE")
        self._romem_config = romem_config or {}
        self._romem_temporal_awareness = parse_bool(
            romem_temporal_awareness or os.getenv("TEMPUS_TEMPORAL_AWARENESS"),
            default=True,
        )
        enable_tkge_value = romem_enable_tkge_tunnel
        if enable_tkge_value is None:
            enable_tkge_value = os.getenv("TEMPUS_ENABLE_TKGE_TUNNEL")
        self._romem_enable_tkge_tunnel = parse_optional_bool(enable_tkge_value)
        self._romem_tkge_verbose = romem_tkge_verbose
        self._exp_name = exp_name or "multitq"
        self.romem = self._init_romem(self._exp_name)

    def _init_romem(self, run_name: str):
        from romem import RoMem
        from romem.utils.config_utils import BaseConfig

        run_dir = self._romem_root / _safe_dir_name(run_name)
        config = BaseConfig()
        apply_romem_config(config, self._romem_config)
        config.save_dir = str(run_dir)
        if self._romem_llm_model:
            config.llm_name = self._romem_llm_model
        if self._romem_embedding_model:
            config.embedding_model_name = self._romem_embedding_model
        if self._romem_openie_mode:
            config.openie_mode = self._romem_openie_mode
        config.temporal_awareness = self._romem_temporal_awareness
        if self._romem_enable_tkge_tunnel is not None:
            config.enable_tkge_tunnel = self._romem_enable_tkge_tunnel
        if self._romem_tkge_verbose is not None:
            config.tkge_verbose = int(self._romem_tkge_verbose)
        config.tkge_verbose_epoch_interval = 1
        config.retrieval_top_k = self.search_limit
        return RoMem(
            global_config=config,
            save_dir=config.save_dir,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
        )

    # ── Ingestion ────────────────────────────────────────────────

    async def ingest_all_triples(
        self, triples: Iterable[TemporalTriple], *, train_tkge: bool = True,
    ) -> None:
        """Ingest all KG triples in a single batch."""
        triples = list(triples)
        if not triples:
            return
        from romem.utils.misc_utils import TimedTriple

        docs = [self._triple_to_sentence(t) for t in triples]
        timed_triples = [
            TimedTriple(
                triple=(t.head, t.relation, t.tail),
                happen_time=t.timestamp.strftime("%Y-%m-%d"),
                system_time=t.timestamp.isoformat(),
            )
            for t in triples
        ]
        self.romem.index_triples(
            docs=docs,
            timed_triples=timed_triples,
            embed_texts=docs,
            train_tkge=train_tkge,
        )
        logger.info("MultiTQ ingest mode=romem count=%d", len(docs))

    def train_tkge(self) -> None:
        """Trigger TKGE training once on the complete graph."""
        self.romem.train_tkge()

    def tkge_checkpoint_path(self) -> str:
        """Return the default TKGE checkpoint path in the outputs dir."""
        return str(Path(self.romem.global_config.save_dir) / "tkge_model.pt")

    def save_tkge_checkpoint(self) -> dict:
        """Save full TKGE model to disk."""
        path = self.tkge_checkpoint_path()
        info = self.romem.save_tkge_checkpoint(path)
        logger.info("TKGE checkpoint saved to %s (train_calls=%s)", path, info.get("train_calls"))
        return info

    def load_tkge_checkpoint(self) -> dict | None:
        """Load TKGE model from disk if a checkpoint exists. Returns info dict or None."""
        path = self.tkge_checkpoint_path()
        if not Path(path).exists():
            return None
        try:
            info = self.romem.load_tkge_checkpoint(path)
            logger.info(
                "TKGE checkpoint loaded from %s (train_calls=%s, entities=%s, relations=%s)",
                path, info.get("train_calls"), info.get("num_entities"), info.get("num_relations"),
            )
            return info
        except Exception as exc:
            logger.warning("Failed to load TKGE checkpoint %s: %s", path, exc)
            return None

    # ── QA evaluation ────────────────────────────────────────────

    async def evaluate_question(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        retrieval_metrics, context_docs, details = await self._retrieve(question)
        llm_metrics, answers = await self._answer(question, context_docs)
        accuracy_metrics, match_strategies = self._verify_answers(question, answers)
        return (
            retrieval_metrics,
            {**llm_metrics, **accuracy_metrics},
            {**details, "llm_answers": answers, "context_docs": context_docs,
             "match_strategies": match_strategies},
        )

    async def _retrieve(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], list[str], dict]:
        try:
            retrieval = self.romem.retrieve(
                queries=[question.question],
                num_to_retrieve=self.search_limit,
            )
        except Exception as exc:
            logger.warning("RoMem search failed for quid=%s: %s", question.quid, exc)
            retrieval = []

        docs: list[str] = []
        if retrieval:
            docs = retrieval[0].docs

        metrics: dict[str, float] = {}
        matching_fact = ""
        for k in self.answer_context_sizes:
            subset = docs[:k]
            found = self._check_answer_in_context(question.answers, subset)
            metrics[f"answer_in_context@{k}"] = 1.0 if found else 0.0
            if found and not matching_fact:
                matching_fact = self._find_matching_fact(question.answers, subset)

        # Hits@k and MRR — find rank of first doc containing a gold answer
        match_rank: int | None = None
        for idx, doc in enumerate(docs, start=1):
            if not doc:
                continue
            doc_lower = doc.lower().replace("_", " ")
            for answer in question.answers:
                if answer.lower().replace("_", " ") in doc_lower:
                    match_rank = idx
                    break
            if match_rank is not None:
                break
        metrics["hits@1"] = 1.0 if match_rank == 1 else 0.0
        metrics["hits@3"] = 1.0 if match_rank is not None and match_rank <= 3 else 0.0
        metrics["hits@10"] = 1.0 if match_rank is not None and match_rank <= 10 else 0.0
        metrics["mrr"] = 1.0 / match_rank if match_rank else 0.0

        return metrics, docs, {"matching_fact": matching_fact}

    async def _answer(
        self, question: MultiTQQuestion, context_docs: list[str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None:
            return metrics, answers
        for k in self.answer_context_sizes:
            subset = context_docs[:k]
            if not subset:
                metrics[f"llm@{k}_context_size"] = 0.0
                continue
            facts = [f"- {doc}" for doc in subset if doc]
            if not facts:
                metrics[f"llm@{k}_context_size"] = 0.0
                continue
            prompt = (
                f"Question: {question.question}\n\n"
                f"Supporting facts:\n{os.linesep.join(facts)}\n\n"
                'Provide a concise answer grounded in the facts. '
                'Respond with JSON {"answer": "<text>"}.'
            )
            messages = [
                Message(role="system", content="You answer questions using only provided facts."),
                Message(role="user", content=prompt),
            ]
            prediction = ""
            try:
                response = await client.generate_response(messages, response_model=QAResponse)
                prediction = response.get("answer", "")
            except Exception as exc:
                logger.warning("LLM answer failed for quid=%s (top-%s): %s", question.quid, k, exc)
            answers[k] = prediction
            metrics[f"llm@{k}_context_size"] = float(len(facts))
        return metrics, answers

    def _verify_answers(
        self, question: MultiTQQuestion, answers: dict[int, str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        strategies: dict[int, str] = {}
        for k, prediction in answers.items():
            is_correct, strategy = verify_answer(
                prediction, question.answers, question.answer_type,
            )
            metrics[f"llm@{k}_accuracy"] = 1.0 if is_correct else 0.0
            strategies[k] = strategy
        return metrics, strategies

    def finalize_usage_logging(self) -> None:
        return

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _triple_to_sentence(triple: TemporalTriple) -> str:
        date_str = triple.timestamp.strftime("%Y-%m-%d")
        return f"On {date_str}, {triple.head} {triple.relation} {triple.tail}."

    @staticmethod
    def _check_answer_in_context(answers: list[str], docs: list[str]) -> bool:
        context = " ".join(docs).lower().replace("_", " ")
        for answer in answers:
            normalized = answer.lower().replace("_", " ")
            if normalized in context:
                return True
        return False

    @staticmethod
    def _find_matching_fact(answers: list[str], docs: list[str]) -> str:
        for doc in docs:
            doc_lower = doc.lower().replace("_", " ")
            for answer in answers:
                if answer.lower().replace("_", " ") in doc_lower:
                    return doc
        return ""

