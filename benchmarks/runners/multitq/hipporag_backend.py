"""HippoRAG backend for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import logging
import os
from typing import Iterable

from pydantic import BaseModel

from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.runners.base_backend import HippoRAGBackendBase
from benchmarks.types import TemporalTriple, MultiTQQuestion

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class MultiTQHippoRAGBackend(HippoRAGBackendBase):
    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        hippo_save_dir: str | None = None,
        hippo_llm_model: str | None = None,
        hippo_embedding_model: str | None = None,
        hippo_openie_mode: str | None = None,
    ):
        super().__init__(
            hippo_save_dir=hippo_save_dir,
            hippo_llm_model=hippo_llm_model,
            hippo_embedding_model=hippo_embedding_model,
            hippo_openie_mode=hippo_openie_mode,
        )
        self.search_limit = search_limit or int(os.getenv("MULTITQ_SEARCH_LIMIT", "50"))
        self._hippo_llm_model = hippo_llm_model or os.getenv("HIPPO_LLM_MODEL")

        ctx_values: list[int] = []
        ctx_env = os.getenv("MULTITQ_CONTEXT_K")
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        elif ctx_env:
            ctx_values = [int(p) for p in ctx_env.split(",") if p.strip().isdigit()]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({k for k in ctx_values if k > 0})

        self.answer_llm_client = self._initialize_answer_llm(
            answer_llm_model or os.getenv("MULTITQ_ANSWER_LLM_MODEL"),
            answer_llm_api_key or os.getenv("MULTITQ_ANSWER_LLM_API_KEY"),
            answer_llm_base_url or os.getenv("MULTITQ_ANSWER_LLM_BASE_URL"),
        )

    # ── Ingestion ──────────────────────────────────────────────────

    async def ingest_all_triples(self, triples: Iterable[TemporalTriple]) -> None:
        """Ingest all KG triples in a single batch (no LLM extraction)."""
        triples = list(triples)
        if not triples:
            return
        docs = [self._triple_to_sentence(t) for t in triples]
        triples_per_doc = [[(t.head, t.relation, t.tail)] for t in triples]
        self._hipporag_index_structured(docs, triples_per_doc)
        self.increment_episode_count(len(docs))
        logger.info("MultiTQ ingest mode=hipporag(structured) count=%d", len(docs))

    # ── QA evaluation ──────────────────────────────────────────────

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

    async def _retrieve(self, question: MultiTQQuestion) -> tuple[dict, list, dict]:
        try:
            retrieval = self.hipporag.retrieve(
                queries=[question.question], num_to_retrieve=self.search_limit,
            )
        except Exception as exc:
            logger.warning("HippoRAG search failed for quid=%s: %s", question.quid, exc)
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

    async def _answer(self, question: MultiTQQuestion, context_docs: list[str]) -> tuple[dict, dict[int, str]]:
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
                'Respond with JSON {{"answer": "<text>"}}.'
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

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _triple_to_sentence(triple: TemporalTriple) -> str:
        date_str = triple.timestamp.strftime("%Y-%m-%d")
        return f"On {date_str}, {triple.head} {triple.relation} {triple.tail}."

    @staticmethod
    def _check_answer_in_context(answers: list[str], docs: list[str]) -> bool:
        context = " ".join(docs).lower().replace("_", " ")
        for answer in answers:
            if answer.lower().replace("_", " ") in context:
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

    def _initialize_answer_llm(self, model: str | None, api_key: str | None, base_url: str | None):
        if not model:
            model = self._hippo_llm_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            return None
        config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        name = (config.model or "").lower()
        if not name or name.startswith("gpt"):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)
