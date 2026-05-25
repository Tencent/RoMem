"""HippoRAG backend implementation for the DMR-MSC benchmark."""

from __future__ import annotations

import logging
import os
from typing import Iterable

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import (
    AnswerResponse,
    initialize_answer_llm,
    normalize_text,
    token_overlap,
)
from benchmarks.runners.base_backend import HippoRAGBackendBase
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class DmrMscHippoRAGBackend(HippoRAGBackendBase):
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
        self.search_limit = search_limit or int(os.getenv('DMR_MSC_SEARCH_LIMIT', '50'))
        ctx_values: list[int] = []
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        else:
            ctx_env = os.getenv('DMR_MSC_CONTEXT_K')
            if ctx_env:
                ctx_values = [
                    int(value.strip())
                    for value in ctx_env.split(',')
                    if value.strip().isdigit()
                ]
            else:
                legacy = os.getenv('DMR_MSC_CONTEXT_SIZE')
                if legacy and legacy.isdigit():
                    ctx_values = [int(legacy)]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({size for size in ctx_values if size > 0})
        self.answer_llm_client = initialize_answer_llm(
            model=answer_llm_model or os.getenv('DMR_MSC_ANSWER_LLM_MODEL'),
            api_key=answer_llm_api_key
            or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY'),
            base_url=answer_llm_base_url
            or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL')
            or os.getenv('OPENAI_BASE_URL'),
        )

    async def ingest_example(self, example: DmrMscExample) -> None:
        self._reset_hipporag(sample_id=example.example_id)
        docs = [episode.content for episode in example.episodes]
        if docs:
            self.hipporag.index(docs=docs)
            self.increment_episode_count(len(docs))

    async def clear_graph(self) -> None:
        self._reset_hipporag(sample_id=None)

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, retrieval_details = await self._retrieve_answer_support(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        await self.clear_graph()
        details = {
            **retrieval_details,
            'llm_answers': llm_answers,
        }
        return retrieval_metrics, llm_metrics, details

    async def _retrieve_answer_support(
        self, example: DmrMscExample
    ) -> tuple[dict, list[str], dict]:
        try:
            retrieval = self.hipporag.retrieve(
                [example.question],
                num_to_retrieve=self.search_limit,
            )
            docs = retrieval[0].docs if retrieval else []
            facts = retrieval[0].facts if retrieval else []
        except Exception as exc:
            logger.warning('HippoRAG search failed for %s: %s', example.example_id, exc)
            docs = []
            facts = []

        normalized_answer = normalize_text(example.answer)
        match_rank: int | None = None
        matching_fact = ''
        top_fact = ''
        for idx, doc_str in enumerate(docs, start=1):
            doc_str = (doc_str or '').strip()
            if idx == 1:
                top_fact = doc_str
            if match_rank is not None or not doc_str:
                continue
            if self._fact_matches_answer(doc_str, normalized_answer):
                match_rank = idx
                matching_fact = doc_str

        retrieval_metrics: dict[str, float] = {
            'retrieval_rank': float(match_rank or 0),
            'retrieval_hit@1': 1.0 if match_rank == 1 else 0.0,
            'retrieval_hit@3': 1.0 if match_rank and match_rank <= 3 else 0.0,
            'retrieval_mrr': 1.0 / match_rank if match_rank else 0.0,
            'retrieval_results_returned': float(len(docs)),
        }
        details = {
            'matching_fact': matching_fact,
            'top_fact': top_fact,
        }
        return retrieval_metrics, docs, details

    async def _answer_with_llm(
        self, example: DmrMscExample, context_docs: Iterable[str]
    ) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes or self.answer_llm_client is None:
            return metrics, answers
        doc_list = list(context_docs)
        if not doc_list:
            return metrics, answers

        for k in self.answer_context_sizes:
            subset = [doc for doc in doc_list[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_exact'] = 0.0
                metrics[f'llm@{k}_f1'] = 0.0
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            system_prompt = (
                'You are a careful assistant. Use only the provided facts to answer the question.'
            )
            user_prompt = f"""Question: {example.question}

Supporting facts:
{os.linesep.join(f'- {doc}' for doc in subset)}

Provide a short answer grounded in the facts. Respond with JSON as {{"answer": "<text>"}}."""
            messages = [
                Message(role='system', content=system_prompt),
                Message(role='user', content=user_prompt),
            ]
            generated_answer = ''
            try:
                response = await self.answer_llm_client.generate_response(
                    messages,
                    response_model=AnswerResponse,
                )
                generated_answer = response.get('answer', '')
            except Exception as exc:
                logger.warning(
                    'HippoRAG answer failed for %s (top-%s context): %s',
                    example.example_id,
                    k,
                    exc,
                )
                generated_answer = ''

            answers[k] = generated_answer
            exact_match = 1.0 if self._facts_equal(generated_answer, example.answer) else 0.0
            f1 = self._f1_score(generated_answer, example.answer)
            is_correct, _ = await score_answer(example.question, example.answer, generated_answer)
            metrics[f'llm@{k}_exact'] = exact_match
            metrics[f'llm@{k}_f1'] = f1
            metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0
            metrics[f'llm@{k}_context_size'] = float(len(subset))

        return metrics, answers

    def _fact_matches_answer(self, fact: str, normalized_answer: str) -> bool:
        if not fact or not normalized_answer:
            return False
        normalized_fact = normalize_text(fact)
        if not normalized_fact:
            return False
        if normalized_answer in normalized_fact or normalized_fact in normalized_answer:
            return True
        overlap = token_overlap(normalized_fact, normalized_answer)
        return overlap >= 0.6

    def _facts_equal(self, predicted: str, reference: str) -> bool:
        return normalize_text(predicted) == normalize_text(reference)

    def _f1_score(self, predicted: str, reference: str) -> float:
        pred_tokens = list(filter(None, normalize_text(predicted).split()))
        ref_tokens = list(filter(None, normalize_text(reference).split()))
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = 0
        ref_counts = {}
        for token in ref_tokens:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                common += 1
                ref_counts[token] -= 1
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

