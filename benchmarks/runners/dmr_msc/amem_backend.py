"""A-mem-sys backend implementation for the DMR-MSC benchmark."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI

from benchmarks.runners.shared_utils import normalize_text, token_overlap
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)

# Ensure baselines/A-mem-sys is on sys.path so we can import agentic_memory
_AMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'A-mem-sys')
if _AMEM_ROOT not in sys.path:
    sys.path.insert(0, _AMEM_ROOT)


class DmrMscAMemBackend:
    """DMR-MSC backend that uses A-mem-sys (ChromaDB + sentence-transformers)."""

    def __init__(
        self,
        model_name: str | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
    ):
        self.model_name = (
            model_name
            or os.getenv('AMEM_EMBED_MODEL')
            or 'text-embedding-3-small'
        )
        self.embedding_provider = os.getenv('AMEM_EMBED_PROVIDER') or 'openai'
        self.llm_backend = llm_backend or os.getenv('AMEM_LLM_BACKEND') or 'openai'
        self.llm_base_url = os.getenv('AMEM_LLM_BASE_URL') or None
        self.llm_model = (
            llm_model
            or os.getenv('AMEM_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        self.search_limit = search_limit or int(os.getenv('DMR_MSC_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

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
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({size for size in ctx_values if size > 0})

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('DMR_MSC_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        llm_api_key = (
            answer_llm_api_key
            or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        llm_base_url = (
            answer_llm_base_url
            or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL')
            or os.getenv('OPENAI_BASE_URL')
            or None
        )
        self.answer_llm_client: AsyncOpenAI | None = None
        if llm_api_key:
            self.answer_llm_client = AsyncOpenAI(
                api_key=llm_api_key,
                base_url=llm_base_url,
            )

        # memory will be created fresh per example in ingest_example
        self._memory = None

    def _create_memory(self):
        """Create a fresh AgenticMemorySystem instance."""
        from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore
        return AgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            embedding_provider=self.embedding_provider,
            llm_base_url=self.llm_base_url,
        )

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    async def ingest_example(self, example: DmrMscExample) -> None:
        """Reset memory and add all episodes as notes."""
        # Create a fresh memory instance (resets ChromaDB collection internally)
        self._memory = self._create_memory()

        for idx, episode in enumerate(example.episodes):
            try:
                self._memory.add_note(episode.content)
                self.increment_episode_count()
            except Exception as exc:
                logger.warning(
                    'A-mem add_note failed for %s episode %s: %s',
                    example.example_id, idx, exc,
                )

    def _cleanup(self) -> None:
        """Reset the ChromaDB collection and free memory."""
        try:
            if self._memory is not None:
                self._memory.retriever.client.reset()
        except Exception as exc:
            logger.warning('A-mem cleanup failed: %s', exc)
        self._memory = None

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, retrieval_details = await self._retrieve_answer_support(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        self._cleanup()
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve_answer_support(
        self, example: DmrMscExample
    ) -> tuple[dict, list[str], dict]:
        facts: list[str] = []
        if self._memory is not None:
            try:
                results = self._memory.search(example.question, k=self.search_limit)
                for item in results:
                    content = item.get('content', '') or ''
                    facts.append(content.strip())
            except Exception as exc:
                logger.warning('A-mem search failed for %s: %s', example.example_id, exc)

        normalized_answer = normalize_text(example.answer)
        match_rank: int | None = None
        matching_fact = ''
        top_fact = ''
        for idx, fact in enumerate(facts, start=1):
            if idx == 1:
                top_fact = fact
            if match_rank is not None or not fact:
                continue
            if self._fact_matches_answer(fact, normalized_answer):
                match_rank = idx
                matching_fact = fact

        retrieval_metrics: dict[str, float] = {
            'retrieval_rank': float(match_rank or 0),
            'retrieval_hit@1': 1.0 if match_rank == 1 else 0.0,
            'retrieval_hit@3': 1.0 if match_rank and match_rank <= 3 else 0.0,
            'retrieval_mrr': 1.0 / match_rank if match_rank else 0.0,
            'retrieval_results_returned': float(len(facts)),
        }
        details = {
            'matching_fact': matching_fact,
            'top_fact': top_fact,
        }
        return retrieval_metrics, facts, details

    async def _answer_with_llm(
        self, example: DmrMscExample, context_docs: Iterable[str]
    ) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes:
            return metrics, answers
        doc_list = list(context_docs)
        if not doc_list:
            return metrics, answers
        client = self.answer_llm_client
        if not client:
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
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            generated_answer = ''
            try:
                response = await client.chat.completions.create(
                    model=self.answer_llm_model,
                    messages=messages,
                )
                raw = response.choices[0].message.content or ''
                # Parse JSON response
                try:
                    parsed = json.loads(raw)
                    generated_answer = parsed.get('answer', raw)
                except json.JSONDecodeError:
                    generated_answer = raw.strip()
            except Exception as exc:
                logger.warning(
                    'A-mem answer failed for %s (top-%s context): %s',
                    example.example_id, k, exc,
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

    @staticmethod
    def _fact_matches_answer(fact: str, normalized_answer: str) -> bool:
        norm_fact = normalize_text(fact)
        if not norm_fact:
            return False
        if normalized_answer in norm_fact or norm_fact in normalized_answer:
            return True
        return token_overlap(norm_fact, normalized_answer) >= 0.6

    @staticmethod
    def _facts_equal(a: str, b: str) -> bool:
        return normalize_text(a) == normalize_text(b)

    @staticmethod
    def _f1_score(predicted: str, reference: str) -> float:
        pred_tokens = list(filter(None, normalize_text(predicted).split()))
        ref_tokens = list(filter(None, normalize_text(reference).split()))
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = 0
        ref_counts: dict[str, int] = {}
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
