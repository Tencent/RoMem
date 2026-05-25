"""Zep Cloud backend implementation for the DMR-MSC benchmark."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Iterable

from openai import AsyncOpenAI

from benchmarks.runners.shared_utils import normalize_text, token_overlap
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class DmrMscZepBackend:
    """DMR-MSC backend that uses the Zep Cloud Graph API."""

    def __init__(
        self,
        zep_api_key: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        ingest_poll_interval: float = 2.0,
        ingest_poll_timeout: float = 300.0,
    ):
        api_key = zep_api_key or os.getenv('ZEP_API_KEY')
        if not api_key:
            raise ValueError(
                'Zep API key is required. Set ZEP_API_KEY env var or pass zep_api_key.'
            )
        from zep_cloud.client import AsyncZep  # type: ignore

        self.client = AsyncZep(api_key=api_key)
        self.search_limit = search_limit or int(os.getenv('DMR_MSC_SEARCH_LIMIT', '50'))
        self.ingest_poll_interval = ingest_poll_interval
        self.ingest_poll_timeout = ingest_poll_timeout
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

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    async def ingest_example(self, example: DmrMscExample) -> str:
        """Create a Zep user, add episodes via graph API, and wait for processing."""
        user_id = f'dmr_msc_{example.example_id}'
        # Delete any stale user/graph from a prior failed run, then create fresh
        try:
            await self.client.user.delete(user_id)
            logger.debug('Deleted stale Zep user %s before re-creating', user_id)
        except Exception:
            pass  # User didn't exist — expected for first run
        try:
            await self.client.user.add(user_id=user_id)
        except Exception as exc:
            logger.warning('Zep user.add failed for %s: %s', user_id, exc)

        episode_uuids: list[str] = []
        for idx, episode in enumerate(example.episodes):
            try:
                result = await self.client.graph.add(
                    user_id=user_id,
                    type='text',
                    data=episode.content,
                )
                if result and hasattr(result, 'uuid'):
                    episode_uuids.append(result.uuid)
                self.increment_episode_count()
            except Exception as exc:
                logger.warning(
                    'Zep graph.add failed for %s episode %s: %s',
                    example.example_id, idx, exc,
                )

        # Wait for async processing to complete
        await self._wait_for_processing(user_id, episode_uuids)
        return user_id

    async def _wait_for_processing(
        self, user_id: str, episode_uuids: list[str]
    ) -> None:
        """Poll Zep until all episodes are processed or timeout is reached."""
        if not episode_uuids:
            # No UUIDs to track — fall back to a fixed wait
            await asyncio.sleep(min(self.ingest_poll_timeout, 30.0))
            return

        elapsed = 0.0
        while elapsed < self.ingest_poll_timeout:
            all_done = True
            for uuid in episode_uuids:
                try:
                    ep = await self.client.graph.episode.get(uuid)
                    status = getattr(ep, 'status', None) or getattr(ep, 'processed', None)
                    if status in (None, 'pending', 'processing', False):
                        all_done = False
                        break
                except Exception:
                    # Episode might not be queryable yet
                    all_done = False
                    break
            if all_done:
                logger.debug('All %s episodes processed for user %s', len(episode_uuids), user_id)
                return
            await asyncio.sleep(self.ingest_poll_interval)
            elapsed += self.ingest_poll_interval

        logger.warning(
            'Zep processing timeout (%.0fs) for user %s with %s episodes',
            self.ingest_poll_timeout, user_id, len(episode_uuids),
        )

    async def cleanup_user(self, user_id: str) -> None:
        """Delete the Zep user (and all associated graph data)."""
        try:
            await self.client.user.delete(user_id)
        except Exception as exc:
            logger.warning('Zep user.delete failed for %s: %s', user_id, exc)

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        user_id = await self.ingest_example(example)
        retrieval_metrics, context, retrieval_details = await self._retrieve_answer_support(
            example, user_id
        )
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        await self.cleanup_user(user_id)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve_answer_support(
        self, example: DmrMscExample, user_id: str
    ) -> tuple[dict, list[str], dict]:
        facts: list[str] = []
        try:
            results = await self.client.graph.search(
                query=example.question,
                user_id=user_id,
                scope='edges',
                limit=self.search_limit,
                reranker='rrf',
            )
            if results and hasattr(results, 'edges'):
                for edge in results.edges:
                    fact = getattr(edge, 'fact', '') or ''
                    facts.append(fact.strip())
            elif isinstance(results, list):
                for item in results:
                    fact = getattr(item, 'fact', '') or ''
                    facts.append(fact.strip())
        except Exception as exc:
            logger.warning('Zep graph.search failed for %s: %s', example.example_id, exc)

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
                    'Zep answer failed for %s (top-%s context): %s',
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
