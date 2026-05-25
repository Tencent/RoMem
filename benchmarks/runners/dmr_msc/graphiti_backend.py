"""Graphiti backend implementation for the DMR-MSC benchmark."""

from __future__ import annotations

import logging
import os
from typing import Iterable

from tqdm import tqdm

from baselines.graphiti.graphiti_core.nodes import EpisodeType
from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from benchmarks.runners.shared_utils import (
    AnswerResponse,
    normalize_text,
    token_overlap,
)
from benchmarks.runners.base_backend import GraphitiBackendBase
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class DmrMscGraphitiBackend(GraphitiBackendBase):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        reset_mode: str = 'delete',
        search_limit: int | None = None,
        answer_context_size: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        search_reranker: str | None = None,
    ):
        super().__init__(
            uri,
            user,
            password,
            database=database,
            usage_interval_env='DMR_MSC_USAGE_INTERVAL',
            search_reranker=search_reranker,
        )
        self.reset_mode = reset_mode.lower()
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
            elif answer_context_size:
                ctx_values = [answer_context_size]
            else:
                legacy = os.getenv('DMR_MSC_CONTEXT_SIZE')
                if legacy and legacy.isdigit():
                    ctx_values = [int(legacy)]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({size for size in ctx_values if size > 0})
        self.answer_llm_client = self._init_answer_llm_client(
            answer_llm_model or os.getenv('DMR_MSC_ANSWER_LLM_MODEL'),
            answer_llm_api_key or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY'),
            answer_llm_base_url or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL'),
        )

    async def ingest_example(self, example: DmrMscExample) -> None:
        progress = tqdm(
            example.episodes,
            desc=f'Ingesting DMR sessions ({example.example_id})',
            leave=False,
        )
        for idx, episode in enumerate(progress):
            source_description = (
                episode.metadata.get('segment')
                or episode.metadata.get('type')
                or example.window_id
            )
            try:
                await self.graphiti.add_episode(
                    name=f'{example.example_id}-episode-{idx}',
                    episode_body=episode.content,
                    source_description=str(source_description),
                    reference_time=episode.reference_time,
                    source=EpisodeType.message,
                )
            except Exception as exc:
                logger.warning(
                    'add_episode failed for %s episode %s, skipping: %s',
                    example.example_id, idx, exc,
                )
            self.increment_episode_count()
        progress.close()

    async def clear_graph(self) -> None:
        if self.reset_mode == 'delete':
            # Batched delete to avoid Neo4j transaction memory limits
            while True:
                result = await self.graphiti.driver.execute_query(
                    'MATCH (n) WITH n LIMIT 1000 DETACH DELETE n RETURN count(*) AS deleted',
                    params={},
                )
                records = result[0] if isinstance(result, tuple) else result
                deleted = 0
                if isinstance(records, list) and records:
                    row = records[0]
                    deleted = row.get('deleted', 0) if isinstance(row, dict) else row[0]
                if deleted == 0:
                    break
        elif self.reset_mode == 'recreate':
            await self.graphiti.driver.close()
            self._init_graphiti()
        else:
            raise ValueError(f'Unknown reset_mode: {self.reset_mode}')

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        await self.clear_graph()
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
    ) -> tuple[dict, list, dict]:
        try:
            results = await self._search_edges(example.question, self.search_limit)
        except Exception as exc:
            logger.warning('Search failed for %s: %s', example.example_id, exc)
            results = []

        normalized_answer = normalize_text(example.answer)
        match_rank: int | None = None
        matching_fact = ''
        top_fact = ''
        facts: list[str] = []
        for idx, edge in enumerate(results, start=1):
            fact = (getattr(edge, 'fact', '') or '').strip()
            facts.append(fact)
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
            'retrieval_results_returned': float(len(results)),
        }
        details = {
            'matching_fact': matching_fact,
            'top_fact': top_fact,
        }
        return retrieval_metrics, results, details

    async def _answer_with_llm(
        self, example: DmrMscExample, context_edges: Iterable
    ) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes:
            return metrics, answers
        context_list = list(context_edges)
        context_strings = []
        for edge in context_list:
            fact = getattr(edge, 'fact', '') or ''
            if not fact:
                continue
            context_strings.append(f'- {fact}')

        if not context_list or not context_strings:
            return metrics, answers

        client = self.answer_llm_client or self.graphiti.llm_client

        for k in self.answer_context_sizes:
            subset = context_list[:k]
            formatted_subset = []
            for edge in subset:
                fact = getattr(edge, 'fact', '') or ''
                if not fact:
                    continue
                formatted_subset.append(f'- {fact}')
            if not formatted_subset:
                metrics[f'llm@{k}_exact'] = 0.0
                metrics[f'llm@{k}_f1'] = 0.0
                metrics[f'llm@{k}_context_size'] = 0.0
                continue

            system_prompt = (
                'You are a careful assistant. Use only the provided facts to answer the question.'
            )
            user_prompt = f"""Question: {example.question}

Supporting facts:
{os.linesep.join(formatted_subset)}

Provide a short answer grounded in the facts. Respond with JSON as {{"answer": "<text>"}}."""
            messages = [
                Message(role='system', content=system_prompt),
                Message(role='user', content=user_prompt),
            ]
            generated_answer = ''
            try:
                response = await client.generate_response(
                    messages,
                    response_model=AnswerResponse,
                )
                generated_answer = response.get('answer', '')
            except Exception as exc:
                logger.warning(
                    'Answer generation failed for %s (top-%s context): %s',
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
            metrics[f'llm@{k}_context_size'] = float(len(formatted_subset))

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

    def _init_answer_llm_client(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        if not any([model, api_key, base_url]):
            return self.graphiti.llm_client
        config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        model_name = (config.model or '').lower()
        if not model_name or model_name.startswith('gpt'):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)
