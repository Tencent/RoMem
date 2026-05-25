"""Graphiti backend for LoCoMo benchmarking."""

from __future__ import annotations

import logging
import os
from typing import List

from tqdm import tqdm

from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from baselines.graphiti.graphiti_core.nodes import EpisodeType
from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from benchmarks.runners.shared_utils import AnswerResponse
from benchmarks.runners.base_backend import GraphitiBackendBase
from benchmarks.runners.locomo.qa_utils import (
    build_mc_prompt,
    build_prompt,
    match_choice_index,
    normalize_cat5_answer,
)
from benchmarks.types import LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)


class LocomoGraphitiBackend(GraphitiBackendBase):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        reset_mode: str = 'delete',
        llm_context_sizes: List[int] | None = None,
        llm_model_override: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        search_reranker: str | None = None,
    ):
        super().__init__(
            uri,
            user,
            password,
            database=database,
            usage_interval_env='LOCOMO_USAGE_INTERVAL',
            search_reranker=search_reranker,
        )
        self.search_limit = int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
        self.reset_mode = reset_mode.lower()
        if llm_context_sizes is None:
            ctx_env = os.getenv('LOCOMO_LLM_CONTEXT_K', '5,10')
            llm_context_sizes = [int(k) for k in ctx_env.split(',') if k]
        self.llm_context_sizes = sorted(set(llm_context_sizes))
        self.llm_model_override = llm_model_override or os.getenv('LOCOMO_LLM_MODEL')
        self.llm_api_key = llm_api_key or os.getenv('LOCOMO_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.llm_base_url = llm_base_url or os.getenv('LOCOMO_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
        self.answer_llm_client = self._initialize_answer_llm(
            model=self.llm_model_override,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )
        self.last_context_docs: list[str] = []

    async def ingest_sample(self, sample: LocomoSample) -> None:
        progress = tqdm(
            sample.episodes,
            desc=f'Ingesting sessions ({sample.sample_id})',
            leave=False,
        )
        for idx, episode in enumerate(progress):
            await self.graphiti.add_episode(
                name=f'{sample.sample_id}-session-{idx}',
                episode_body=episode.content,
                source_description=episode.metadata.get('session_id', 'locomo'),
                reference_time=episode.reference_time,
                source=EpisodeType.message,
            )
            self.increment_episode_count()
        progress.close()

    async def ingest_example(self, example: LocomoExample) -> None:
        progress = tqdm(
            example.episodes,
            desc=f'Ingesting sessions ({example.question_id})',
            leave=False,
        )
        for idx, episode in enumerate(progress):
            await self.graphiti.add_episode(
                name=f'{example.question_id}-session-{idx}',
                episode_body=episode.content,
                source_description=episode.metadata.get('session_id', 'locomo'),
                reference_time=episode.reference_time,
                source=EpisodeType.message,
            )
            self.increment_episode_count()
        progress.close()

    async def clear_graph(self) -> None:
        if self.reset_mode == 'delete':
            await self.graphiti.driver.execute_query('MATCH (n) DETACH DELETE n', params={})
        elif self.reset_mode == 'recreate':
            await self.graphiti.driver.close()
            self._init_graphiti()
        else:
            raise ValueError(f'Unknown reset_mode: {self.reset_mode}')

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        if self.answer_llm_client is None:
            return '', []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        episodes, edges = await self._search_context(qa.question, max_k)
        context_docs: list[str] = []
        context_ids: list[str] = []
        seen: set[str] = set()

        for episode in episodes:
            body = getattr(episode, 'episode_body', '') or getattr(episode, 'body', '') or ''
            if body:
                context_docs.append(body)
            session_id = getattr(episode, 'source_description', '') or getattr(episode, 'name', '')
            if session_id and session_id not in seen:
                seen.add(session_id)
                context_ids.append(session_id)

        if not context_docs and edges:
            for edge in edges:
                fact = getattr(edge, 'fact', '') or ''
                if fact:
                    context_docs.append(fact)

        self.last_context_docs = [doc for doc in context_docs if doc]
        context = '\n'.join(self.last_context_docs)
        prompt, answer_key = build_prompt(context, qa)
        messages = [
            Message(role='system', content='You answer using only the provided context.'),
            Message(role='user', content=prompt + '\n\nRespond with JSON {"answer": "<text>"}.' ),
        ]
        answer_text = ''
        try:
            response = await self.answer_llm_client.generate_response(
                messages,
                response_model=AnswerResponse,
            )
            answer_text = str(response.get('answer', '')).strip()
        except Exception as exc:
            logger.warning('LLM answer failed for %s: %s', sample.sample_id, exc)
        answer_text = normalize_cat5_answer(answer_text, answer_key)
        return answer_text, context_ids

    async def answer_mc_question(self, example: LocomoExample) -> tuple[int, list[str]]:
        if self.answer_llm_client is None:
            return -1, []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        episodes, edges = await self._search_context(example.question, max_k)
        context_docs: list[str] = []
        context_ids: list[str] = []
        seen: set[str] = set()

        for episode in episodes:
            body = getattr(episode, 'episode_body', '') or getattr(episode, 'body', '') or ''
            if body:
                context_docs.append(body)
            session_id = getattr(episode, 'source_description', '') or getattr(episode, 'name', '')
            if session_id and session_id not in seen:
                seen.add(session_id)
                context_ids.append(session_id)

        if not context_docs and edges:
            for edge in edges:
                fact = getattr(edge, 'fact', '') or ''
                if fact:
                    context_docs.append(fact)

        context = '\n'.join(doc for doc in context_docs if doc)
        prompt = build_mc_prompt(context, example.question, example.choices)
        messages = [
            Message(role='system', content='You answer using only the provided context.'),
            Message(role='user', content=prompt + '\n\nRespond with JSON {"answer": "<text>"}.' ),
        ]
        answer_text = ''
        try:
            response = await self.answer_llm_client.generate_response(
                messages,
                response_model=AnswerResponse,
            )
            answer_text = str(response.get('answer', '')).strip()
        except Exception as exc:
            logger.warning('LLM answer failed for %s: %s', example.question_id, exc)
        predicted_index = match_choice_index(answer_text, example.choices)
        return predicted_index, context_ids

    def _initialize_answer_llm(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        if not model:
            model = (
                os.getenv('OPENAI_MODEL')
                or os.getenv('LOCOMO_ANSWER_LLM_MODEL')
                or 'gpt-4o-mini'
            )
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        base_url = base_url or os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return None
        config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        name = (config.model or '').lower()
        if not name or name.startswith('gpt'):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)

    async def _search_context(self, query: str, num_results: int):
        config = self._resolve_search_config(num_results)
        if config is None:
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = num_results
        results = await self.graphiti.search_(query, config=config)
        episodes = getattr(results, 'episodes', []) or []
        edges = getattr(results, 'edges', []) or []
        return episodes[:num_results], edges[:num_results]
