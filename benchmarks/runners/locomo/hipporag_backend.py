"""HippoRAG backend for LoCoMo benchmarking."""

from __future__ import annotations

import logging
import os
from typing import List

from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import AnswerResponse
from benchmarks.runners.base_backend import HippoRAGBackendBase
from benchmarks.runners.locomo.qa_utils import (
    build_mc_prompt,
    build_prompt,
    match_choice_index,
    normalize_cat5_answer,
)
from benchmarks.types import LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)


class LocomoHippoRAGBackend(HippoRAGBackendBase):
    def __init__(
        self,
        search_limit: int | None = None,
        llm_context_sizes: List[int] | None = None,
        llm_model_override: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
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
        self.search_limit = search_limit or int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
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
        self._doc_to_session: dict[str, str] = {}
        self.last_context_docs: list[str] = []

    async def ingest_sample(self, sample: LocomoSample) -> None:
        # Reset HippoRAG state per sample
        self._reset_hipporag(sample_id=sample.sample_id)
        docs = [episode.content for episode in sample.episodes]
        self._doc_to_session = {
            episode.content: str(episode.metadata.get('session_id', ''))
            for episode in sample.episodes
        }
        if docs:
            self.hipporag.index(docs=docs)
            self.increment_episode_count(len(docs))

    async def ingest_example(self, example: LocomoExample) -> None:
        conv_id = example.question_id.split('_q', 1)[0]
        self._reset_hipporag(sample_id=conv_id)
        docs = [episode.content for episode in example.episodes]
        self._doc_to_session = {
            episode.content: str(episode.metadata.get('session_id', ''))
            for episode in example.episodes
        }
        if docs:
            self.hipporag.index(docs=docs)
            self.increment_episode_count(len(docs))

    async def clear_graph(self) -> None:
        self._reset_hipporag()

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        if self.answer_llm_client is None:
            return '', []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        try:
            retrieval = self.hipporag.retrieve([qa.question], num_to_retrieve=max_k)
            docs = retrieval[0].docs if retrieval else []
        except Exception as exc:
            logger.warning('HippoRAG context retrieval failed for %s: %s', sample.sample_id, exc)
            docs = []

        self.last_context_docs = [doc for doc in docs if doc]
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

        context_ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            session_id = self._doc_to_session.get(doc)
            if session_id and session_id not in seen:
                seen.add(session_id)
                context_ids.append(session_id)
        return answer_text, context_ids

    async def answer_mc_question(self, example: LocomoExample) -> tuple[int, list[str]]:
        if self.answer_llm_client is None:
            return -1, []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        try:
            retrieval = self.hipporag.retrieve([example.question], num_to_retrieve=max_k)
            docs = retrieval[0].docs if retrieval else []
        except Exception as exc:
            logger.warning('HippoRAG context retrieval failed for %s: %s', example.question_id, exc)
            docs = []

        context = '\n'.join(doc for doc in docs if doc)
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

        context_ids: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            session_id = self._doc_to_session.get(doc)
            if session_id and session_id not in seen:
                seen.add(session_id)
                context_ids.append(session_id)
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
