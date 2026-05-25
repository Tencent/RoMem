"""RoMem backend for LoCoMo benchmarking."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import (
    AnswerResponse,
    initialize_answer_llm,
    parse_bool,
    parse_optional_bool,
)
from benchmarks.runners.locomo.qa_utils import (
    build_mc_prompt,
    build_prompt,
    match_choice_index,
    normalize_cat5_answer,
)
from benchmarks.runners.romem_utils import apply_romem_config
from benchmarks.types import LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)


def _safe_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "example")


class LocomoRoMemBackend:
    def __init__(
        self,
        search_limit: int | None = None,
        llm_context_sizes: List[int] | None = None,
        llm_model_override: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
    ):
        self.search_limit = search_limit or int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
        if llm_context_sizes is None:
            ctx_env = os.getenv('LOCOMO_LLM_CONTEXT_K', '5,10')
            llm_context_sizes = [int(k) for k in ctx_env.split(',') if k]
        self.llm_context_sizes = sorted(set(llm_context_sizes))
        self.llm_model_override = llm_model_override or os.getenv('LOCOMO_LLM_MODEL')
        self.llm_api_key = llm_api_key or os.getenv('LOCOMO_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.llm_base_url = llm_base_url or os.getenv('LOCOMO_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
        self.answer_llm_client = initialize_answer_llm(
            model=self.llm_model_override,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
        )

        self._romem_root = Path(
            romem_save_dir
            or os.getenv('TEMPUS_SAVE_DIR')
            or 'outputs/romem_locomo'
        )
        self._romem_llm_model = romem_llm_model or os.getenv('TEMPUS_LLM_MODEL')
        self._romem_embedding_model = romem_embedding_model or os.getenv('TEMPUS_EMBED_MODEL')
        self._romem_openie_mode = romem_openie_mode or os.getenv('TEMPUS_OPENIE_MODE')
        self._romem_config = romem_config or {}
        self._romem_temporal_awareness = parse_bool(
            romem_temporal_awareness or os.getenv('TEMPUS_TEMPORAL_AWARENESS'),
            default=True,
        )
        enable_tkge_value = romem_enable_tkge_tunnel
        if enable_tkge_value is None:
            enable_tkge_value = os.getenv('TEMPUS_ENABLE_TKGE_TUNNEL')
        self._romem_enable_tkge_tunnel = parse_optional_bool(enable_tkge_value)
        self._romem_tkge_verbose = romem_tkge_verbose

        self.romem = None
        self._doc_to_session: dict[str, str] = {}
        self._last_mc_docs: list[str] = []
        self.last_context_docs: list[str] = []

    def _init_romem(self, sample_id: str):
        from romem import RoMem
        from romem.utils.config_utils import BaseConfig

        run_dir = self._romem_root / _safe_dir_name(sample_id)
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
        config.retrieval_top_k = self.search_limit
        return RoMem(
            global_config=config,
            save_dir=config.save_dir,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
        )

    async def ingest_sample(self, sample: LocomoSample) -> None:
        self.romem = self._init_romem(sample.sample_id)
        docs = [episode.content for episode in sample.episodes]
        self._doc_to_session = {
            episode.content: str(episode.metadata.get('session_id', ''))
            for episode in sample.episodes
        }
        if docs:
            self.romem.index(docs=docs)

    async def ingest_example(self, example: LocomoExample) -> None:
        conv_id = example.question_id.split('_q', 1)[0]
        self.romem = self._init_romem(conv_id)
        docs = [episode.content for episode in example.episodes]
        self._doc_to_session = {
            episode.content: str(episode.metadata.get('session_id', ''))
            for episode in example.episodes
        }
        if docs:
            self.romem.index(docs=docs)

    async def clear_graph(self) -> None:
        self.romem = None

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        if self.romem is None:
            return '', []
        if not self.answer_llm_client:
            return '', []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        try:
            retrieval = self.romem.retrieve([qa.question], num_to_retrieve=max_k)
            docs = retrieval[0].docs if retrieval else []
        except Exception as exc:
            logger.warning('RoMem context retrieval failed for %s: %s', sample.sample_id, exc)
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
        if self.romem is None or self.answer_llm_client is None:
            return -1, []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        try:
            retrieval = self.romem.retrieve([example.question], num_to_retrieve=max_k)
            docs = retrieval[0].docs if retrieval else []
        except Exception as exc:
            logger.warning('RoMem context retrieval failed for %s: %s', example.question_id, exc)
            docs = []
        self._last_mc_docs = docs
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

