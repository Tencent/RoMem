"""A-mem-sys backend for LoCoMo benchmarking."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from openai import AsyncOpenAI

from benchmarks.runners.locomo.qa_utils import (
    build_mc_prompt,
    build_prompt,
    match_choice_index,
    normalize_cat5_answer,
)
from benchmarks.types import LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)

# Ensure baselines/A-mem-sys is on sys.path so we can import agentic_memory
_AMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'A-mem-sys')
if _AMEM_ROOT not in sys.path:
    sys.path.insert(0, _AMEM_ROOT)


class LocomoAMemBackend:
    """LoCoMo backend that uses A-mem-sys (ChromaDB + sentence-transformers)."""

    def __init__(
        self,
        model_name: str | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        search_limit: int | None = None,
        llm_context_sizes: List[int] | None = None,
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
        self.amem_llm_backend = llm_backend or os.getenv('AMEM_LLM_BACKEND') or 'openai'
        self.amem_llm_base_url = os.getenv('AMEM_LLM_BASE_URL') or None
        self.amem_llm_model = (
            llm_model
            or os.getenv('AMEM_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        self.search_limit = search_limit or int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
        if llm_context_sizes is None:
            ctx_env = os.getenv('LOCOMO_LLM_CONTEXT_K', '5,10')
            llm_context_sizes = [int(k) for k in ctx_env.split(',') if k.strip().isdigit()]
        self.llm_context_sizes = sorted(set(llm_context_sizes))

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('LOCOMO_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        answer_api_key = (
            answer_llm_api_key
            or os.getenv('LOCOMO_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        answer_base_url = (
            answer_llm_base_url
            or os.getenv('LOCOMO_ANSWER_LLM_BASE_URL')
            or os.getenv('OPENAI_BASE_URL')
            or None
        )
        self.answer_llm_client: AsyncOpenAI | None = None
        if answer_api_key:
            self.answer_llm_client = AsyncOpenAI(
                api_key=answer_api_key,
                base_url=answer_base_url,
            )

        self._base_dir = os.getenv('AMEM_BASE_DIR') or str(Path('outputs/amem_locomo').resolve())
        self._memory = None
        self.last_context_docs: list[str] = []
        self._episodes_processed = 0
        self._note_to_session: dict[str, str] = {}  # note_id -> session_id

    def _cache_dir(self, sample_id: str) -> str:
        return str(Path(self._base_dir) / f'sample_{sample_id}')

    def _create_memory(self, persist_directory: str | None = None):
        """Create an AgenticMemorySystem instance, optionally with persistence."""
        from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore
        return AgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.amem_llm_backend,
            llm_model=self.amem_llm_model,
            embedding_provider=self.embedding_provider,
            persist_directory=persist_directory,
            llm_base_url=self.amem_llm_base_url,
        )

    def _save_cache(self, cache_dir: str) -> None:
        """Save memories dict and note-to-session mapping to cache."""
        import pickle
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, 'memories.pkl'), 'wb') as f:
            pickle.dump(self._memory.memories, f)
        with open(os.path.join(cache_dir, 'note_to_session.pkl'), 'wb') as f:
            pickle.dump(self._note_to_session, f)

    def _load_cache(self, cache_dir: str) -> bool:
        """Load memories and mapping from cache. Returns True if cache exists."""
        import pickle
        mem_path = os.path.join(cache_dir, 'memories.pkl')
        map_path = os.path.join(cache_dir, 'note_to_session.pkl')
        if not os.path.isfile(mem_path) or not os.path.isfile(map_path):
            return False
        with open(mem_path, 'rb') as f:
            self._memory.memories = pickle.load(f)
        with open(map_path, 'rb') as f:
            self._note_to_session = pickle.load(f)
        return True

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    async def clear_graph(self) -> None:
        """Release current memory (cache is kept on disk)."""
        self._memory = None

    async def ingest_sample(self, sample: LocomoSample) -> None:
        """Ingest episodes, loading from cache if available."""
        cache_dir = self._cache_dir(sample.sample_id)
        self._note_to_session.clear()

        # Try loading from cache
        self._memory = self._create_memory(persist_directory=cache_dir)
        if self._load_cache(cache_dir):
            print(f'  Loaded cached A-mem for sample {sample.sample_id}', flush=True)
            return

        # No cache — ingest from scratch
        n_eps = len(sample.episodes)
        print(f'  Ingesting {n_eps} episodes for sample {sample.sample_id}...', flush=True)
        for i, episode in enumerate(sample.episodes, 1):
            session_id = episode.metadata.get('session_id', '')
            try:
                note_id = self._memory.add_note(episode.content)
                self._note_to_session[note_id] = session_id
                self.increment_episode_count()
                print(f'    Episode {i}/{n_eps} done ({session_id})', flush=True)
            except Exception as exc:
                print(f'    Episode {i}/{n_eps} FAILED ({session_id}): {exc}', flush=True)
        self._save_cache(cache_dir)
        print(f'  Ingested {n_eps} episodes for sample {sample.sample_id}', flush=True)

    async def ingest_example(self, example: LocomoExample) -> None:
        """Ingest MC example episodes, loading from cache if available."""
        conv_id = example.question_id.split('_q', 1)[0]
        cache_dir = self._cache_dir(conv_id)
        self._note_to_session.clear()

        self._memory = self._create_memory(persist_directory=cache_dir)
        if self._load_cache(cache_dir):
            print(f'  Loaded cached A-mem for MC example {conv_id}', flush=True)
            return

        n_eps = len(example.episodes)
        print(f'  Ingesting {n_eps} episodes for MC example {conv_id}...', flush=True)
        for i, episode in enumerate(example.episodes, 1):
            session_id = episode.metadata.get('session_id', '')
            try:
                note_id = self._memory.add_note(episode.content)
                self._note_to_session[note_id] = session_id
                self.increment_episode_count()
                print(f'    Episode {i}/{n_eps} done ({session_id})', flush=True)
            except Exception as exc:
                print(f'    Episode {i}/{n_eps} FAILED ({session_id}): {exc}', flush=True)
        self._save_cache(cache_dir)
        print(f'  Ingested {n_eps} episodes for MC example {conv_id}', flush=True)

    def _search_context(self, query: str, limit: int) -> tuple[list[str], list[str]]:
        """Search memory and return (docs, context_ids).

        context_ids are session IDs (e.g. S1, S2) mapped from note IDs,
        so they're compatible with LoCoMo's recall_from_context().
        """
        if self._memory is None:
            return [], []
        try:
            results = self._memory.search(query, k=limit)
        except Exception as exc:
            logger.warning('A-mem search failed for query "%s": %s', query, exc)
            return [], []

        docs: list[str] = []
        context_ids: list[str] = []
        seen: set[str] = set()
        for item in results:
            content = item.get('content', '') or ''
            docs.append(content)
            # Map note_id back to session_id for recall evaluation
            note_id = item.get('id', '')
            session_id = self._note_to_session.get(note_id, '')
            if session_id and session_id not in seen:
                seen.add(session_id)
                context_ids.append(session_id)
        if not docs:
            logger.warning('A-mem retrieval returned empty docs for query: %s', query)
        return docs, context_ids

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        """Search memory and generate answer for a QA pair."""
        if self.answer_llm_client is None:
            return '', []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs, context_ids = self._search_context(qa.question, max_k)
        self.last_context_docs = [doc for doc in docs if doc]
        context = '\n'.join(self.last_context_docs)
        prompt, answer_key = build_prompt(context, qa)
        messages = [
            {'role': 'system', 'content': 'You answer using only the provided context.'},
            {'role': 'user', 'content': prompt + '\n\nRespond with JSON {"answer": "<text>"}.'},
        ]
        answer_text = ''
        try:
            response = await self.answer_llm_client.chat.completions.create(
                model=self.answer_llm_model,
                messages=messages,
            )
            raw = response.choices[0].message.content or ''
            try:
                parsed = json.loads(raw)
                answer_text = str(parsed.get('answer', '')).strip()
            except json.JSONDecodeError:
                answer_text = raw.strip()
        except Exception as exc:
            logger.warning('A-mem LLM answer failed for sample %s: %s', sample.sample_id, exc)
        answer_text = normalize_cat5_answer(answer_text, answer_key)
        return answer_text, context_ids

    async def answer_mc_question(self, example: LocomoExample) -> tuple[int, list[str]]:
        """Search memory and answer a multiple-choice question."""
        if self.answer_llm_client is None:
            return -1, []
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs, context_ids = self._search_context(example.question, max_k)
        context = '\n'.join(doc for doc in docs if doc)
        prompt = build_mc_prompt(context, example.question, example.choices)
        messages = [
            {'role': 'system', 'content': 'You answer using only the provided context.'},
            {'role': 'user', 'content': prompt + '\n\nRespond with JSON {"answer": "<text>"}.'},
        ]
        answer_text = ''
        try:
            response = await self.answer_llm_client.chat.completions.create(
                model=self.answer_llm_model,
                messages=messages,
            )
            raw = response.choices[0].message.content or ''
            try:
                parsed = json.loads(raw)
                answer_text = str(parsed.get('answer', '')).strip()
            except json.JSONDecodeError:
                answer_text = raw.strip()
        except Exception as exc:
            logger.warning('A-mem LLM MC answer failed for %s: %s', example.question_id, exc)
        predicted_index = match_choice_index(answer_text, example.choices)
        return predicted_index, context_ids
