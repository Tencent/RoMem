"""LiCoMemory backend for LoCoMo benchmarking using the Cognigraph system."""

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

# Add baselines/LiCoMemory to sys.path so LiCoMemory modules can be imported
_LICOMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'LiCoMemory')
if _LICOMEM_ROOT not in sys.path:
    sys.path.insert(0, _LICOMEM_ROOT)


class LocomoLiCoMemBackend:
    """LoCoMo backend that uses LiCoMemory (Cognigraph) for memory storage and retrieval."""

    def __init__(
        self,
        search_limit: int | None = None,
        llm_context_sizes: List[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        # LiCoMemory LLM config (for graph building / entity extraction)
        licomem_llm_model: str | None = None,
        licomem_llm_api_key: str | None = None,
        licomem_llm_base_url: str | None = None,
        # LiCoMemory embedding config
        licomem_embed_model: str | None = None,
        licomem_embed_api_key: str | None = None,
        licomem_embed_api_type: str | None = None,
        licomem_embed_dimensions: int | None = None,
        # Results dir for LiCoMemory graph storage
        licomem_base_dir: str | None = None,
    ):
        self.search_limit = search_limit or int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
        if llm_context_sizes is None:
            ctx_env = os.getenv('LOCOMO_LLM_CONTEXT_K', '5,10')
            llm_context_sizes = [int(k) for k in ctx_env.split(',') if k]
        self.llm_context_sizes = sorted(set(llm_context_sizes))

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('LOCOMO_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-5-mini'
        )
        llm_api_key = (
            answer_llm_api_key
            or os.getenv('LOCOMO_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        llm_base_url = (
            answer_llm_base_url
            or os.getenv('LOCOMO_ANSWER_LLM_BASE_URL')
            or os.getenv('OPENAI_BASE_URL')
            or None
        )
        self.answer_llm_client: AsyncOpenAI | None = None
        if llm_api_key:
            self.answer_llm_client = AsyncOpenAI(
                api_key=llm_api_key,
                base_url=llm_base_url,
            )

        # LiCoMemory LLM settings (for graph building / entity extraction)
        self._licomem_llm_model = (
            licomem_llm_model
            or os.getenv('LICOMEM_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-5-mini'
        )
        self._licomem_llm_api_key = (
            licomem_llm_api_key
            or os.getenv('LICOMEM_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
            or ''
        )
        self._licomem_llm_base_url = (
            licomem_llm_base_url
            or os.getenv('LICOMEM_LLM_BASE_URL')
            or os.getenv('OPENAI_BASE_URL')
            or 'https://api.openai.com/v1'
        )

        # LiCoMemory embedding settings
        self._licomem_embed_api_type = (
            licomem_embed_api_type
            or os.getenv('LICOMEM_EMBED_API_TYPE')
            or 'openai'
        )
        self._licomem_embed_model = (
            licomem_embed_model
            or os.getenv('LICOMEM_EMBED_MODEL')
            or 'text-embedding-3-small'
        )
        self._licomem_embed_api_key = (
            licomem_embed_api_key
            or os.getenv('LICOMEM_EMBED_API_KEY')
            or os.getenv('OPENAI_API_KEY')
            or ''
        )
        self._licomem_embed_dimensions = (
            licomem_embed_dimensions
            or int(os.getenv('LICOMEM_EMBED_DIMENSIONS', '1536'))
        )
        self._licomem_base_dir = (
            licomem_base_dir
            or os.getenv('LICOMEM_BASE_DIR')
            or str(Path('outputs/licomem_locomo').resolve())
        )

        self._episodes_processed = 0
        self.last_context_docs: list[str] = []

    def _build_config(self, working_dir: str):
        """Build LiCoMemory Config programmatically (no YAML file needed)."""
        from init.config import (
            Config, LLMConfig, QueryLLMConfig, EmbeddingConfig,
            ChunkConfig, GraphConfig, RetrieverConfig, QueryConfig,
            StorageConfig, EvaluationConfig,
        )
        config = Config()
        config.index_name = 'locomo_graph'
        config.data_type = 'LOCOMO'
        config.working_dir = working_dir

        # LLM config (for graph building / entity extraction)
        config.llm = LLMConfig(
            api_type='openai',
            api_key=self._licomem_llm_api_key,
            base_url=self._licomem_llm_base_url,
            model=self._licomem_llm_model,
            max_token=4096,
            temperature=0.0,
            enable_concurrent=True,
            max_concurrent=8,
            timeout=300,
        )

        # Query LLM config (inherit from main LLM unless overridden)
        config.query_llm = QueryLLMConfig(
            api_type='openai',
            api_key='',
            base_url='',
            model='',
            max_token=0,
            temperature=-1.0,
            timeout=0,
        )

        # Embedding config
        config.embedding = EmbeddingConfig(
            api_type=self._licomem_embed_api_type,
            api_key=self._licomem_embed_api_key,
            model=self._licomem_embed_model,
            cache_dir=str(Path(self._licomem_base_dir) / 'embed_cache'),
            dimensions=self._licomem_embed_dimensions,
            max_token_size=8102,
            embed_batch_size=32,
            embedding_func_max_async=8,
        )

        # Chunk config
        config.chunk = ChunkConfig(
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            token_model=self._licomem_llm_model,
            dialogue_input=True,
        )

        # Graph config
        config.graph = GraphConfig(
            graph_type='dynamic_memory',
            force=True,
            add=False,
            entity_merge_threshold=0.85,
            relationship_merge_threshold=0.9,
        )

        # Retriever config
        config.retriever = RetrieverConfig(
            top_k=self.search_limit,
            top_k_triples=self.search_limit,
            top_chunks=15,
            enable_summary=False,
            top_summary=1,
            enable_visual=False,
            enable_full=True,
            enable_sessiontime=True,
            enable_CogniRank=False,
        )

        # Query config
        config.query = QueryConfig(
            query_type='qa',
            only_need_context=False,
            enable_hybrid_query=True,
        )

        # Storage config
        config.storage = StorageConfig(
            storage_type='networkx',
            persist_format='pickle',
            enable_backup=False,
        )

        # Evaluation config
        config.evaluation = EvaluationConfig(
            enable_llm_eval=False,
        )

        return config

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    def _episodes_to_corpus(self, episodes) -> list[dict]:
        """Convert EpisodePayload list to LiCoMemory corpus format.

        Wraps plain-text episodes in "Narrator": "..." format so that
        LiCoMemory's DialogChunkProcessor can parse them as dialogue turns.
        """
        corpus = []
        for idx, episode in enumerate(episodes):
            session_time = ''
            if episode.reference_time:
                try:
                    session_time = episode.reference_time.strftime('%Y-%m-%d')
                except Exception:
                    session_time = str(episode.reference_time)
            # Wrap plain text as dialogue so DialogChunkProcessor can parse it
            escaped = episode.content.replace('"', '\\"')
            context = f'"Narrator": "{escaped}"'
            corpus.append({
                'session_id': episode.metadata.get('session_id', f'session_{idx}'),
                'context': context,
                'content': episode.content,
                'session_time': session_time,
            })
        return corpus

    async def ingest_sample(self, sample: LocomoSample) -> object:
        """Create a new GraphRAG instance, loading from cache if available."""
        from init.graph_rag import GraphRAG

        working_dir = str(
            Path(self._licomem_base_dir) / f'sample_{sample.sample_id}'
        )
        os.makedirs(working_dir, exist_ok=True)

        # Check for cached graph
        cached_pkl = os.path.join(working_dir, 'locomo_graph.pkl')
        use_cache = os.path.isfile(cached_pkl)

        config = self._build_config(working_dir)
        if use_cache:
            config.graph.force = False
            config.graph.add = False
        graph_rag = GraphRAG(config, base_dir=working_dir)

        if use_cache:
            logger.info('Loading cached graph for sample %s', sample.sample_id)
        else:
            corpus = self._episodes_to_corpus(sample.episodes)
            try:
                await graph_rag.insert(corpus)
                self.increment_episode_count(len(corpus))
                logger.info(
                    'LiCoMemory inserted %s episodes for sample %s',
                    len(corpus), sample.sample_id,
                )
            except Exception as exc:
                logger.warning(
                    'LiCoMemory insert failed for sample %s: %s',
                    sample.sample_id, exc,
                )

        self._current_graph_rag = graph_rag
        self._current_sample_id = sample.sample_id
        return graph_rag

    async def ingest_example(self, example: LocomoExample) -> object:
        """Create a new GraphRAG instance, loading from cache if available."""
        from init.graph_rag import GraphRAG

        conv_id = example.question_id.split('_q', 1)[0]
        working_dir = str(
            Path(self._licomem_base_dir) / f'example_{conv_id}'
        )
        os.makedirs(working_dir, exist_ok=True)

        cached_pkl = os.path.join(working_dir, 'locomo_graph.pkl')
        use_cache = os.path.isfile(cached_pkl)

        config = self._build_config(working_dir)
        if use_cache:
            config.graph.force = False
            config.graph.add = False
        graph_rag = GraphRAG(config, base_dir=working_dir)

        if use_cache:
            logger.info('Loading cached graph for MC example %s', conv_id)
        else:
            corpus = self._episodes_to_corpus(example.episodes)
            try:
                await graph_rag.insert(corpus)
                self.increment_episode_count(len(corpus))
                logger.info(
                    'LiCoMemory inserted %s episodes for MC example %s',
                    len(corpus), example.question_id,
                )
            except Exception as exc:
                logger.warning(
                    'LiCoMemory insert failed for MC example %s: %s',
                    example.question_id, exc,
                )

        self._current_graph_rag = graph_rag
        self._current_conv_id = conv_id
        return graph_rag

    async def clear_graph(self) -> None:
        """Release current graph state (keeps cached pkl for reuse)."""
        self._current_graph_rag = None
        self._current_sample_id = None
        self._current_conv_id = None

        self._current_graph_rag = None
        self.last_context_docs = []

    def _extract_context_from_result(self, result: dict | str) -> list[str]:
        """Extract text context from a LiCoMemory query result dict."""
        docs: list[str] = []
        if isinstance(result, str):
            if result.strip():
                docs.append(result)
            return docs

        if not isinstance(result, dict):
            return docs

        # Prefer triples (src, relation, tgt) as they are the most informative
        triples = result.get('triples', [])
        for triple in triples:
            if isinstance(triple, dict):
                src = triple.get('src', '')
                rel = triple.get('relation', '')
                tgt = triple.get('tgt', '')
                fact = f'{src} {rel} {tgt}'.strip()
                if fact:
                    docs.append(fact)

        # Also add chunks for richer context
        chunks = result.get('chunks', [])
        for chunk in chunks:
            if isinstance(chunk, str) and chunk.strip():
                docs.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get('content') or chunk.get('text') or ''
                if text.strip():
                    docs.append(text)

        return docs

    def _extract_session_ids_from_result(self, result: dict | str, max_ids: int = 10) -> list[str]:
        """Extract top-K session IDs from a LiCoMemory query result dict.

        Only returns up to *max_ids* unique session IDs (default 10) to keep
        recall evaluation consistent with other baselines that cap at search_limit.
        Session IDs are ordered by relevance (triples first, then chunks).
        """
        if not isinstance(result, dict):
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        # Extract from triples (most relevant — ranked by retrieval)
        for triple in result.get('triples', []):
            if len(ordered) >= max_ids:
                break
            sid = triple.get('session_id', '') if isinstance(triple, dict) else ''
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        # Extract from chunks (fallback)
        for chunk in result.get('chunks', []):
            if len(ordered) >= max_ids:
                break
            sid = chunk.get('session_id', '') if isinstance(chunk, dict) else ''
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        return ordered

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        if self.answer_llm_client is None:
            return '', []

        graph_rag = getattr(self, '_current_graph_rag', None)
        if graph_rag is None:
            logger.warning('No current graph_rag for sample %s', sample.sample_id)
            return '', []

        result = {}
        try:
            result = await graph_rag.query(qa.question, question_time='2024/01/01 (Mon) 12:00')
        except Exception as exc:
            logger.warning(
                'LiCoMemory query failed for sample %s question: %s',
                sample.sample_id, exc,
            )

        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs = self._extract_context_from_result(result)[:max_k]
        context_ids = self._extract_session_ids_from_result(result)
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
                answer_text = str(parsed.get('answer', raw)).strip()
            except json.JSONDecodeError:
                answer_text = raw.strip()
        except Exception as exc:
            logger.warning('LLM answer failed for %s: %s', sample.sample_id, exc)

        answer_text = normalize_cat5_answer(answer_text, answer_key)
        return answer_text, context_ids

    async def answer_mc_question(self, example: LocomoExample) -> tuple[int, list[str]]:
        if self.answer_llm_client is None:
            return -1, []

        graph_rag = getattr(self, '_current_graph_rag', None)
        if graph_rag is None:
            logger.warning('No current graph_rag for MC example %s', example.question_id)
            return -1, []

        result = {}
        try:
            result = await graph_rag.query(example.question, question_time='2024/01/01 (Mon) 12:00')
        except Exception as exc:
            logger.warning(
                'LiCoMemory query failed for MC example %s: %s',
                example.question_id, exc,
            )

        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs = self._extract_context_from_result(result)[:max_k]
        context_ids = self._extract_session_ids_from_result(result)
        self.last_context_docs = [doc for doc in docs if doc]

        context = '\n'.join(self.last_context_docs)
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
                answer_text = str(parsed.get('answer', raw)).strip()
            except json.JSONDecodeError:
                answer_text = raw.strip()
        except Exception as exc:
            logger.warning('LLM MC answer failed for %s: %s', example.question_id, exc)

        predicted_index = match_choice_index(answer_text, example.choices)
        return predicted_index, context_ids
