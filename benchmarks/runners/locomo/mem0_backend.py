"""Mem0 backend for LoCoMo benchmarking using local Mem0 (FAISS + optional graph)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List
import importlib.metadata as importlib_metadata

from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from benchmarks.runners.shared_utils import AnswerResponse
from benchmarks.runners.base_backend import Mem0BackendBase
from benchmarks.runners.locomo.qa_utils import (
    build_mc_prompt,
    build_prompt,
    match_choice_index,
    normalize_cat5_answer,
)
from benchmarks.types import LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)


class LocomoMem0Backend(Mem0BackendBase):
    def __init__(
        self,
        neo4j_uri: str | None,
        neo4j_user: str | None,
        neo4j_password: str | None,
        neo4j_database: str | None = None,
        search_limit: int | None = None,
        llm_context_sizes: List[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
    ):
        self.search_limit = search_limit or int(os.getenv('LOCOMO_SEARCH_LIMIT', '50'))
        if llm_context_sizes is None:
            ctx_env = os.getenv('LOCOMO_LLM_CONTEXT_K', '5,10')
            llm_context_sizes = [int(k) for k in ctx_env.split(',') if k]
        self.llm_context_sizes = sorted(set(llm_context_sizes))
        self.answer_llm_model = answer_llm_model or os.getenv('LOCOMO_ANSWER_LLM_MODEL') or os.getenv('OPENAI_MODEL')
        self.answer_llm_api_key = answer_llm_api_key or os.getenv('LOCOMO_ANSWER_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.answer_llm_base_url = answer_llm_base_url or os.getenv('LOCOMO_ANSWER_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')

        mem0_root = Path(__file__).resolve().parents[3] / 'baselines' / 'mem0'
        mem0_path = str(mem0_root)
        if mem0_path not in sys.path:
            sys.path.insert(0, mem0_path)
        _orig_version = importlib_metadata.version
        def _safe_version(name: str):
            if name == 'mem0ai':
                return '0.0.0'
            return _orig_version(name)
        importlib_metadata.version = _safe_version  # type: ignore
        try:
            from mem0.configs.base import MemoryConfig
            from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig
            from mem0.embeddings.configs import EmbedderConfig
            from mem0.configs.vector_stores.faiss import FAISSConfig
        finally:
            importlib_metadata.version = _orig_version  # type: ignore
        mem_user = os.getenv('MEM0_USER_ID', 'locomo')
        mem0_dir = os.getenv('MEM0_DIR', str(Path('outputs/mem0_locomo').resolve()))
        os.environ['MEM0_DIR'] = mem0_dir
        neo4j_cfg = Neo4jConfig(
            url=neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=neo4j_user or os.getenv('NEO4J_USER', 'neo4j'),
            password=neo4j_password or os.getenv('NEO4J_PASSWORD', ''),
            database=neo4j_database or os.getenv('LOCOMO_NEO4J_DATABASE', None),
            base_label=True,
        )
        graph_store = GraphStoreConfig(provider='neo4j', config=neo4j_cfg)
        mem0_embed_model = os.getenv('MEM0_EMBED_MODEL') or 'text-embedding-3-small'
        mem0_embed_dims = int(os.getenv('MEM0_EMBED_DIMS', '1536'))
        embed_provider = os.getenv('MEM0_EMBED_PROVIDER', 'openai')
        embed_config = {
            'model': mem0_embed_model,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'embedding_dims': mem0_embed_dims,
            'openai_base_url': os.getenv('OPENAI_BASE_URL'),
        }
        if embed_provider.lower() in {'huggingface', 'hf', 'sentence_transformers'}:
            embed_config.setdefault('model_kwargs', {})['trust_remote_code'] = True
        config = MemoryConfig(
            graph_store=graph_store,
            embedder=EmbedderConfig(
                provider=embed_provider,
                config=embed_config,
            ),
        )
        config.vector_store.provider = 'faiss'
        config.vector_store.config = FAISSConfig(
            path=os.getenv('MEM0_VECTOR_PATH', str(Path(mem0_dir) / 'faiss_run')),
            collection_name=mem_user,
            embedding_model_dims=mem0_embed_dims,
            distance_strategy='cosine',
            normalize_L2=True,
        )
        mem0_llm_model = self.answer_llm_model or os.getenv('OPENAI_MODEL')
        mem0_llm_api_key = self.answer_llm_api_key or os.getenv('OPENAI_API_KEY')
        mem0_llm_base_url = self.answer_llm_base_url or os.getenv('OPENAI_BASE_URL')
        config.llm.provider = 'openai'
        config.llm.config = {
            'model': mem0_llm_model,
            'api_key': mem0_llm_api_key,
            'openai_base_url': mem0_llm_base_url,
        }
        self._neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self._neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', '')
        self._neo4j_database = neo4j_database or os.getenv('LOCOMO_NEO4J_DATABASE', None) or 'neo4j'
        super().__init__(mem0_config=config, mem0_user_id=mem_user, usage_interval_env='LOCOMO_USAGE_INTERVAL', mem0_dir=mem0_dir)
        self._maybe_wipe_all_stores()
        self._log_graph_status()
        self.last_context_docs: list[str] = []
        self.answer_llm_client = self._initialize_answer_llm(
            model=self.answer_llm_model,
            api_key=self.answer_llm_api_key,
            base_url=self.answer_llm_base_url,
        )

    def _user_id(self, sample: LocomoSample) -> str:
        return f'{self.mem_user_id}_{sample.sample_id}'

    def _user_id_mc(self, example: LocomoExample) -> str:
        conv_id = example.question_id.split('_q', 1)[0]
        return f'{self.mem_user_id}_{conv_id}'

    def _add_memory(self, user_id: str, message: str, metadata: dict, retries: int = 3):
        for attempt in range(retries):
            try:
                return self.memory.add(
                    messages=message,
                    user_id=user_id,
                    metadata=metadata,
                    infer=False,
                )
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning('Mem0 add retry %s for %s: %s', attempt + 1, user_id, exc)

    async def ingest_sample(self, sample: LocomoSample) -> None:
        user_id = self._user_id(sample)
        self._wipe_all_stores()

        for episode in sample.episodes:
            try:
                self._add_memory(
                    user_id=user_id,
                    message=episode.content,
                    metadata={'session_id': episode.metadata.get('session_id')},
                )
                self.increment_episode_count()
            except Exception as exc:
                logger.warning('Mem0 add failed: %s', exc)
        logger.info('Mem0 added %s episodes for user %s', len(sample.episodes), user_id)
        self._log_graph_counts(user_id)

    async def ingest_example(self, example: LocomoExample) -> None:
        user_id = self._user_id_mc(example)
        self._wipe_all_stores()
        for episode in example.episodes:
            try:
                self._add_memory(
                    user_id=user_id,
                    message=episode.content,
                    metadata={'session_id': episode.metadata.get('session_id')},
                )
                self.increment_episode_count()
            except Exception as exc:
                logger.warning('Mem0 add failed: %s', exc)
        logger.info('Mem0 added %s episodes for user %s', len(example.episodes), user_id)
        self._log_graph_counts(user_id)

    async def clear_graph(self) -> None:
        pass

    async def answer_question(self, sample: LocomoSample, qa: LocomoQA) -> tuple[str, list[str]]:
        if self.answer_llm_client is None:
            return '', []
        user_id = self._user_id(sample)
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs, context_ids = self._search_context(user_id, qa.question, max_k)
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
        return answer_text, context_ids

    async def answer_mc_question(self, example: LocomoExample) -> tuple[int, list[str]]:
        if self.answer_llm_client is None:
            return -1, []
        user_id = self._user_id_mc(example)
        max_k = max(self.llm_context_sizes) if self.llm_context_sizes else self.search_limit
        docs, context_ids = self._search_context(user_id, example.question, max_k)
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
        return predicted_index, context_ids

    def _search_context(self, user_id: str, query: str, limit: int) -> tuple[list[str], list[str]]:
        try:
            retrieval = self.memory.search(
                query,
                user_id=user_id,
                limit=limit,
            )
        except Exception as exc:
            logger.warning('Mem0 search failed for %s: %s', query, exc)
            return [], []

        docs: list[str] = []
        context_ids: list[str] = []
        seen: set[str] = set()
        if isinstance(retrieval, dict) and 'results' in retrieval:
            for res in retrieval.get('results', []):
                if not res:
                    continue
                docs.append(res.get('memory') or res.get('text') or '')
                metadata = res.get('metadata') or {}
                session_id = metadata.get('session_id')
                if session_id and session_id not in seen:
                    seen.add(session_id)
                    context_ids.append(session_id)
        elif isinstance(retrieval, list):
            for res in retrieval:
                if isinstance(res, dict):
                    docs.append(res.get('memory', '') or res.get('text', ''))
                    metadata = res.get('metadata') or {}
                    session_id = metadata.get('session_id')
                    if session_id and session_id not in seen:
                        seen.add(session_id)
                        context_ids.append(session_id)
                else:
                    docs.append(str(res))
        if not docs:
            logger.warning('Mem0 retrieval returned empty docs; query=%s user=%s', query, user_id)
        return docs, context_ids

    def _log_graph_status(self) -> None:
        try:
            logger.info(
                'Mem0 graph enabled=%s neo4j=%s',
                getattr(self.memory, 'enable_graph', False),
                getattr(self.memory.graph.config, 'url', None) if getattr(self.memory, 'graph', None) else None,
            )
        except Exception:
            pass

    def _log_graph_counts(self, user_id: str) -> None:
        try:
            graph = getattr(self.memory, 'graph', None)
            if not graph:
                return
            res = graph.query(
                """
                MATCH (n) WHERE n.user_id = $uid RETURN count(n) AS nodes
                """,
                params={'uid': user_id},
            )
            nodes = res[0].get('nodes', 0) if res else 0
            res_rel = graph.query(
                """
                MATCH ()-[r]-() WHERE r.user_id = $uid RETURN count(r) AS rels
                """,
                params={'uid': user_id},
            )
            rels = res_rel[0].get('rels', 0) if res_rel else 0
            logger.info('Mem0 graph counts for %s: nodes=%s rels=%s', user_id, nodes, rels)
        except Exception as exc:
            logger.warning('Mem0 graph count query failed: %s', exc)

    def _maybe_wipe_all_stores(self) -> None:
        if os.getenv('MEM0_CLEAR_ALL', '0') not in {'1', 'true', 'True'}:
            return
        self._wipe_all_stores()

    def _wipe_all_stores(self) -> None:
        try:
            if getattr(self.memory, 'vector_store', None):
                self.memory.vector_store.reset()
            self._clear_default_database()
            logger.info('Mem0 stores wiped (vector + graph) user=ALL')
        except Exception as exc:
            logger.warning('Mem0 store wipe failed: %s', exc)

    def _clear_default_database(self) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as exc:
            logger.warning('Neo4j driver import failed: %s', exc)
            return
        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        try:
            with driver.session(database=self._neo4j_database) as session:
                total_deleted = 0
                while True:
                    query = """
                    CALL {
                        MATCH (n)
                        WITH n LIMIT $batch_size
                        DETACH DELETE n
                        RETURN COUNT(*) AS deleted
                    }
                    RETURN deleted
                    """
                    result = session.run(query, batch_size=1000).single()
                    deleted = result['deleted'] if result else 0
                    total_deleted += deleted
                    if deleted == 0:
                        break
                logger.info('Cleared %s nodes from Neo4j (db=%s)', total_deleted, self._neo4j_database)
        finally:
            driver.close()

    def _initialize_answer_llm(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        if not model:
            model = os.getenv('OPENAI_MODEL') or 'gpt-4o-mini'
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        base_url = base_url or os.getenv('OPENAI_BASE_URL')
        if not api_key:
            return None
        config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        name = (config.model or '').lower()
        if not name or name.startswith('gpt'):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)
