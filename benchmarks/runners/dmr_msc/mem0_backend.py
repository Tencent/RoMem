"""Mem0 backend implementation for the DMR-MSC benchmark (local Mem0: FAISS + Neo4j)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable
import sys

import importlib.metadata as importlib_metadata

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import (
    AnswerResponse,
    initialize_answer_llm,
    normalize_text,
    token_overlap,
)
from benchmarks.runners.base_backend import Mem0BackendBase
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class DmrMscMem0Backend(Mem0BackendBase):
    def __init__(
        self,
        neo4j_uri: str | None,
        neo4j_user: str | None,
        neo4j_password: str | None,
        neo4j_database: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
    ):
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
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({size for size in ctx_values if size > 0})
        self.answer_llm_client = initialize_answer_llm(
            model=answer_llm_model or os.getenv('DMR_MSC_ANSWER_LLM_MODEL'),
            api_key=answer_llm_api_key or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY') or os.getenv('OPENAI_API_KEY'),
            base_url=answer_llm_base_url or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL'),
        )

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
        mem_user = os.getenv('MEM0_USER_ID', 'dmr_msc')
        mem0_dir = os.getenv('MEM0_DIR', str(Path('outputs/mem0_dmr_msc').resolve()))
        os.environ['MEM0_DIR'] = mem0_dir
        self._neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self._neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', '')
        self._neo4j_database = neo4j_database or os.getenv('DMR_MSC_NEO4J_DATABASE') or 'neo4j'

        neo4j_cfg = Neo4jConfig(
            url=self._neo4j_uri,
            username=self._neo4j_user,
            password=self._neo4j_password,
            database=self._neo4j_database,
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
        mem0_llm_model = answer_llm_model or os.getenv('OPENAI_MODEL') or os.getenv('DMR_MSC_ANSWER_LLM_MODEL')
        mem0_llm_api_key = answer_llm_api_key or os.getenv('OPENAI_API_KEY') or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY')
        mem0_llm_base_url = answer_llm_base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL')
        config.llm.provider = 'openai'
        config.llm.config = {
            'model': mem0_llm_model,
            'api_key': mem0_llm_api_key,
            'openai_base_url': mem0_llm_base_url,
        }
        super().__init__(mem0_config=config, mem0_user_id=mem_user, usage_interval_env='DMR_MSC_USAGE_INTERVAL', mem0_dir=mem0_dir)
        self._maybe_wipe_all_stores()
        self._log_graph_status()

    def _user_id(self, example: DmrMscExample) -> str:
        return f'{self.mem_user_id}_{example.example_id}'

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

    async def ingest_example(self, example: DmrMscExample) -> None:
        self._wipe_all_stores()
        user_id = self._user_id(example)
        for episode in example.episodes:
            metadata = {
                'segment': episode.metadata.get('segment'),
                'type': episode.metadata.get('type'),
                'window_id': example.window_id,
            }
            try:
                self._add_memory(
                    user_id=user_id,
                    message=episode.content,
                    metadata=metadata,
                )
                self.increment_episode_count()
            except Exception as exc:
                logger.warning('Mem0 add failed: %s', exc)
        self._log_graph_counts()

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, retrieval_details = await self._retrieve_answer_support(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        self._wipe_all_stores()
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve_answer_support(self, example: DmrMscExample) -> tuple[dict, list[str], dict]:
        user_id = self._user_id(example)
        try:
            retrieval = self.memory.search(example.question, user_id=user_id, limit=self.search_limit)
            docs: list[str] = []
            if isinstance(retrieval, dict) and 'results' in retrieval:
                docs = [res.get('memory', '') if isinstance(res, dict) else str(res) for res in retrieval.get('results', [])]
            elif isinstance(retrieval, list):
                docs = [res.get('memory', '') if isinstance(res, dict) else str(res) for res in retrieval]
            else:
                docs = []
        except Exception as exc:
            logger.warning('Mem0 search failed for %s: %s', example.example_id, exc)
            docs = []

        normalized_answer = normalize_text(example.answer)
        match_rank: int | None = None
        matching_fact = ''
        top_fact = ''
        for idx, fact in enumerate(docs, start=1):
            fact = (fact or '').strip()
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
            'retrieval_results_returned': float(len(docs)),
        }
        details = {
            'matching_fact': matching_fact,
            'top_fact': top_fact,
        }
        return retrieval_metrics, docs, details

    async def _answer_with_llm(self, example: DmrMscExample, context_docs: Iterable[str]) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes:
            return metrics, answers
        doc_list = list(context_docs)
        if not doc_list:
            return metrics, answers
        client = self.answer_llm_client

        for k in self.answer_context_sizes:
            subset = [doc for doc in doc_list[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_exact'] = 0.0
                metrics[f'llm@{k}_f1'] = 0.0
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            system_prompt = 'You are a careful assistant. Use only the provided facts to answer the question.'
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
                response = await client.generate_response(
                    messages,
                    response_model=AnswerResponse,
                )
                generated_answer = response.get('answer', '')
            except Exception as exc:
                logger.warning(
                    'Mem0 answer failed for %s (top-%s context): %s',
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

    def _log_graph_status(self) -> None:
        try:
            logger.info(
                'Mem0 graph enabled=%s neo4j=%s',
                getattr(self.memory, 'enable_graph', False),
                getattr(self.memory.graph.config, 'url', None) if getattr(self.memory, 'graph', None) else None,
            )
        except Exception:
            pass

    def _log_graph_counts(self) -> None:
        try:
            graph = getattr(self.memory, 'graph', None)
            if not graph:
                return
            res = graph.query(
                """
                MATCH (n) RETURN count(n) AS nodes
                """,
                params={},
            )
            nodes = res[0].get('nodes', 0) if res else 0
            res_rel = graph.query(
                """
                MATCH ()-[r]-() RETURN count(r) AS rels
                """,
                params={},
            )
            rels = res_rel[0].get('rels', 0) if res_rel else 0
            logger.info('Mem0 graph counts: nodes=%s rels=%s', nodes, rels)
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
            logger.info('Mem0 stores wiped (vector + graph)')
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
                    result = session.run(
                        """
                        CALL {
                            MATCH (n)
                            WITH n LIMIT $batch_size
                            DETACH DELETE n
                            RETURN COUNT(*) AS deleted
                        }
                        RETURN deleted
                        """,
                        batch_size=1000,
                    ).single()
                    deleted = result['deleted'] if result else 0
                    total_deleted += deleted
                    if deleted == 0:
                        break
                logger.info('Cleared %s nodes from Neo4j (db=%s)', total_deleted, self._neo4j_database)
        finally:
            driver.close()
