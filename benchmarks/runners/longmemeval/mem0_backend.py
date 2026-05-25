"""Mem0 backend for LongMemEval using local Mem0 (FAISS + optional graph) in line with Mem0 add/search semantics."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List

import importlib.metadata as importlib_metadata
from pydantic import BaseModel

from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from benchmarks.runners.base_backend import Mem0BackendBase
from benchmarks.types import LongmemevalExample
from dataset.longmemeval.code.retrieval.eval_utils import evaluate_retrieval

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class JudgeResponse(BaseModel):
    label: str


class LongmemevalMem0Backend(Mem0BackendBase):
    def __init__(
        self,
        neo4j_uri: str | None,
        neo4j_user: str | None,
        neo4j_password: str | None,
        neo4j_database: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: List[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        judge_llm_model: str | None = None,
        judge_llm_api_key: str | None = None,
        judge_llm_base_url: str | None = None,
        disable_judge: bool = False,
    ):
        self.search_limit = search_limit or int(os.getenv('LONGMEM_SEARCH_LIMIT', '50'))
        if answer_context_sizes is None:
            ctx_env = os.getenv('LONGMEM_CONTEXT_K', '5,10')
            answer_context_sizes = [int(k) for k in ctx_env.split(',') if k]
        self.answer_context_sizes = sorted(set(answer_context_sizes))
        self.answer_llm_model = answer_llm_model or os.getenv('LONGMEM_ANSWER_LLM_MODEL') or os.getenv('OPENAI_MODEL')
        self.answer_llm_api_key = answer_llm_api_key or os.getenv('LONGMEM_ANSWER_LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.answer_llm_base_url = answer_llm_base_url or os.getenv('LONGMEM_ANSWER_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL')
        self.disable_judge = disable_judge
        self.judge_llm_model = judge_llm_model or os.getenv('LONGMEM_JUDGE_LLM_MODEL')
        self.judge_llm_api_key = judge_llm_api_key or os.getenv('LONGMEM_JUDGE_LLM_API_KEY')
        self.judge_llm_base_url = judge_llm_base_url or os.getenv('LONGMEM_JUDGE_LLM_BASE_URL')
        self._answer_session_ids: set[str] = set()
        # Ensure mem0 is importable from baselines/mem0
        mem0_root = Path(__file__).resolve().parents[3] / 'baselines' / 'mem0'
        mem0_path = str(mem0_root)
        if mem0_path not in sys.path:
            sys.path.insert(0, mem0_path)
        # Mem0 __init__ calls importlib.metadata.version("mem0ai"); fake it if not installed, then restore.
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
            from mem0.memory.graph_memory import MemoryGraph
        finally:
            importlib_metadata.version = _orig_version  # type: ignore
        mem_user = os.getenv('MEM0_USER_ID', 'longmemeval')
        mem0_dir = os.getenv('MEM0_DIR', str(Path('outputs/mem0_longmemeval').resolve()))
        os.environ['MEM0_DIR'] = mem0_dir
        neo4j_cfg = Neo4jConfig(
            url=neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=neo4j_user or os.getenv('NEO4J_USER', 'neo4j'),
            password=neo4j_password or os.getenv('NEO4J_PASSWORD', ''),
            database=neo4j_database or os.getenv('LONGMEMEVAL_NEO4J_DATABASE'),
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
            path=os.getenv('MEM0_VECTOR_PATH', 'outputs/mem0_longmemeval/faiss_run'),
            collection_name=mem_user,
            embedding_model_dims=mem0_embed_dims,
            distance_strategy='cosine',
            normalize_L2=True,
        )
        # override mem0 internal LLM for extraction/update (single OpenAI-compatible path)
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
        self._neo4j_database = neo4j_database or os.getenv('LONGMEMEVAL_NEO4J_DATABASE', None) or 'neo4j'
        # Patch Mem0 entity parsing to skip malformed entries rather than crashing
        self._patch_mem0_entity_extraction(MemoryGraph)
        self._patch_mem0_relationship_cleanup(MemoryGraph)
        self._patch_mem0_graph_add(MemoryGraph)
        super().__init__(mem0_config=config, mem0_user_id=mem_user, usage_interval_env='LONGMEMEVAL_USAGE_INTERVAL', mem0_dir=mem0_dir)
        self._maybe_wipe_all_stores()
        self._log_graph_status()

    async def clear_state(self) -> None:
        self._answer_session_ids = set()
        self._answer_turn_ids: set[str] = set()

    def _user_id(self, example: LongmemevalExample) -> str:
        return f'{self.mem_user_id}_{example.question_id}'

    def _add_memory(self, user_id: str, messages, metadata: dict, retries: int = 3):
        for attempt in range(retries):
            try:
                return self.memory.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata,
                    infer=False,
                )
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning('Mem0 add retry %s for %s: %s', attempt + 1, user_id, exc)

    async def ingest_episode(self, example: LongmemevalExample, episode_text: str, reference_time, source_description: str | None = None, turn_id: str | None = None) -> None:
        user_id = self._user_id(example)
        messages = episode_text
        self._add_memory(
            user_id=user_id,
            messages=messages,
            metadata={'reference_time': str(reference_time), 'session_id': source_description, 'turn_id': turn_id or ''},
        )
        self.increment_episode_count()

    async def ingest_example(self, example: LongmemevalExample) -> None:
        await self.clear_state()
        user_id = self._user_id(example)
        self._wipe_all_stores()
        turn_counts: dict[str, int] = {}
        for episode in example.episodes:
            session_id = episode.metadata.get('session_id')
            turn_id = None
            if session_id:
                turn_idx = turn_counts.get(session_id, 0) + 1
                turn_counts[session_id] = turn_idx
                turn_id = f'{session_id}_{turn_idx}'
                if episode.metadata.get('has_answer'):
                    self._answer_turn_ids.add(turn_id)
            await self.ingest_episode(
                example=example,
                episode_text=episode.content,
                reference_time=episode.reference_time,
                source_description=session_id or 'longmemeval',
                turn_id=turn_id,
            )
            if episode.metadata.get('is_answer_session') or episode.metadata.get('has_answer'):
                if session_id:
                    self._answer_session_ids.add(session_id)
        # log ingestion success
        logger.info('Mem0 added %s episodes for user %s', len(example.episodes), user_id)
        self._log_graph_counts(user_id)

    async def evaluate_example(self, example: LongmemevalExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, details = await self._retrieve(example)
        llm_metrics, answers = await self._answer(example, context)
        judge_metrics = await self._judge_answers(example, answers)
        await self.clear_state()
        try:
            self.memory.delete_all(user_id=self._user_id(example))
        except Exception:
            pass
        return (
            retrieval_metrics,
            {**llm_metrics, **judge_metrics},
            {**details, 'llm_answers': answers, 'context_docs': context},
        )

    async def _retrieve(self, example: LongmemevalExample) -> tuple[dict, list, dict]:
        user_id = self._user_id(example)
        try:
            retrieval = self.memory.search(
                example.question,
                user_id=user_id,
                limit=self.search_limit,
            )
        except Exception as exc:
            logger.warning('Mem0 search failed for %s: %s', example.question_id, exc)
            retrieval = []
        docs: list[str] = []
        ranked_turns: list[str] = []
        snippet = ''

        def _extract(res: dict) -> None:
            mem = res.get('memory') or res.get('text') or ''
            meta = res.get('metadata') or {}
            turn_id = meta.get('turn_id') or ''
            docs.append(mem)
            if turn_id:
                ranked_turns.append(turn_id)

        if isinstance(retrieval, dict) and 'results' in retrieval:
            for res in retrieval.get('results', []):
                if res:
                    _extract(res)
        elif isinstance(retrieval, list):
            for res in retrieval:
                if isinstance(res, dict):
                    _extract(res)
                else:
                    docs.append(str(res))

        # Find first matching snippet
        for turn_id, doc in zip(ranked_turns, docs):
            if turn_id in self._answer_turn_ids and doc:
                snippet = doc
                break

        metrics: dict[str, float] = {}
        if not ranked_turns or not self._answer_turn_ids:
            for k in (5, 10, 15):
                metrics[f'turn_recall_all@{k}'] = 0.0
                metrics[f'turn_ndcg_any@{k}'] = 0.0
            return metrics, docs, {'matching_fact': snippet}

        turn_corpus_ids = ranked_turns
        turn_rankings = list(range(len(turn_corpus_ids)))
        turn_correct_docs = list(self._answer_turn_ids)
        for k in (5, 10, 15):
            _recall_any, recall_all, ndcg_any = evaluate_retrieval(
                turn_rankings, turn_correct_docs, turn_corpus_ids, k=k
            )
            metrics[f'turn_recall_all@{k}'] = float(recall_all)
            metrics[f'turn_ndcg_any@{k}'] = float(ndcg_any)
        return metrics, docs, {'matching_fact': snippet}

    async def _answer(self, example: LongmemevalExample, context_docs: Iterable[str]) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self._get_answer_llm_client()
        if client is None:
            return metrics, answers
        docs_list = [d for d in context_docs if d]
        for k in self.answer_context_sizes:
            subset = docs_list[:k]
            if not subset:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            facts = [f'- {doc}' for doc in subset if doc]
            if not facts:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            prompt = f"""Question: {example.question}

Supporting facts:
{os.linesep.join(facts)}

Provide a concise answer grounded in the facts. Respond with JSON {{"answer": "<text>"}}."""
            messages = [
                Message(role='system', content='You answer questions using only provided facts.'),
                Message(role='user', content=prompt),
            ]
            prediction = ''
            try:
                response = await client.generate_response(messages, response_model=QAResponse)
                prediction = response.get('answer', '')
            except Exception as exc:
                logger.warning('LLM answer failed for %s (top-%s): %s', example.question_id, k, exc)
            answers[k] = prediction
            metrics[f'llm@{k}_context_size'] = float(len(facts))
        return metrics, answers

    async def _judge_answers(self, example: LongmemevalExample, answers: dict[int, str]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for k, prediction in answers.items():
            verdict = await self._judge_single_answer(example, prediction)
            metrics[f'llm@{k}_accuracy'] = 1.0 if verdict else 0.0
        return metrics

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

    async def _judge_single_answer(self, example: LongmemevalExample, prediction: str) -> bool:
        if not prediction:
            return False
        if self.disable_judge:
            return self._f1(prediction, example.answer) >= 0.7
        client = self._get_judge_llm_client()
        if client is None:
            return self._f1(prediction, example.answer) >= 0.7
        prompt = self._build_judge_prompt(example, prediction)
        if not prompt:
            return self._f1(prediction, example.answer) >= 0.7
        messages = [
            Message(
                role='system',
                content='You are a strict evaluator. Reply only with JSON {"label": "yes"} or {"label": "no"}.',
            ),
            Message(role='user', content=prompt),
        ]
        try:
            response = await client.generate_response(
                messages,
                response_model=JudgeResponse,
                max_tokens=32,
            )
            verdict = response.get('label', '').strip().lower()
            return verdict.startswith('y')
        except Exception as exc:
            logger.warning('Judge evaluation failed for %s: %s', example.question_id, exc)
            return self._f1(prediction, example.answer) >= 0.7

    def _get_answer_llm_client(self):
        if not self.answer_llm_api_key:
            return None
        cfg = LLMConfig(model=self.answer_llm_model, api_key=self.answer_llm_api_key, base_url=self.answer_llm_base_url)
        if (cfg.model or '').lower().startswith('gpt'):
            return OpenAIClient(config=cfg)
        return OpenAIGenericClient(config=cfg)

    def _get_judge_llm_client(self):
        model = self.judge_llm_model or self.answer_llm_model
        api_key = self.judge_llm_api_key or self.answer_llm_api_key
        base_url = self.judge_llm_base_url or self.answer_llm_base_url
        if not api_key:
            return None
        cfg = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        if (cfg.model or '').lower().startswith('gpt'):
            return OpenAIClient(config=cfg)
        return OpenAIGenericClient(config=cfg)

    def _f1(self, prediction: str, answer: str) -> float:
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(answer)
        if not pred_tokens or not ref_tokens:
            return 0.0
        ref_counts = {}
        for tok in ref_tokens:
            ref_counts[tok] = ref_counts.get(tok, 0) + 1
        matching = 0
        for tok in pred_tokens:
            if ref_counts.get(tok, 0) > 0:
                matching += 1
                ref_counts[tok] -= 1
        precision = matching / len(pred_tokens)
        recall = matching / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _build_judge_prompt(self, example: LongmemevalExample, prediction: str) -> str:
        question = example.question
        answer = example.answer
        qtype = example.question_type.replace('-', '_').lower()
        abstention = '_abs' in example.question_id.lower()
        if abstention:
            return ''
        prompt = f'Question type: {qtype}\nQuestion: {question}\nGold answer: {answer}\nModel answer: {prediction}\nIs the model answer correct?'
        return prompt

    @staticmethod
    def _patch_mem0_entity_extraction(MemoryGraph):
        if getattr(MemoryGraph, '_patched_skip_missing_entity', False):
            return
        orig = MemoryGraph._retrieve_nodes_from_data
        def _safe_retrieve(self, data, filters):
            try:
                return orig(self, data, filters)
            except KeyError as exc:
                logger.warning('Mem0 entity extraction failed (skip): %s', exc)
                return {}
        MemoryGraph._retrieve_nodes_from_data = _safe_retrieve
        MemoryGraph._patched_skip_missing_entity = True

    @staticmethod
    def _patch_mem0_relationship_cleanup(MemoryGraph):
        if getattr(MemoryGraph, '_patched_safe_relationships', False):
            return
        def _safe_remove(self, entities):
            cleaned = []
            for item in entities or []:
                if not isinstance(item, dict):
                    continue
                src = item.get('source')
                dst = item.get('destination')
                rel = item.get('relationship')
                if src is None or dst is None or rel is None:
                    logger.warning('Skipping malformed relation (missing keys): %s', item)
                    continue
                try:
                    cleaned.append(
                        {
                            'source': _sanitize_identifier(src),
                            'destination': _sanitize_identifier(dst),
                            'relationship': _sanitize_identifier(rel),
                        }
                    )
                except Exception as exc:
                    logger.warning('Skipping relation due to error: %s', exc)
            return cleaned
        MemoryGraph._remove_spaces_from_entities = _safe_remove
        MemoryGraph._patched_safe_relationships = True

    @staticmethod
    def _patch_mem0_graph_add(MemoryGraph):
        if getattr(MemoryGraph, '_patched_safe_add', False):
            return
        orig_add = MemoryGraph.add
        def _safe_add(self, data, filters):
            for attempt in range(2):
                # try:
                return orig_add(self, data, filters)
                # except Exception as exc:
                #     logger.warning('Mem0 graph add failed (attempt %s): %s', attempt + 1, exc)
            logger.warning('Mem0 graph add giving up after retries; skipping graph write.')
            return {"deleted_entities": [], "added_entities": []}
        MemoryGraph.add = _safe_add
        MemoryGraph._patched_safe_add = True


def _tokenize(value: str) -> list[str]:
    return [tok for tok in (value or '').lower().split() if tok]


def _sanitize_identifier(value) -> str:
    import re
    s = str(value).lower().strip()
    s = re.sub(r'[^a-z0-9_]', '_', s)
    return s
