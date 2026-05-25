"""Mem0 backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

import importlib.metadata as importlib_metadata

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import AnswerResponse, initialize_answer_llm
from benchmarks.runners.base_backend import Mem0BackendBase
from benchmarks.types import FinTMMBenchExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class FinTMMBenchMem0Backend(Mem0BackendBase):
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
        self.search_limit = search_limit or int(os.getenv('FINTMMBENCH_SEARCH_LIMIT', '50'))
        ctx_values: list[int] = []
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        else:
            ctx_env = os.getenv('FINTMMBENCH_CONTEXT_K')
            if ctx_env:
                ctx_values = [
                    int(v.strip()) for v in ctx_env.split(',') if v.strip().isdigit()
                ]
        if not ctx_values:
            ctx_values = [3, 5, 10]
        self.answer_context_sizes = sorted({s for s in ctx_values if s > 0})
        self.answer_llm_client = initialize_answer_llm(
            model=answer_llm_model or os.getenv('FINTMMBENCH_ANSWER_LLM_MODEL'),
            api_key=answer_llm_api_key or os.getenv('FINTMMBENCH_ANSWER_LLM_API_KEY') or os.getenv('OPENAI_API_KEY'),
            base_url=answer_llm_base_url or os.getenv('FINTMMBENCH_ANSWER_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL'),
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

        mem_user = os.getenv('MEM0_USER_ID', 'fintmmbench')
        mem0_dir = os.getenv('MEM0_DIR', str(Path('outputs/mem0_fintmmbench').resolve()))
        os.environ['MEM0_DIR'] = mem0_dir
        self._neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self._neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self._neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', '')
        self._neo4j_database = neo4j_database or os.getenv('FINTMMBENCH_NEO4J_DATABASE') or 'neo4j'

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
        mem0_llm_model = answer_llm_model or os.getenv('OPENAI_MODEL') or os.getenv('FINTMMBENCH_ANSWER_LLM_MODEL')
        mem0_llm_api_key = answer_llm_api_key or os.getenv('OPENAI_API_KEY') or os.getenv('FINTMMBENCH_ANSWER_LLM_API_KEY')
        mem0_llm_base_url = answer_llm_base_url or os.getenv('OPENAI_BASE_URL') or os.getenv('FINTMMBENCH_ANSWER_LLM_BASE_URL')
        config.llm.provider = 'openai'
        config.llm.config = {
            'model': mem0_llm_model,
            'api_key': mem0_llm_api_key,
            'openai_base_url': mem0_llm_base_url,
        }
        super().__init__(mem0_config=config, mem0_user_id=mem_user, usage_interval_env='FINTMMBENCH_USAGE_INTERVAL', mem0_dir=mem0_dir)

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the entire FinTMMBench corpus into Mem0 (one-time)."""
        from benchmarks.loaders.fintmmbench import _doc_to_text, _parse_date
        from tqdm import tqdm

        by_date: dict[str, list[tuple[str, str]]] = {}
        for uid, doc in corpus.items():
            text = _doc_to_text(doc)
            if not text:
                continue
            date_str = doc.get('Date', '2022-01-01')
            by_date.setdefault(date_str, []).append((uid, text))

        total_docs = sum(len(v) for v in by_date.values())
        logger.info('Ingesting %d corpus documents across %d dates', total_docs, len(by_date))

        doc_count = 0
        progress = tqdm(total=total_docs, desc='Ingesting corpus')
        for date_str in sorted(by_date.keys()):
            for uid, text in by_date[date_str]:
                try:
                    self.memory.add(
                        messages=text,
                        user_id=self.mem_user_id,
                        metadata={'uuid': uid, 'date': date_str},
                        infer=False,
                    )
                    self.increment_episode_count()
                except Exception as exc:
                    logger.warning('Mem0 add failed for %s: %s', uid, exc)
                doc_count += 1
                progress.update(1)
        progress.close()

        logger.info('Corpus ingestion complete (%d documents)', total_docs)

    async def evaluate_example(self, example: FinTMMBenchExample) -> tuple[dict, dict, dict]:
        retrieval_metrics, context, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(self, example: FinTMMBenchExample) -> tuple[dict, list[str], dict]:
        try:
            retrieval = self.memory.search(example.question, user_id=self.mem_user_id, limit=self.search_limit)
            raw_results: list[dict] = []
            if isinstance(retrieval, dict) and 'results' in retrieval:
                raw_results = [r for r in retrieval.get('results', []) if isinstance(r, dict)]
            elif isinstance(retrieval, list):
                raw_results = [r for r in retrieval if isinstance(r, dict)]
            docs = [r.get('memory', '') or str(r) for r in raw_results]
        except Exception as exc:
            logger.warning('Mem0 search failed for %s: %s', example.uuid, exc)
            raw_results = []
            docs = []

        # Compute retrieval metrics via source UUID stored in metadata
        gold_ids = set(example.source_ids)
        RECALL_KS = [1, 3, 5, 10]

        matched_uids: list[str | None] = [None] * len(raw_results)
        first_match_rank = None
        for idx, res in enumerate(raw_results):
            uid = (res.get('metadata') or {}).get('uuid')
            if uid and uid in gold_ids:
                matched_uids[idx] = uid
                if first_match_rank is None:
                    first_match_rank = idx + 1

        mrr = 1.0 / first_match_rank if first_match_rank else 0.0

        metrics: dict[str, float] = {
            'retrieval_mrr': mrr,
            'retrieval_results_returned': float(len(docs)),
            'retrieval_gold_sources': float(len(gold_ids)),
        }
        for k in RECALL_KS:
            found = set(u for u in matched_uids[:k] if u is not None)
            metrics[f'retrieval_recall@{k}'] = len(found) / len(gold_ids) if gold_ids else 0.0
        details = {'top_doc': docs[0] if docs else ''}
        return metrics, docs, details

    async def _answer_with_llm(
        self, example: FinTMMBenchExample, context_docs: list[str],
    ) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes or not context_docs:
            return metrics, answers
        client = self.answer_llm_client

        for k in self.answer_context_sizes:
            subset = [doc for doc in context_docs[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_exact'] = 0.0
                metrics[f'llm@{k}_f1'] = 0.0
                metrics[f'llm@{k}_accuracy'] = 0.0
                continue
            system_prompt = (
                'You are a financial analyst assistant. '
                'Use only the provided financial data to answer the question. '
                'Be concise and precise.'
            )
            user_prompt = f"""Question: {example.question}

Financial data:
{os.linesep.join(f'- {doc}' for doc in subset)}

Provide a short answer grounded in the data. Respond with JSON as {{"answer": "<text>"}}."""
            messages = [
                Message(role='system', content=system_prompt),
                Message(role='user', content=user_prompt),
            ]
            generated_answer = ''
            try:
                response = await client.generate_response(messages, response_model=AnswerResponse)
                generated_answer = response.get('answer', '')
            except Exception as exc:
                logger.warning('LLM answer failed for %s (top-%d): %s', example.uuid, k, exc)

            answers[k] = generated_answer
            is_correct, _ = await score_answer(example.question, example.answer, generated_answer)
            metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0
        return metrics, answers


