"""LiCoMemory backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

from benchmarks.evaluators.llm_judge import score_answer
from benchmarks.types import FinTMMBenchExample

logger = logging.getLogger(__name__)

# Add baselines/LiCoMemory to sys.path so LiCoMemory modules can be imported
_LICOMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'LiCoMemory')
if _LICOMEM_ROOT not in sys.path:
    sys.path.insert(0, _LICOMEM_ROOT)


class FinTMMBenchLiCoMemBackend:
    """FinTMMBench backend that uses LiCoMemory (Cognigraph) for memory storage and retrieval."""

    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
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
        self.search_limit = search_limit or int(os.getenv('FINTMMBENCH_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

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

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('FINTMMBENCH_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        llm_api_key = (
            answer_llm_api_key
            or os.getenv('FINTMMBENCH_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        llm_base_url = (
            answer_llm_base_url
            or os.getenv('FINTMMBENCH_ANSWER_LLM_BASE_URL')
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
            or 'gpt-4o-mini'
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
            or str(Path('outputs/licomem_fintmmbench').resolve())
        )

        # Will be set after ingest_corpus
        self._graph_rag = None
        # Maps session_id → source document uid
        self._session_to_uid: dict[str, str] = {}

    def _build_config(self, working_dir: str):
        """Build LiCoMemory Config programmatically (no YAML file needed)."""
        from init.config import (
            Config, LLMConfig, QueryLLMConfig, EmbeddingConfig,
            ChunkConfig, GraphConfig, RetrieverConfig, QueryConfig,
            StorageConfig, EvaluationConfig,
        )
        config = Config()
        config.index_name = 'fintmmbench_graph'
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

    # ── Ingestion ──────────────────────────────────────────────────

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the entire FinTMMBench corpus into LiCoMemory (one-time).

        Documents are grouped by date so each date becomes a session.
        A session_id→uid mapping is built so retrieval recall can be computed.
        The graph is cached in a pkl file for subsequent runs.
        """
        from benchmarks.loaders.fintmmbench import _doc_to_text
        from init.graph_rag import GraphRAG

        working_dir = str(Path(self._licomem_base_dir) / 'graph')
        os.makedirs(working_dir, exist_ok=True)

        # Cache detection
        cached_pkl = os.path.join(working_dir, 'fintmmbench_graph.pkl')
        session_map_pkl = os.path.join(working_dir, 'session_to_uid.pkl')
        use_cache = os.path.isfile(cached_pkl)

        config = self._build_config(working_dir)
        if use_cache:
            config.graph.force = False
            config.graph.add = False
        graph_rag = GraphRAG(config, base_dir=working_dir)

        if use_cache:
            print(
                f'[LiCoMemory/FinTMMBench] Loading cached graph from {cached_pkl}',
                flush=True,
            )
            # Restore session→uid map and uid→text if they exist
            if os.path.isfile(session_map_pkl):
                import pickle
                with open(session_map_pkl, 'rb') as f:
                    saved = pickle.load(f)
                if isinstance(saved, dict) and 'session_to_uid' in saved:
                    self._session_to_uid = saved['session_to_uid']
                    self._uid_to_text = saved.get('uid_to_text', {})
                else:
                    # Backward compat: old cache was just session_to_uid dict
                    self._session_to_uid = saved if isinstance(saved, dict) else {}
                    self._uid_to_text = {}
            self._graph_rag = graph_rag
            return

        # Group docs by date → one session per date
        by_date: dict[str, list[tuple[str, str]]] = {}
        for uid, doc in corpus.items():
            text = _doc_to_text(doc)
            if not text:
                continue
            date_str = doc.get('Date', '2022-01-01')
            by_date.setdefault(date_str, []).append((uid, text))

        total_docs = sum(len(v) for v in by_date.values())
        print(
            f'[LiCoMemory/FinTMMBench] Ingesting {total_docs} documents '
            f'across {len(by_date)} dates...',
            flush=True,
        )

        licomem_corpus = []
        session_to_uid: dict[str, str] = {}
        uid_to_text: dict[str, str] = {}
        session_counter = 0
        for date_str in sorted(by_date.keys()):
            for uid, text in by_date[date_str]:
                session_id = f'doc_{session_counter}'
                session_counter += 1
                session_to_uid[session_id] = uid
                uid_to_text[uid] = text

                escaped = text.replace('"', '\\"')
                context = f'"Narrator": "{escaped}"'
                licomem_corpus.append({
                    'session_id': session_id,
                    'context': context,
                    'content': text,
                    'session_time': date_str,
                })

        try:
            await graph_rag.insert(licomem_corpus)
            self.increment_episode_count(len(licomem_corpus))
            print(
                f'[LiCoMemory/FinTMMBench] Inserted {len(licomem_corpus)} sessions.',
                flush=True,
            )
        except Exception as exc:
            logger.warning('LiCoMemory insert failed for FinTMMBench corpus: %s', exc)

        # Persist session→uid mapping and uid→text alongside the graph
        self._session_to_uid = session_to_uid
        self._uid_to_text = uid_to_text
        try:
            import pickle
            with open(session_map_pkl, 'wb') as f:
                pickle.dump({'session_to_uid': session_to_uid, 'uid_to_text': uid_to_text}, f)
        except Exception as exc:
            logger.warning('Could not save session map: %s', exc)

        self._graph_rag = graph_rag

    # ── Evaluation ─────────────────────────────────────────────────

    async def evaluate_example(
        self, example: FinTMMBenchExample
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        """Evaluate a single FinTMMBench example.

        Returns (retrieval_metrics, llm_metrics, details).
        """
        retrieval_metrics, context_docs, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context_docs)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(
        self, example: FinTMMBenchExample
    ) -> tuple[dict[str, float], list[str], dict]:
        """Query LiCoMemory and compute retrieval recall metrics."""
        result: dict = {}
        if self._graph_rag is not None:
            try:
                result = await self._graph_rag.query(
                    example.question,
                    question_time='2024/01/01 (Mon) 12:00',
                )
            except Exception as exc:
                logger.warning(
                    'LiCoMemory query failed for %s: %s', example.uuid, exc
                )

        # Extract text facts for LLM answering
        docs = self._extract_docs_from_result(result)

        # Extract session IDs from result for retrieval recall
        retrieved_session_ids = self._extract_session_ids_from_result(result)

        # Map session IDs → source UIDs and check against gold source_ids
        gold_ids = set(example.source_ids)
        RECALL_KS = [1, 3, 5, 10]

        retrieved_uids: list[str | None] = []
        for sid in retrieved_session_ids:
            uid = self._session_to_uid.get(sid)
            retrieved_uids.append(uid)

        first_match_rank: int | None = None
        for idx, uid in enumerate(retrieved_uids, start=1):
            if uid and uid in gold_ids:
                first_match_rank = idx
                break

        mrr = 1.0 / first_match_rank if first_match_rank else 0.0
        metrics: dict[str, float] = {
            'retrieval_mrr': mrr,
            'retrieval_results_returned': float(len(docs)),
            'retrieval_gold_sources': float(len(gold_ids)),
        }
        for k in RECALL_KS:
            found = {u for u in retrieved_uids[:k] if u is not None and u in gold_ids}
            metrics[f'retrieval_recall@{k}'] = len(found) / len(gold_ids) if gold_ids else 0.0

        # Build full-document context for LLM answering (deduplicated, ranked)
        uid_to_text = getattr(self, '_uid_to_text', {})
        full_docs: list[str] = []
        seen_uids: set[str] = set()
        for uid in retrieved_uids:
            if uid and uid not in seen_uids:
                seen_uids.add(uid)
                text = uid_to_text.get(uid, '')
                if text:
                    full_docs.append(text)
        # Fallback to chunk-level docs if no full docs found
        if not full_docs:
            full_docs = docs

        details = {'top_doc': full_docs[0] if full_docs else ''}
        return metrics, full_docs, details

    async def _answer_with_llm(
        self, example: FinTMMBenchExample, context_docs: list[str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        """Generate answers for each context-size k using the answer LLM."""
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None or not context_docs:
            return metrics, answers

        for k in self.answer_context_sizes:
            subset = [doc for doc in context_docs[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_accuracy'] = 0.0
                continue
            system_prompt = (
                'You are a financial analyst assistant. '
                'Use only the provided financial data to answer the question. '
                'Be concise and precise.'
            )
            user_prompt = (
                f'Question: {example.question}\n\n'
                f'Financial data:\n'
                + '\n'.join(f'- {doc}' for doc in subset)
                + '\n\nProvide a short answer grounded in the data. '
                'Respond with JSON {"answer": "<text>"}.'
            )
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            generated_answer = ''
            try:
                response = await client.chat.completions.create(
                    model=self.answer_llm_model,
                    messages=messages,
                )
                raw = response.choices[0].message.content or ''
                try:
                    parsed = json.loads(raw)
                    generated_answer = str(parsed.get('answer', raw)).strip()
                except json.JSONDecodeError:
                    generated_answer = raw.strip()
            except Exception as exc:
                logger.warning(
                    'LLM answer failed for %s (top-%d): %s',
                    example.uuid, k, exc,
                )

            answers[k] = generated_answer
            is_correct, _ = await score_answer(
                example.question, example.answer, generated_answer
            )
            metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0

        return metrics, answers

    # ── Helpers ─────────────────────────────────────────────────────

    def _extract_docs_from_result(self, result: dict | str) -> list[str]:
        """Extract text documents from a LiCoMemory query result dict.

        Prioritises full chunk text (richer context for LLM answering)
        over short triple strings.
        """
        docs: list[str] = []
        seen: set[str] = set()
        if isinstance(result, str):
            if result.strip():
                docs.append(result)
            return docs
        if not isinstance(result, dict):
            return docs

        # 1. Chunks first — full document text gives the LLM better context
        for chunk in result.get('chunks', []):
            text = ''
            if isinstance(chunk, str):
                text = chunk.strip()
            elif isinstance(chunk, dict):
                text = (chunk.get('content') or chunk.get('text') or '').strip()
            if text and text not in seen:
                seen.add(text)
                docs.append(text)

        # 2. Triples as fallback if chunks are sparse
        for triple in result.get('triples', []):
            if isinstance(triple, dict):
                desc = triple.get('description', '')
                if desc and desc not in seen:
                    seen.add(desc)
                    docs.append(desc)
                else:
                    src = triple.get('src', '')
                    rel = triple.get('relation', '')
                    tgt = triple.get('tgt', '')
                    fact = f'{src} {rel} {tgt}'.strip()
                    if fact and fact not in seen:
                        seen.add(fact)
                        docs.append(fact)

        return docs

    def _extract_session_ids_from_result(
        self, result: dict | str, max_ids: int = 50
    ) -> list[str]:
        """Extract top-K session IDs from a LiCoMemory query result dict.

        Session IDs are ordered by relevance (triples first, then chunks).
        """
        if not isinstance(result, dict):
            return []
        seen: set[str] = set()
        ordered: list[str] = []
        for triple in result.get('triples', []):
            if len(ordered) >= max_ids:
                break
            sid = triple.get('session_id', '') if isinstance(triple, dict) else ''
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        for chunk in result.get('chunks', []):
            if len(ordered) >= max_ids:
                break
            sid = chunk.get('session_id', '') if isinstance(chunk, dict) else ''
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        return ordered
