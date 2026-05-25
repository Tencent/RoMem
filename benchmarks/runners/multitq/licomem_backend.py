"""LiCoMemory backend for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI

from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.types import MultiTQQuestion, TemporalTriple

logger = logging.getLogger(__name__)

# Add baselines/LiCoMemory to sys.path so LiCoMemory modules can be imported
_LICOMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'LiCoMemory')
if _LICOMEM_ROOT not in sys.path:
    sys.path.insert(0, _LICOMEM_ROOT)


class MultiTQLiCoMemBackend:
    """MultiTQ backend that uses LiCoMemory (Cognigraph) for memory storage and retrieval."""

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
        self.search_limit = search_limit or int(os.getenv('MULTITQ_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

        ctx_values: list[int] = []
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        else:
            ctx_env = os.getenv('MULTITQ_CONTEXT_K')
            if ctx_env:
                ctx_values = [
                    int(v.strip()) for v in ctx_env.split(',') if v.strip().isdigit()
                ]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({s for s in ctx_values if s > 0})

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('MULTITQ_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        llm_api_key = (
            answer_llm_api_key
            or os.getenv('MULTITQ_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        llm_base_url = (
            answer_llm_base_url
            or os.getenv('MULTITQ_ANSWER_LLM_BASE_URL')
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
            or str(Path('outputs/licomem_multitq').resolve())
        )

        # Will be set after ingest_all_triples
        self._graph_rag = None

    def _build_config(self, working_dir: str):
        """Build LiCoMemory Config programmatically (no YAML file needed)."""
        from init.config import (
            Config, LLMConfig, QueryLLMConfig, EmbeddingConfig,
            ChunkConfig, GraphConfig, RetrieverConfig, QueryConfig,
            StorageConfig, EvaluationConfig,
        )
        config = Config()
        config.index_name = 'multitq_graph'
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

    async def ingest_all_triples(self, triples: Iterable[TemporalTriple]) -> None:
        """Ingest KG triples directly into LiCoMemory's graph.

        Bypasses LLM-based entity/relation extraction — triples are already
        structured, so we build the NetworkX graph and embedding index directly.
        Result is cached as a pkl file for subsequent runs.
        """
        from init.graph_rag import GraphRAG

        working_dir = str(Path(self._licomem_base_dir) / 'graph')
        os.makedirs(working_dir, exist_ok=True)

        cached_pkl = os.path.join(working_dir, 'multitq_graph.pkl')
        use_cache = os.path.isfile(cached_pkl)

        config = self._build_config(working_dir)
        if use_cache:
            config.graph.force = False
            config.graph.add = False
        graph_rag = GraphRAG(config, base_dir=working_dir)

        if use_cache:
            print(f'[LiCoMemory/MultiTQ] Loading cached graph from {cached_pkl}', flush=True)
            self._graph_rag = graph_rag
            return

        all_triples = list(triples)
        print(
            f'[LiCoMemory/MultiTQ] Building graph directly from {len(all_triples)} triples '
            f'(no LLM extraction)...',
            flush=True,
        )

        # Build entities and relationships dicts for GraphBuilder
        entities: list[dict] = []
        relationships: list[dict] = []
        entity_set: set[str] = set()
        chunks: list[dict] = []

        for idx, t in enumerate(all_triples):
            date_str = t.timestamp.strftime('%Y-%m-%d') if t.timestamp else f'time_{t.time_id}'
            session_id = f'time_{t.time_id}'
            chunk_text = f'On {date_str}, {t.head} {t.relation} {t.tail}.'
            chunk_id = idx

            # Collect unique entities
            for ent_name in (t.head, t.tail):
                if ent_name not in entity_set:
                    entity_set.add(ent_name)
                    entities.append({
                        'entity': ent_name,
                        'type': 'entity',
                        'chunk_id': chunk_id,
                        'description': '',
                    })

            # Build relationship (keys must match what GraphBuilder expects)
            relationships.append({
                'src': t.head,
                'tgt': t.tail,
                'relation': t.relation,
                'chunk_id': chunk_id,
                'session_id': session_id,
                'session_time': date_str,
                'description': chunk_text,
            })

            # Store chunk for retrieval
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'session_id': session_id,
                'session_time': date_str,
            })

        # Build graph directly via GraphBuilder
        dm = graph_rag.core.graph
        dm.graph_builder.build_from_entities_and_relationships(entities, relationships)

        # Store chunks for retrieval
        dm.chunk_storage = {c['chunk_id']: c for c in chunks}

        # Build entity index
        dm.entity_name_to_index = {
            e['entity']: i for i, e in enumerate(entities)
        }

        # Pre-compute embeddings for entities and relationships
        await dm._precompute_embeddings(entities, relationships)

        # Save the graph
        import pickle
        graph_data = {
            'graph': dm.graph_builder.graph,
            'chunk_storage': dm.chunk_storage,
        }
        with open(cached_pkl, 'wb') as f:
            pickle.dump(graph_data, f)

        self.increment_episode_count(len(all_triples))
        n_nodes = dm.graph_builder.graph.number_of_nodes()
        n_edges = dm.graph_builder.graph.number_of_edges()
        print(
            f'[LiCoMemory/MultiTQ] Graph built: {n_nodes} nodes, {n_edges} edges, '
            f'{len(chunks)} chunks. Saved to {cached_pkl}',
            flush=True,
        )

        self._graph_rag = graph_rag

    # ── Evaluation ─────────────────────────────────────────────────

    async def evaluate_question(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        """Evaluate a single MultiTQ question.

        Returns (retrieval_metrics, llm_metrics, details).
        """
        retrieval_metrics, context_docs, retrieval_details = await self._retrieve(question)
        llm_metrics, answers = await self._answer_with_llm(question, context_docs)
        accuracy_metrics, match_strategies = self._verify_answers(question, answers)
        details = {
            **retrieval_details,
            'llm_answers': answers,
            'context_docs': context_docs,
            'match_strategies': match_strategies,
        }
        return retrieval_metrics, {**llm_metrics, **accuracy_metrics}, details

    async def _retrieve(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], list[str], dict]:
        """Query LiCoMemory and compute retrieval metrics."""
        facts: list[str] = []
        result: dict = {}

        if self._graph_rag is not None:
            try:
                result = await self._graph_rag.query(
                    question.question,
                    question_time='2024/01/01 (Mon) 12:00',
                )
            except Exception as exc:
                logger.warning(
                    'LiCoMemory query failed for quid=%s: %s', question.quid, exc
                )

        facts = self._extract_facts_from_result(result)

        # Hits@k and MRR — find rank of first fact containing a gold answer
        match_rank: int | None = None
        matching_fact = ''
        for idx, fact in enumerate(facts, start=1):
            if not fact:
                continue
            fact_lower = fact.lower().replace('_', ' ')
            for answer in question.answers:
                if answer.lower().replace('_', ' ') in fact_lower:
                    match_rank = idx
                    matching_fact = fact
                    break
            if match_rank is not None:
                break

        metrics: dict[str, float] = {
            'hits@1': 1.0 if match_rank == 1 else 0.0,
            'hits@3': 1.0 if match_rank is not None and match_rank <= 3 else 0.0,
            'hits@10': 1.0 if match_rank is not None and match_rank <= 10 else 0.0,
            'mrr': 1.0 / match_rank if match_rank else 0.0,
        }

        # answer_in_context@k metrics
        for k in self.answer_context_sizes:
            subset = facts[:k]
            found = self._check_answer_in_context(question.answers, subset)
            metrics[f'answer_in_context@{k}'] = 1.0 if found else 0.0

        details = {'matching_fact': matching_fact}
        return metrics, facts, details

    async def _answer_with_llm(
        self, question: MultiTQQuestion, context_docs: list[str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        """Generate answers for each context-size k using the answer LLM."""
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None:
            return metrics, answers

        for k in self.answer_context_sizes:
            subset = [doc for doc in context_docs[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            facts_text = '\n'.join(f'- {doc}' for doc in subset)
            user_prompt = (
                f'Question: {question.question}\n\n'
                f'Supporting facts:\n{facts_text}\n\n'
                'Provide a concise answer grounded in the facts. '
                'Respond with JSON {"answer": "<text>"}.'
            )
            messages = [
                {'role': 'system', 'content': 'You answer questions using only provided facts.'},
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
                    'LLM answer failed for quid=%s (top-%s): %s',
                    question.quid, k, exc,
                )
            answers[k] = generated_answer
            metrics[f'llm@{k}_context_size'] = float(len(subset))
        return metrics, answers

    def _verify_answers(
        self, question: MultiTQQuestion, answers: dict[int, str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        """Verify each generated answer using verify_answer() from multitq_verifier."""
        acc_metrics: dict[str, float] = {}
        strategies: dict[int, str] = {}
        for k, prediction in answers.items():
            is_correct, strategy = verify_answer(
                prediction, question.answers, question.answer_type,
            )
            acc_metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0
            strategies[k] = strategy
        return acc_metrics, strategies

    # ── Helpers ─────────────────────────────────────────────────────

    def _extract_facts_from_result(self, result: dict | str) -> list[str]:
        """Extract text facts from a LiCoMemory query result dict."""
        facts: list[str] = []
        if isinstance(result, str):
            if result.strip():
                facts.append(result)
            return facts
        if not isinstance(result, dict):
            return facts

        # Prefer triples (most structured/informative)
        for triple in result.get('triples', []):
            if isinstance(triple, dict):
                src = triple.get('src', '')
                rel = triple.get('relation', '')
                tgt = triple.get('tgt', '')
                fact = f'{src} {rel} {tgt}'.strip()
                if fact:
                    facts.append(fact)

        # Also include chunks for richer context
        for chunk in result.get('chunks', []):
            if isinstance(chunk, str) and chunk.strip():
                if chunk not in facts:
                    facts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get('content') or chunk.get('text') or ''
                if text.strip() and text not in facts:
                    facts.append(text)

        return facts

    @staticmethod
    def _check_answer_in_context(answers: list[str], docs: list[str]) -> bool:
        context = ' '.join(docs).lower().replace('_', ' ')
        for answer in answers:
            if answer.lower().replace('_', ' ') in context:
                return True
        return False
