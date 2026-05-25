"""
Shared backend utilities for Graphiti-based benchmarks.
"""

from __future__ import annotations

import logging
import os

import copy
import sys
from pathlib import Path
from typing import Optional

from baselines.graphiti.graphiti_core import Graphiti
from baselines.graphiti.graphiti_core.driver.neo4j_driver import Neo4jDriver
from baselines.graphiti.graphiti_core.llm_client.usage_tracker import usage_tracker
from baselines.graphiti.graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    COMBINED_HYBRID_SEARCH_MMR,
    COMBINED_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
    EDGE_HYBRID_SEARCH_MMR,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)

logger = logging.getLogger(__name__)


class GraphitiBackendBase:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        usage_interval_env: str = 'BENCHMARK_USAGE_INTERVAL',
        search_reranker: str | None = None,
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database or 'neo4j'
        self._init_graphiti()
        self.usage_log_interval = int(os.getenv(usage_interval_env, '0'))
        self._episodes_processed = 0
        self._search_reranker = (search_reranker or '').strip().lower()

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step
        if (
            self.usage_log_interval > 0
            and self._episodes_processed % self.usage_log_interval == 0
        ):
            self._log_usage_summary(prefix=f'episodes={self._episodes_processed}')

    def finalize_usage_logging(self) -> None:
        if self._episodes_processed:
            self.log_usage_summary(prefix='final')

    def log_usage_summary(self, prefix: str = '') -> None:
        self._log_usage_summary(prefix=prefix)

    def _log_usage_summary(self, prefix: str = '') -> None:
        summary = usage_tracker.summary()
        per_model = summary.get('per_model', {})
        totals = summary.get('totals', {})

        if prefix == 'final' or (
            self.usage_log_interval > 0 and self._episodes_processed % self.usage_log_interval == 0
        ):
            for model, stats in per_model.items():
                logger.info(
                    '[LLM][usage-summary] %s model=%s prompt=%s completion=%s cost=%.6f %s',
                    prefix,
                    model,
                    stats.get('prompt_tokens', 0),
                    stats.get('completion_tokens', 0),
                    stats.get('cost', 0.0),
                    stats.get('currency', ''),
                )

            if totals:
                logger.info(
                    '[LLM][usage-summary] %s totals prompt=%s completion=%s cost=%.6f',
                    prefix,
                    totals.get('prompt_tokens', 0),
                    totals.get('completion_tokens', 0),
                    totals.get('cost', 0.0),
                )

    def _init_graphiti(self) -> None:
        driver = Neo4jDriver(self._uri, self._user, self._password, database=self._database)
        self.graphiti = Graphiti(graph_driver=driver)

    async def _search_edges(
        self,
        query: str,
        num_results: int,
        group_ids: Optional[list[str]] = None,
    ):
        """
        Run an edge search with optional reranker override.
        """
        config = self._resolve_search_config(num_results)
        if config is None:
            return await self.graphiti.search(query, group_ids=group_ids, num_results=num_results)
        return (await self.graphiti.search_(query, config=config, group_ids=group_ids)).edges

    def _resolve_search_config(self, num_results: int):
        """
        Build a SearchConfig based on the configured reranker name.
        """
        name = self._search_reranker
        if not name or name == 'basic':
            return None
        mapping = {
            'cross_encoder': COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            'combined_cross_encoder': COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            'rrf': COMBINED_HYBRID_SEARCH_RRF,
            'combined_rrf': COMBINED_HYBRID_SEARCH_RRF,
            'mmr': COMBINED_HYBRID_SEARCH_MMR,
            'combined_mmr': COMBINED_HYBRID_SEARCH_MMR,
            'edge_rrf': EDGE_HYBRID_SEARCH_RRF,
            'edge_mmr': EDGE_HYBRID_SEARCH_MMR,
            'node_distance': EDGE_HYBRID_SEARCH_NODE_DISTANCE,
            'episode_mentions': EDGE_HYBRID_SEARCH_EPISODE_MENTIONS,
        }
        base = mapping.get(name)
        if base is None:
            raise ValueError(f'Unknown search_reranker: {name}')
        # Avoid mutating shared configs
        cfg = base.model_copy(deep=True) if hasattr(base, 'model_copy') else copy.deepcopy(base)
        cfg.limit = num_results
        return cfg


class HippoRAGBackendBase:
    """
    Base helpers for HippoRAG-backed implementations.
    """

    def __init__(
        self,
        *,
        hippo_save_dir: str | None = None,
        hippo_llm_model: str | None = None,
        hippo_embedding_model: str | None = None,
        hippo_openie_mode: str | None = None,
        hippo_retrieval_top_k: int | None = None,
    ):
        self._episodes_processed = 0
        self._hippo_opts = {
            'save_dir': hippo_save_dir,
            'llm_model': hippo_llm_model,
            'embedding_model': hippo_embedding_model,
            'openie_mode': hippo_openie_mode,
            'retrieval_top_k': hippo_retrieval_top_k,
        }
        self.hipporag = self._init_hipporag()

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def finalize_usage_logging(self) -> None:
        pass

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {}

    def _init_hipporag(self, sample_id: str | None = None):
        # Make sure HippoRAG package is importable
        hippo_root = Path(__file__).resolve().parents[3] / 'baselines' / 'HippoRAG' / 'src'
        hippo_path = str(hippo_root)
        if hippo_path not in sys.path:
            sys.path.insert(0, hippo_path)

        from baselines.HippoRAG.src.hipporag import HippoRAG  # type: ignore
        from baselines.HippoRAG.src.hipporag.utils.config_utils import BaseConfig  # type: ignore

        config = BaseConfig()
        if self._hippo_opts.get('save_dir'):
            base_dir = self._hippo_opts['save_dir']
            if sample_id:
                import re
                safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)
                base_dir = os.path.join(base_dir, safe_id)
            config.save_dir = base_dir
        if self._hippo_opts.get('llm_model'):
            config.llm_name = self._hippo_opts['llm_model']
        if self._hippo_opts.get('embedding_model'):
            # Strip RoMem-style package routing prefixes (e.g. "Transformers/",
            # "VLLM/") so HippoRAG receives a plain HF repo id.
            raw_embed = self._hippo_opts['embedding_model']
            for prefix in ('Transformers/', 'VLLM/'):
                if raw_embed.startswith(prefix):
                    raw_embed = raw_embed[len(prefix):]
                    break
            config.embedding_model_name = raw_embed
        if self._hippo_opts.get('openie_mode'):
            config.openie_mode = self._hippo_opts['openie_mode']
        if self._hippo_opts.get('retrieval_top_k') is not None:
            config.retrieval_top_k = self._hippo_opts['retrieval_top_k']
        config.linking_top_k = max(config.retrieval_top_k, 50)

        # Pick up vLLM / local-server base URLs from env if available.
        llm_base_url = os.getenv('HIPPO_LLM_BASE_URL') or os.getenv('VLLM_BASE_URL')
        embedding_base_url = os.getenv('HIPPO_EMBEDDING_BASE_URL')
        if llm_base_url:
            config.llm_base_url = llm_base_url
        if embedding_base_url:
            config.embedding_base_url = embedding_base_url

        return HippoRAG(
            global_config=config,
            save_dir=config.save_dir,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
            llm_base_url=config.llm_base_url,
            embedding_base_url=config.embedding_base_url,
        )

    def _hipporag_index_structured(
        self,
        docs: list[str],
        triples_per_doc: list[list[tuple[str, str, str]]],
    ) -> None:
        """Index docs with known triples, bypassing LLM-based OpenIE.

        Pre-writes the OpenIE cache file with the supplied entities/triples so
        that ``hipporag.index()`` finds all chunks already processed and skips
        the expensive LLM extraction step.
        """
        import json
        from baselines.HippoRAG.src.hipporag.utils.misc_utils import compute_mdhash_id  # type: ignore

        cache_path = self.hipporag.openie_results_path

        # Load existing cache entries (accumulated from prior ingest calls)
        existing: list[dict] = []
        if os.path.isfile(cache_path):
            try:
                with open(cache_path) as f:
                    existing = json.load(f).get('docs', [])
            except (json.JSONDecodeError, IOError):
                existing = []
        existing_ids = {e.get('idx') or compute_mdhash_id(e['passage'], 'chunk-')
                        for e in existing}

        # Build new OpenIE entries from known triples
        for doc, triples in zip(docs, triples_per_doc):
            chunk_id = compute_mdhash_id(doc, 'chunk-')
            if chunk_id in existing_ids:
                continue
            entities: set[str] = set()
            for h, _r, t in triples:
                entities.add(h)
                entities.add(t)
            existing.append({
                'idx': chunk_id,
                'passage': doc,
                'extracted_entities': sorted(entities),
                'extracted_triples': [list(tri) for tri in triples],
            })

        # Write updated cache
        all_ents = [e for chunk in existing for e in chunk['extracted_entities']]
        num = len(all_ents) or 1
        avg_chars = round(sum(len(e) for e in all_ents) / num, 4)
        avg_words = round(sum(len(e.split()) for e in all_ents) / num, 4)
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({
                'docs': existing,
                'avg_ent_chars': avg_chars,
                'avg_ent_words': avg_words,
            }, f)

        # Call index() with cache — LLM OpenIE is skipped
        self.hipporag.global_config.force_openie_from_scratch = False
        self.hipporag.index(docs=docs)

    def _reset_hipporag(self, sample_id: str | None = None):
        self.hipporag = self._init_hipporag(sample_id=sample_id)


class Mem0BackendBase:
    """
    Base helpers for Mem0-backed implementations.
    """

    def __init__(
        self,
        mem0_config=None,
        *,
        mem0_client=None,
        mem0_user_id: str = 'mem0',
        usage_interval_env: str = 'BENCHMARK_USAGE_INTERVAL',
        mem0_dir: str | None = None,
    ):
        self._episodes_processed = 0
        self.usage_log_interval = int(os.getenv(usage_interval_env, '0'))
        self.mem_user_id = mem0_user_id
        if mem0_dir:
            os.environ['MEM0_DIR'] = mem0_dir
        if mem0_client is not None:
            self.memory = mem0_client
        else:
            self.memory = self._init_mem0(mem0_config)
            # Log vector store path and collection for debugging
            try:
                logger.info(
                    'Mem0 initialized with vector_store path=%s collection=%s',
                    getattr(self.memory.vector_store, 'path', ''),
                    getattr(self.memory.vector_store, 'collection_name', ''),
                )
            except Exception:
                pass

    def _init_mem0(self, mem0_config):
        mem0_root = Path(__file__).resolve().parents[3] / 'baselines' / 'mem0'
        mem0_path = str(mem0_root)
        if mem0_path not in sys.path:
            sys.path.insert(0, mem0_path)
        from mem0 import Memory  # type: ignore

        return Memory(config=mem0_config)

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step
        if (
            self.usage_log_interval > 0
            and self._episodes_processed % self.usage_log_interval == 0
        ):
            self.log_usage_summary(prefix=f'episodes={self._episodes_processed}')

    def log_usage_summary(self, prefix: str = '') -> None:
        summary = usage_tracker.summary()
        per_model = summary.get('per_model', {})
        totals = summary.get('totals', {})
        if prefix == 'final' or (
            self.usage_log_interval > 0 and self._episodes_processed % self.usage_log_interval == 0
        ):
            for model, stats in per_model.items():
                logger.info(
                    '[LLM][usage-summary] %s model=%s prompt=%s completion=%s cost=%.6f',
                    prefix,
                    model,
                    stats.get('prompt_tokens', 0),
                    stats.get('completion_tokens', 0),
                    stats.get('cost', 0.0),
                )
            logger.info(
                '[LLM][usage-summary] %s totals prompt=%s completion=%s cost=%.6f',
                prefix,
                totals.get('prompt_tokens', 0),
                totals.get('completion_tokens', 0),
                totals.get('cost', 0.0),
            )

    def finalize_usage_logging(self) -> None:
        if self._episodes_processed:
            self.log_usage_summary(prefix='final')
