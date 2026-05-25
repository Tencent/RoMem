"""LiCoMemory backend implementation for the DMR-MSC benchmark."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI

from benchmarks.runners.shared_utils import normalize_text, token_overlap
from benchmarks.types import DmrMscExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)

# Add baselines/LiCoMemory to sys.path so LiCoMemory modules can be imported
_LICOMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'LiCoMemory')
if _LICOMEM_ROOT not in sys.path:
    sys.path.insert(0, _LICOMEM_ROOT)


class DmrMscLiCoMemBackend:
    """DMR-MSC backend that uses LiCoMemory (Cognigraph) for memory storage and retrieval."""

    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        # LiCoMemory LLM config
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
        self.search_limit = search_limit or int(os.getenv('DMR_MSC_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

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

        self.answer_llm_model = (
            answer_llm_model
            or os.getenv('DMR_MSC_ANSWER_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-5-mini'
        )
        llm_api_key = (
            answer_llm_api_key
            or os.getenv('DMR_MSC_ANSWER_LLM_API_KEY')
            or os.getenv('OPENAI_API_KEY')
        )
        llm_base_url = (
            answer_llm_base_url
            or os.getenv('DMR_MSC_ANSWER_LLM_BASE_URL')
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
            or str(Path('outputs/licomem_dmr_msc').resolve())
        )

    def _build_config(self, working_dir: str):
        """Build LiCoMemory Config programmatically (no YAML file needed)."""
        from init.config import (
            Config, LLMConfig, QueryLLMConfig, EmbeddingConfig,
            ChunkConfig, GraphConfig, RetrieverConfig, QueryConfig,
            StorageConfig, EvaluationConfig,
        )
        config = Config()
        config.index_name = 'dmr_msc_graph'
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
            enable_sessiontime=False,
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

    @staticmethod
    def _format_episode_for_licomem(episode) -> str:
        """Format an episode for LiCoMemory's DialogChunkProcessor.

        Dialogue episodes (type=dialog) are converted to the LOCOMO format
        ("Speaker 1": "text" "Speaker 2": "text") so the dialogue-aware
        chunker and extractor can parse proper turns.
        Non-dialogue episodes are wrapped as "Narrator": "text".
        """
        import re
        ep_type = episode.metadata.get('type', '')
        content = episode.content

        if ep_type == 'dialog':
            # Convert "Speaker 1: text\nSpeaker 2: text" to
            # "Speaker 1": "text" "Speaker 2": "text"
            parts = []
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                m = re.match(r'^(Speaker \d+|[^:]+):\s*(.+)$', line)
                if m:
                    speaker = m.group(1).strip()
                    text = m.group(2).strip().replace('"', '\\"')
                    parts.append(f'"{speaker}": "{text}"')
                else:
                    text = line.replace('"', '\\"')
                    parts.append(f'"Narrator": "{text}"')
            return ' '.join(parts) if parts else f'"Narrator": "{content}"'
        else:
            escaped = content.replace('"', '\\"')
            return f'"Narrator": "{escaped}"'

    async def ingest_example(self, example: DmrMscExample) -> object:
        """Create a new GraphRAG instance and insert all episodes as a corpus."""
        from init.graph_rag import GraphRAG

        # Use a per-example temp directory to keep graphs isolated
        working_dir = str(
            Path(self._licomem_base_dir) / f'example_{example.example_id}'
        )
        os.makedirs(working_dir, exist_ok=True)

        config = self._build_config(working_dir)
        graph_rag = GraphRAG(config, base_dir=working_dir)

        # Convert episodes to LiCoMemory corpus format
        corpus = []
        for idx, episode in enumerate(example.episodes):
            session_time = ''
            if episode.reference_time:
                try:
                    session_time = episode.reference_time.strftime('%Y-%m-%d')
                except Exception:
                    session_time = str(episode.reference_time)
            context = self._format_episode_for_licomem(episode)
            corpus.append({
                'session_id': episode.metadata.get('session_id', f'session_{idx}'),
                'context': context,
                'content': episode.content,
                'session_time': session_time,
            })
            self.increment_episode_count()

        try:
            await graph_rag.insert(corpus)
            logger.debug(
                'LiCoMemory inserted %s episodes for example %s',
                len(corpus), example.example_id,
            )
        except Exception as exc:
            logger.warning(
                'LiCoMemory insert failed for %s: %s',
                example.example_id, exc,
            )

        return graph_rag

    async def cleanup_example(self, example_id: str) -> None:
        """Remove the per-example working directory."""
        import shutil
        working_dir = str(
            Path(self._licomem_base_dir) / f'example_{example_id}'
        )
        try:
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir, ignore_errors=True)
                logger.debug('Cleaned up LiCoMemory working dir: %s', working_dir)
        except Exception as exc:
            logger.warning('LiCoMemory cleanup failed for %s: %s', example_id, exc)

    async def evaluate_example(self, example: DmrMscExample) -> tuple[dict, dict, dict]:
        graph_rag = await self.ingest_example(example)
        retrieval_metrics, context, retrieval_details = await self._retrieve_answer_support(
            example, graph_rag
        )
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        await self.cleanup_example(example.example_id)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve_answer_support(
        self, example: DmrMscExample, graph_rag: object
    ) -> tuple[dict, list[str], dict]:
        """Query LiCoMemory and extract triples/facts for retrieval metrics."""
        facts: list[str] = []
        try:
            result = await graph_rag.query(example.question, question_time='2024/01/01 (Mon) 12:00')
            if isinstance(result, dict):
                # Extract triples — these are dicts with 'src', 'relation', 'tgt'
                triples = result.get('triples', [])
                for triple in triples:
                    if isinstance(triple, dict):
                        src = triple.get('src', '')
                        rel = triple.get('relation', '')
                        tgt = triple.get('tgt', '')
                        fact = f'{src} {rel} {tgt}'.strip()
                        if fact:
                            facts.append(fact)

                # Also include chunks as fallback context (for LLM answering)
                chunks = result.get('chunks', [])
                if isinstance(chunks, list):
                    for chunk in chunks:
                        if isinstance(chunk, str) and chunk.strip():
                            if chunk not in facts:
                                facts.append(chunk)
                        elif isinstance(chunk, dict):
                            text = chunk.get('content') or chunk.get('text') or ''
                            if text.strip() and text not in facts:
                                facts.append(text)
            elif isinstance(result, str) and result.strip():
                facts.append(result)
        except Exception as exc:
            logger.warning(
                'LiCoMemory query failed for %s: %s',
                example.example_id, exc,
            )

        normalized_answer = normalize_text(example.answer)
        match_rank: int | None = None
        matching_fact = ''
        top_fact = ''
        for idx, fact in enumerate(facts, start=1):
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
            'retrieval_results_returned': float(len(facts)),
        }
        details = {
            'matching_fact': matching_fact,
            'top_fact': top_fact,
        }
        return retrieval_metrics, facts, details

    async def _answer_with_llm(
        self, example: DmrMscExample, context_docs: Iterable[str]
    ) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        if not self.answer_context_sizes:
            return metrics, answers
        doc_list = list(context_docs)
        if not doc_list:
            return metrics, answers
        client = self.answer_llm_client
        if not client:
            return metrics, answers

        for k in self.answer_context_sizes:
            subset = [doc for doc in doc_list[:k] if doc]
            if not subset:
                metrics[f'llm@{k}_exact'] = 0.0
                metrics[f'llm@{k}_f1'] = 0.0
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            system_prompt = (
                'You are a careful assistant. Use only the provided facts to answer the question.'
            )
            user_prompt = f"""Question: {example.question}

Supporting facts:
{os.linesep.join(f'- {doc}' for doc in subset)}

Provide a short answer grounded in the facts. Respond with JSON as {{"answer": "<text>"}}."""
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
                    generated_answer = parsed.get('answer', raw)
                except json.JSONDecodeError:
                    generated_answer = raw.strip()
            except Exception as exc:
                logger.warning(
                    'LiCoMemory answer failed for %s (top-%s context): %s',
                    example.example_id, k, exc,
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
