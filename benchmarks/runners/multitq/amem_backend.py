"""A-mem-sys backend for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

from openai import AsyncOpenAI

from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.types import MultiTQQuestion, TemporalTriple

logger = logging.getLogger(__name__)

# Ensure baselines/A-mem-sys is on sys.path so we can import agentic_memory
_AMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'A-mem-sys')
if _AMEM_ROOT not in sys.path:
    sys.path.insert(0, _AMEM_ROOT)


def _wrap_encode_no_progress(original_encode):
    """Wrap SentenceTransformer.encode to force show_progress_bar=False."""
    import functools

    @functools.wraps(original_encode)
    def _encode(self, *args, **kwargs):
        kwargs.setdefault('show_progress_bar', False)
        return original_encode(self, *args, **kwargs)
    return _encode


class MultiTQAMemBackend:
    """MultiTQ backend that uses A-mem-sys (ChromaDB + embeddings)."""

    def __init__(
        self,
        model_name: str | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
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
        self.llm_backend = llm_backend or os.getenv('AMEM_LLM_BACKEND') or 'openai'
        self.llm_base_url = os.getenv('AMEM_LLM_BASE_URL') or None
        self.llm_model = (
            llm_model
            or os.getenv('AMEM_LLM_MODEL')
            or os.getenv('OPENAI_MODEL')
            or 'gpt-4o-mini'
        )
        self.search_limit = search_limit or int(os.getenv('MULTITQ_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

        ctx_values: list[int] = []
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        else:
            ctx_env = os.getenv('MULTITQ_CONTEXT_K')
            if ctx_env:
                ctx_values = [
                    int(v.strip())
                    for v in ctx_env.split(',')
                    if v.strip().isdigit()
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

        self._base_dir = os.getenv('AMEM_BASE_DIR') or str(
            Path('outputs/amem_multitq').resolve()
        )
        self._memory = None

    # ── Helpers ─────────────────────────────────────────────────────

    def _create_memory(self, persist_directory: str | None = None):
        """Create an AgenticMemorySystem instance."""
        from agentic_memory.memory_system import AgenticMemorySystem  # type: ignore

        return AgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            embedding_provider=self.embedding_provider,
            persist_directory=persist_directory,
            llm_base_url=self.llm_base_url,
        )

    def _cache_path(self, cache_dir: str) -> str:
        return os.path.join(cache_dir, 'memories.pkl')

    def _save_cache(self, cache_dir: str) -> None:
        """Persist in-memory memories dict to disk."""
        os.makedirs(cache_dir, exist_ok=True)
        with open(self._cache_path(cache_dir), 'wb') as f:
            pickle.dump(self._memory.memories, f)

    def _load_cache(self, cache_dir: str) -> bool:
        """Load memories from disk cache. Returns True if cache was found."""
        mem_path = self._cache_path(cache_dir)
        if not os.path.isfile(mem_path):
            return False
        with open(mem_path, 'rb') as f:
            self._memory.memories = pickle.load(f)
        return True

    @staticmethod
    def _triple_to_sentence(triple: TemporalTriple) -> str:
        if triple.timestamp is not None:
            date_str = triple.timestamp.strftime('%Y-%m-%d')
        else:
            date_str = f'time_{triple.time_id}'
        return f'On {date_str}, {triple.head} {triple.relation} {triple.tail}.'

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    # ── Ingestion ────────────────────────────────────────────────────

    async def ingest_all_triples(self, triples: Iterable[TemporalTriple]) -> None:
        """Ingest all KG triples into A-mem. Uses disk cache if available."""
        all_triples = list(triples)
        if not all_triples:
            return

        cache_dir = self._base_dir
        self._memory = self._create_memory(persist_directory=cache_dir)

        if self._load_cache(cache_dir):
            print(
                f'  Loaded cached A-mem for MultiTQ ({len(self._memory.memories)} memories)',
                flush=True,
            )
            return

        n = len(all_triples)
        # Suppress per-triple tqdm batch progress bars from sentence_transformers/chromadb
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('chromadb').setLevel(logging.WARNING)
        try:
            import sentence_transformers
            sentence_transformers.SentenceTransformer.encode = _wrap_encode_no_progress(
                sentence_transformers.SentenceTransformer.encode
            )
        except Exception:
            pass

        log_interval = max(1, n // 20)  # log ~20 times
        t0 = time.time()
        logger.info('Ingesting %d triples into A-mem...', n)
        for i, triple in enumerate(all_triples, 1):
            text = self._triple_to_sentence(triple)
            try:
                self._memory.add_note(text)
                self.increment_episode_count()
            except Exception as exc:
                logger.warning('A-mem add_note failed for triple %s: %s', i, exc)
            if i % log_interval == 0 or i == n:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta_s = (n - i) / rate if rate > 0 else 0
                eta_h, eta_rem = divmod(int(eta_s), 3600)
                eta_m = eta_rem // 60
                logger.info(
                    '  [%d/%d] %.1f%% | %.1f triples/s | elapsed %.0fs | ETA %dh %02dm',
                    i, n, 100 * i / n, rate, elapsed, eta_h, eta_m,
                )

        self._save_cache(cache_dir)
        print(f'  Ingested {n} triples into A-mem.', flush=True)

    # ── Evaluation ──────────────────────────────────────────────────

    async def evaluate_question(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        retrieval_metrics, context_docs, details = await self._retrieve(question)
        llm_metrics, answers = await self._answer(question, context_docs)
        accuracy_metrics, match_strategies = self._verify_answers(question, answers)
        return (
            retrieval_metrics,
            {**llm_metrics, **accuracy_metrics},
            {**details, 'llm_answers': answers, 'context_docs': context_docs,
             'match_strategies': match_strategies},
        )

    async def _retrieve(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], list[str], dict]:
        docs: list[str] = []
        if self._memory is not None:
            try:
                results = self._memory.search(question.question, k=self.search_limit)
                for item in results:
                    content = item.get('content', '') or ''
                    docs.append(content.strip())
            except Exception as exc:
                logger.warning(
                    'A-mem search failed for quid=%s: %s', question.quid, exc
                )

        metrics: dict[str, float] = {}
        matching_fact = ''

        # answer_in_context@k
        for k in self.answer_context_sizes:
            subset = docs[:k]
            found = self._check_answer_in_context(question.answers, subset)
            metrics[f'answer_in_context@{k}'] = 1.0 if found else 0.0
            if found and not matching_fact:
                matching_fact = self._find_matching_fact(question.answers, subset)

        # Hits@k and MRR
        match_rank: int | None = None
        for idx, doc in enumerate(docs, start=1):
            if not doc:
                continue
            doc_lower = doc.lower().replace('_', ' ')
            for answer in question.answers:
                if answer.lower().replace('_', ' ') in doc_lower:
                    match_rank = idx
                    break
            if match_rank is not None:
                break

        metrics['hits@1'] = 1.0 if match_rank == 1 else 0.0
        metrics['hits@3'] = 1.0 if match_rank is not None and match_rank <= 3 else 0.0
        metrics['hits@10'] = 1.0 if match_rank is not None and match_rank <= 10 else 0.0
        metrics['mrr'] = 1.0 / match_rank if match_rank else 0.0

        return metrics, docs, {'matching_fact': matching_fact}

    async def _answer(
        self, question: MultiTQQuestion, context_docs: list[str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None:
            return metrics, answers

        for k in self.answer_context_sizes:
            subset = [d for d in context_docs[:k] if d]
            if not subset:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            facts = [f'- {doc}' for doc in subset]
            prompt = (
                f'Question: {question.question}\n\n'
                f'Supporting facts:\n{os.linesep.join(facts)}\n\n'
                'Provide a concise answer grounded in the facts. '
                'Respond with JSON {"answer": "<text>"}.'
            )
            messages = [
                {'role': 'system', 'content': 'You answer questions using only provided facts.'},
                {'role': 'user', 'content': prompt},
            ]
            prediction = ''
            try:
                response = await client.chat.completions.create(
                    model=self.answer_llm_model,
                    messages=messages,
                )
                raw = response.choices[0].message.content or ''
                try:
                    parsed = json.loads(raw)
                    prediction = parsed.get('answer', raw)
                except json.JSONDecodeError:
                    prediction = raw.strip()
            except Exception as exc:
                logger.warning(
                    'LLM answer failed for quid=%s (top-%s): %s', question.quid, k, exc
                )
            answers[k] = prediction
            metrics[f'llm@{k}_context_size'] = float(len(facts))

        return metrics, answers

    def _verify_answers(
        self, question: MultiTQQuestion, answers: dict[int, str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        strategies: dict[int, str] = {}
        for k, prediction in answers.items():
            is_correct, strategy = verify_answer(
                prediction, question.answers, question.answer_type,
            )
            metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0
            strategies[k] = strategy
        return metrics, strategies

    # ── Static helpers ───────────────────────────────────────────────

    @staticmethod
    def _check_answer_in_context(answers: list[str], docs: list[str]) -> bool:
        context = ' '.join(docs).lower().replace('_', ' ')
        for answer in answers:
            if answer.lower().replace('_', ' ') in context:
                return True
        return False

    @staticmethod
    def _find_matching_fact(answers: list[str], docs: list[str]) -> str:
        for doc in docs:
            doc_lower = doc.lower().replace('_', ' ')
            for answer in answers:
                if answer.lower().replace('_', ' ') in doc_lower:
                    return doc
        return ''
