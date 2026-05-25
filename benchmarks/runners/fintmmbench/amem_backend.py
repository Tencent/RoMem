"""A-mem-sys backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path

from openai import AsyncOpenAI

from benchmarks.evaluators.llm_judge import score_answer
from benchmarks.types import FinTMMBenchExample

logger = logging.getLogger(__name__)

# Ensure baselines/A-mem-sys is on sys.path so we can import agentic_memory
_AMEM_ROOT = str(Path(__file__).resolve().parents[3] / 'baselines' / 'A-mem-sys')
if _AMEM_ROOT not in sys.path:
    sys.path.insert(0, _AMEM_ROOT)


class FinTMMBenchAMemBackend:
    """FinTMMBench backend that uses A-mem-sys (ChromaDB + embeddings)."""

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
        self.search_limit = search_limit or int(os.getenv('FINTMMBENCH_SEARCH_LIMIT', '50'))
        self._episodes_processed = 0

        ctx_values: list[int] = []
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        else:
            ctx_env = os.getenv('FINTMMBENCH_CONTEXT_K')
            if ctx_env:
                ctx_values = [
                    int(v.strip())
                    for v in ctx_env.split(',')
                    if v.strip().isdigit()
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

        self._base_dir = os.getenv('AMEM_BASE_DIR') or str(
            Path('outputs/amem_fintmmbench').resolve()
        )
        self._memory = None
        # Maps doc UUID to position in ingestion order (for retrieval recall)
        self._uid_to_text: dict[str, str] = {}

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

    def _uid_map_path(self, cache_dir: str) -> str:
        return os.path.join(cache_dir, 'uid_to_note.pkl')

    def _save_cache(self, cache_dir: str, uid_to_note: dict[str, str]) -> None:
        """Persist memories and UUID-to-note-id mapping to disk."""
        os.makedirs(cache_dir, exist_ok=True)
        with open(self._cache_path(cache_dir), 'wb') as f:
            pickle.dump(self._memory.memories, f)
        with open(self._uid_map_path(cache_dir), 'wb') as f:
            pickle.dump(uid_to_note, f)

    def _load_cache(self, cache_dir: str) -> dict[str, str] | None:
        """Load memories and mapping from disk. Returns uid_to_note or None."""
        mem_path = self._cache_path(cache_dir)
        map_path = self._uid_map_path(cache_dir)
        if not os.path.isfile(mem_path) or not os.path.isfile(map_path):
            return None
        with open(mem_path, 'rb') as f:
            self._memory.memories = pickle.load(f)
        with open(map_path, 'rb') as f:
            uid_to_note: dict[str, str] = pickle.load(f)
        return uid_to_note

    def increment_episode_count(self, step: int = 1) -> None:
        self._episodes_processed += step

    def log_usage_summary(self, prefix: str = '') -> None:
        pass

    def finalize_usage_logging(self) -> None:
        pass

    # ── Ingestion ────────────────────────────────────────────────────

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the entire FinTMMBench corpus into A-mem. Uses disk cache if available."""
        from benchmarks.loaders.fintmmbench import _doc_to_text  # noqa: PLC0415

        if not corpus:
            return

        cache_dir = self._base_dir
        self._memory = self._create_memory(persist_directory=cache_dir)

        cached = self._load_cache(cache_dir)
        if cached is not None:
            self._uid_to_note_id = cached
            print(
                f'  Loaded cached A-mem for FinTMMBench ({len(self._memory.memories)} memories)',
                flush=True,
            )
            return

        # Build ordered list of (uid, text, date)
        by_date: dict[str, list[tuple[str, str]]] = {}
        for uid, doc in corpus.items():
            text = _doc_to_text(doc)
            if not text:
                continue
            date_str = doc.get('Date', '2022-01-01')
            by_date.setdefault(date_str, []).append((uid, text))

        total_docs = sum(len(v) for v in by_date.values())
        print(f'  Ingesting {total_docs} corpus documents into A-mem...', flush=True)

        uid_to_note: dict[str, str] = {}
        doc_count = 0
        for date_str in sorted(by_date.keys()):
            for uid, text in by_date[date_str]:
                try:
                    note_id = self._memory.add_note(text)
                    uid_to_note[uid] = note_id if note_id else uid
                    self.increment_episode_count()
                except Exception as exc:
                    logger.warning('A-mem add_note failed for %s: %s', uid, exc)
                    uid_to_note[uid] = uid
                doc_count += 1
                if doc_count % 50 == 0 or doc_count == total_docs:
                    print(f'    Doc {doc_count}/{total_docs} done', flush=True)

        self._uid_to_note_id = uid_to_note
        self._save_cache(cache_dir, uid_to_note)
        print(f'  Ingested {total_docs} corpus documents into A-mem.', flush=True)

    # ── Evaluation ──────────────────────────────────────────────────

    async def evaluate_example(
        self, example: FinTMMBenchExample
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        retrieval_metrics, context_docs, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context_docs)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(
        self, example: FinTMMBenchExample
    ) -> tuple[dict[str, float], list[str], dict]:
        raw_results: list[dict] = []
        if self._memory is not None:
            try:
                results = self._memory.search(example.question, k=self.search_limit)
                raw_results = [item for item in results if isinstance(item, dict)]
            except Exception as exc:
                logger.warning('A-mem search failed for %s: %s', example.uuid, exc)

        docs: list[str] = [item.get('content', '') or '' for item in raw_results]

        # Compute retrieval metrics: check if gold source_ids appear in retrieved notes
        gold_ids = set(example.source_ids)
        RECALL_KS = [1, 3, 5, 10]

        # Build a reverse mapping: note_id -> uid
        uid_to_note = getattr(self, '_uid_to_note_id', {})
        note_to_uid: dict[str, str] = {v: k for k, v in uid_to_note.items()}

        matched_uids: list[str | None] = []
        first_match_rank: int | None = None
        for idx, item in enumerate(raw_results):
            note_id = item.get('id', '')
            uid = note_to_uid.get(note_id, '')
            if uid and uid in gold_ids:
                matched_uids.append(uid)
                if first_match_rank is None:
                    first_match_rank = idx + 1
            else:
                matched_uids.append(None)

        mrr = 1.0 / first_match_rank if first_match_rank else 0.0

        metrics: dict[str, float] = {
            'retrieval_mrr': mrr,
            'retrieval_results_returned': float(len(docs)),
            'retrieval_gold_sources': float(len(gold_ids)),
        }
        for k in RECALL_KS:
            found = {u for u in matched_uids[:k] if u is not None}
            metrics[f'retrieval_recall@{k}'] = len(found) / len(gold_ids) if gold_ids else 0.0

        details = {'top_doc': docs[0] if docs else ''}
        return metrics, docs, details

    async def _answer_with_llm(
        self, example: FinTMMBenchExample, context_docs: list[str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None or not context_docs:
            return metrics, answers

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
            user_prompt = (
                f'Question: {example.question}\n\n'
                f'Financial data:\n'
                + os.linesep.join(f'- {doc}' for doc in subset)
                + '\n\nProvide a short answer grounded in the data. '
                'Respond with JSON as {"answer": "<text>"}.'
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
                    generated_answer = parsed.get('answer', raw)
                except json.JSONDecodeError:
                    generated_answer = raw.strip()
            except Exception as exc:
                logger.warning(
                    'LLM answer failed for %s (top-%d): %s', example.uuid, k, exc
                )

            answers[k] = generated_answer
            is_correct, _ = await score_answer(
                example.question, example.answer, generated_answer
            )
            metrics[f'llm@{k}_accuracy'] = 1.0 if is_correct else 0.0

        return metrics, answers
