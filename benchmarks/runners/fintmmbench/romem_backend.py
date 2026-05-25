"""RoMem backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import logging
import os
from datetime import timezone
from pathlib import Path
from typing import Iterable

from baselines.graphiti.graphiti_core.prompts.models import Message

from tqdm import tqdm

from benchmarks.runners.shared_utils import (
    AnswerResponse,
    initialize_answer_llm,
    parse_bool,
    parse_optional_bool,
)
from benchmarks.types import EpisodePayload, FinTMMBenchExample
from benchmarks.runners.romem_utils import apply_romem_config
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


def _safe_dir_name(value: str) -> str:
    import re as _re
    return _re.sub(r'[^A-Za-z0-9._-]+', '_', value or 'fintmmbench')


def _reference_iso(episode: EpisodePayload) -> str:
    ref = episode.reference_time
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    else:
        ref = ref.astimezone(timezone.utc)
    return ref.isoformat()


class FinTMMBenchRoMemBackend:
    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
        exp_name: str | None = None,
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
            api_key=answer_llm_api_key or os.getenv('OPENAI_API_KEY'),
            base_url=answer_llm_base_url or os.getenv('FINTMMBENCH_ANSWER_LLM_BASE_URL') or os.getenv('OPENAI_BASE_URL'),
        )
        self._romem_root = Path(
            romem_save_dir or os.getenv('ROMEM_SAVE_DIR') or 'outputs/romem_fintmmbench'
        )
        self._romem_llm_model = romem_llm_model or os.getenv('ROMEM_LLM_MODEL')
        self._romem_embedding_model = romem_embedding_model or os.getenv('ROMEM_EMBED_MODEL')
        self._romem_openie_mode = romem_openie_mode or os.getenv('ROMEM_OPENIE_MODE')
        self._romem_config = romem_config or {}
        self._romem_temporal_awareness = parse_bool(
            romem_temporal_awareness or os.getenv('ROMEM_TEMPORAL_AWARENESS'), default=True,
        )
        enable_tkge_value = romem_enable_tkge_tunnel
        if enable_tkge_value is None:
            enable_tkge_value = os.getenv('ROMEM_ENABLE_TKGE_TUNNEL')
        self._romem_enable_tkge_tunnel = parse_optional_bool(enable_tkge_value)
        self._romem_tkge_verbose = romem_tkge_verbose
        self._exp_name = exp_name or 'fintmmbench'
        self.romem = None
        self._doc_uuid_to_text: dict[str, str] = {}
        self._text_to_uuid: dict[str, str] = {}

    def _init_romem(self):
        from romem import RoMem
        from romem.utils.config_utils import BaseConfig

        run_dir = self._romem_root / 'run'
        config = BaseConfig()
        apply_romem_config(config, self._romem_config)
        config.save_dir = str(run_dir)
        if self._romem_llm_model:
            config.llm_name = self._romem_llm_model
        if self._romem_embedding_model:
            config.embedding_model_name = self._romem_embedding_model
        if self._romem_openie_mode:
            config.openie_mode = self._romem_openie_mode
        config.temporal_awareness = self._romem_temporal_awareness
        if self._romem_enable_tkge_tunnel is not None:
            config.enable_tkge_tunnel = self._romem_enable_tkge_tunnel
        if self._romem_tkge_verbose is not None:
            config.tkge_verbose = self._romem_tkge_verbose
        config.retrieval_top_k = self.search_limit

        self.romem = RoMem(
            global_config=config,
            save_dir=config.save_dir,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
        )

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the FinTMMBench corpus into RoMem (one-time)."""
        from benchmarks.loaders.fintmmbench import _doc_to_text, _parse_date

        if self.romem is None:
            self._init_romem()

        # Group documents by date for chronological ingestion
        by_date: dict[str, list[tuple[str, str]]] = {}
        for uid, doc in corpus.items():
            text = _doc_to_text(doc)
            if not text:
                continue
            self._doc_uuid_to_text[uid] = text
            self._text_to_uuid[text] = uid
            date_str = doc.get('Date', '2022-01-01')
            by_date.setdefault(date_str, []).append((uid, text))

        total_docs = sum(len(v) for v in by_date.values())
        logger.info('Ingesting %d corpus documents across %d dates', total_docs, len(by_date))

        # Flatten into chronological order
        ordered_docs: list[tuple[str, str]] = []
        for date_str in sorted(by_date.keys()):
            for _, text in by_date[date_str]:
                ordered_docs.append((date_str, text))

        # Defer TKGE training: accumulate triples during ingestion but skip
        # per-doc training.  Train once after all docs are indexed.
        tkge_ret = getattr(self.romem, 'tkge_retriever', None)
        tkge_enabled = getattr(self.romem, 'tkge_tunnel_enabled', False)
        _orig_update = None
        if tkge_enabled and tkge_ret is not None:
            _orig_update = tkge_ret.update
            def _update_no_train(triples, *, train=True):
                _orig_update(triples, train=False)
            tkge_ret.update = _update_no_train
            logger.info('TKGE training deferred (accumulating triples only)')

        for date_str, text in tqdm(ordered_docs, desc='Ingesting corpus', unit='doc'):
            dt = _parse_date(date_str)
            obs_time = dt.replace(tzinfo=timezone.utc).isoformat()
            self.romem.index(docs=[text], observed_time=obs_time)

        # Restore original update and train once on complete graph
        if _orig_update is not None:
            tkge_ret.update = _orig_update
            logger.info('Training TKGE on complete graph (%d documents)...', total_docs)
            self.romem.train_tkge()
            logger.info('TKGE training complete')

        logger.info('Corpus ingestion complete (%d documents)', total_docs)

    async def evaluate_example(self, example: FinTMMBenchExample) -> tuple[dict, dict, dict]:
        retrieval_metrics, docs, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, docs)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(self, example: FinTMMBenchExample) -> tuple[dict, list[str], dict]:
        if self.romem is None:
            return self._empty_retrieval(), [], {}

        try:
            results = self.romem.retrieve(
                [example.question],
                num_to_retrieve=self.search_limit,
            )
            docs = results[0].docs if results else []
        except Exception as exc:
            logger.warning('RoMem retrieve failed for %s: %s', example.uuid, exc)
            docs = []

        # Compute retrieval metrics via exact text→UUID lookup
        gold_ids = set(example.source_ids)
        RECALL_KS = [1, 3, 5, 10]

        matched_uids: list[str | None] = [None] * len(docs)
        first_match_rank = None
        for idx, doc_str in enumerate(docs):
            uid = self._text_to_uuid.get(doc_str)
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
        details = {
            'top_doc': docs[0] if docs else '',
        }
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

    @staticmethod
    def _empty_retrieval() -> dict[str, float]:
        d: dict[str, float] = {
            'retrieval_mrr': 0.0,
            'retrieval_results_returned': 0.0,
            'retrieval_gold_sources': 0.0,
        }
        for k in [1, 3, 5, 10]:
            d[f'retrieval_recall@{k}'] = 0.0
        return d
