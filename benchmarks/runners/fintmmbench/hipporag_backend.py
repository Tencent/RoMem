"""HippoRAG backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Iterable

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import AnswerResponse, initialize_answer_llm
from benchmarks.runners.base_backend import HippoRAGBackendBase
from benchmarks.types import FinTMMBenchExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class FinTMMBenchHippoRAGBackend(HippoRAGBackendBase):
    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        hippo_save_dir: str | None = None,
        hippo_llm_model: str | None = None,
        hippo_embedding_model: str | None = None,
        hippo_openie_mode: str | None = None,
        exp_name: str | None = None,
    ):
        super().__init__(
            hippo_save_dir=hippo_save_dir,
            hippo_llm_model=hippo_llm_model,
            hippo_embedding_model=hippo_embedding_model,
            hippo_openie_mode=hippo_openie_mode,
        )
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
        self._doc_uuid_to_text: dict[str, str] = {}
        self._text_to_uuid: dict[str, str] = {}

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the entire FinTMMBench corpus into HippoRAG (one-time)."""
        from benchmarks.loaders.fintmmbench import _doc_to_text, _parse_date

        self._init_hipporag(sample_id='fintmmbench_corpus')

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

        for date_str in sorted(by_date.keys()):
            docs = [text for _, text in by_date[date_str]]
            self.hipporag.index(docs=docs)
            self.increment_episode_count(len(docs))

        logger.info('Corpus ingestion complete (%d documents)', total_docs)

    async def evaluate_example(self, example: FinTMMBenchExample) -> tuple[dict, dict, dict]:
        retrieval_metrics, context, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(self, example: FinTMMBenchExample) -> tuple[dict, list[str], dict]:
        try:
            results = self.hipporag.retrieve(
                [example.question],
                num_to_retrieve=self.search_limit,
            )
            docs = results[0].docs if results else []
        except Exception as exc:
            logger.warning('HippoRAG retrieve failed for %s: %s', example.uuid, exc)
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
        if client is None:
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


