"""Graphiti backend for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Iterable

from tqdm import tqdm

from baselines.graphiti.graphiti_core.nodes import EpisodeType
from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from benchmarks.runners.shared_utils import AnswerResponse
from benchmarks.runners.base_backend import GraphitiBackendBase
from benchmarks.types import FinTMMBenchExample
from benchmarks.evaluators.llm_judge import score_answer

logger = logging.getLogger(__name__)


class FinTMMBenchGraphitiBackend(GraphitiBackendBase):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        search_reranker: str | None = None,
    ):
        super().__init__(
            uri,
            user,
            password,
            database=database,
            usage_interval_env='FINTMMBENCH_USAGE_INTERVAL',
            search_reranker=search_reranker,
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
        self.answer_llm_client = self._init_answer_llm_client(
            answer_llm_model or os.getenv('FINTMMBENCH_ANSWER_LLM_MODEL'),
            answer_llm_api_key or os.getenv('FINTMMBENCH_ANSWER_LLM_API_KEY'),
            answer_llm_base_url or os.getenv('FINTMMBENCH_ANSWER_LLM_BASE_URL'),
        )
        self._episode_to_doc_uid: dict[str, str] = {}

    async def ingest_corpus(self, corpus: dict[str, dict]) -> None:
        """Ingest the entire FinTMMBench corpus into Graphiti (one-time)."""
        from benchmarks.loaders.fintmmbench import _doc_to_text, _parse_date

        by_date: dict[str, list[tuple[str, str, dict]]] = {}
        for uid, doc in corpus.items():
            text = _doc_to_text(doc)
            if not text:
                continue
            date_str = doc.get('Date', '2022-01-01')
            by_date.setdefault(date_str, []).append((uid, text, doc))

        total_docs = sum(len(v) for v in by_date.values())
        logger.info('Ingesting %d corpus documents across %d dates', total_docs, len(by_date))

        progress = tqdm(sorted(by_date.keys()), desc='Ingesting corpus dates', leave=False)
        doc_count = 0
        for date_str in progress:
            items = by_date[date_str]
            dt = _parse_date(date_str)
            ref_time = dt.replace(tzinfo=timezone.utc)
            for uid, text, doc in items:
                source_desc = doc.get('type', 'financial_data')
                try:
                    result = await self.graphiti.add_episode(
                        name=f'fintmmbench-{uid}',
                        episode_body=text,
                        source_description=source_desc,
                        reference_time=ref_time,
                        source=EpisodeType.text,
                    )
                    if result and result.episode:
                        self._episode_to_doc_uid[result.episode.uuid] = uid
                except Exception as exc:
                    logger.warning(
                        'add_episode failed for %s, skipping: %s', uid, exc,
                    )
                self.increment_episode_count()
                doc_count += 1
                progress.set_postfix(docs=doc_count)
        progress.close()

        logger.info('Corpus ingestion complete (%d documents)', total_docs)

    async def evaluate_example(self, example: FinTMMBenchExample) -> tuple[dict, dict, dict]:
        retrieval_metrics, context, retrieval_details = await self._retrieve(example)
        llm_metrics, llm_answers = await self._answer_with_llm(example, context)
        details = {**retrieval_details, 'llm_answers': llm_answers}
        return retrieval_metrics, llm_metrics, details

    async def _retrieve(self, example: FinTMMBenchExample) -> tuple[dict, list[str], dict]:
        try:
            results = await self._search_edges(example.question, self.search_limit)
        except Exception as exc:
            logger.warning('Graphiti search failed for %s: %s', example.uuid, exc)
            results = []

        docs: list[str] = []
        for edge in results:
            fact = (getattr(edge, 'fact', '') or '').strip()
            if fact:
                docs.append(fact)

        # Compute retrieval metrics via episode→doc UUID mapping
        gold_ids = set(example.source_ids)
        RECALL_KS = [1, 3, 5, 10]

        matched_uids: list[str | None] = [None] * len(results)
        first_match_rank = None
        for idx, edge in enumerate(results):
            episode_uuids = getattr(edge, 'episodes', []) or []
            for ep_uuid in episode_uuids:
                doc_uid = self._episode_to_doc_uid.get(ep_uuid)
                if doc_uid and doc_uid in gold_ids:
                    matched_uids[idx] = doc_uid
                    if first_match_rank is None:
                        first_match_rank = idx + 1
                    break

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
        client = self.answer_llm_client or self.graphiti.llm_client

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

    def _init_answer_llm_client(self, model: str | None, api_key: str | None, base_url: str | None):
        if not any([model, api_key, base_url]):
            return self.graphiti.llm_client
        config = LLMConfig(api_key=api_key, model=model, base_url=base_url)
        model_name = (config.model or '').lower()
        if not model_name or model_name.startswith('gpt'):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)

