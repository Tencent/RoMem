"""LiCoMemory runner for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import logging
import os

from benchmarks.evaluators.multitq import MultiTQEvaluator
from benchmarks.loaders.multitq import MultiTQLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.multitq.licomem_backend import MultiTQLiCoMemBackend

logger = logging.getLogger(__name__)


class MultiTQLiCoMemRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None = None,
        eval_split: str = 'test',
        max_time_ids: int | None = None,
        max_examples: int | None = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        licomem_llm_model: str | None = None,
        licomem_llm_api_key: str | None = None,
        licomem_llm_base_url: str | None = None,
        licomem_embed_model: str | None = None,
        licomem_embed_api_key: str | None = None,
        licomem_embed_api_type: str | None = None,
        licomem_embed_dimensions: int | None = None,
        licomem_base_dir: str | None = None,
    ):
        # Suppress verbose LiCoMemory logs
        os.environ.setdefault('LICOMEM_QUIET', '1')

        super().__init__('multitq', exp_name)
        self.loader = MultiTQLoader(
            root=data_path,
            eval_split=eval_split,
            max_time_ids=max_time_ids,
            max_examples=max_examples,
        )
        self.backend = MultiTQLiCoMemBackend(
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
            licomem_llm_model=licomem_llm_model,
            licomem_llm_api_key=licomem_llm_api_key,
            licomem_llm_base_url=licomem_llm_base_url,
            licomem_embed_model=licomem_embed_model,
            licomem_embed_api_key=licomem_embed_api_key,
            licomem_embed_api_type=licomem_embed_api_type,
            licomem_embed_dimensions=licomem_embed_dimensions,
            licomem_base_dir=licomem_base_dir,
        )
        self.evaluator = MultiTQEvaluator('multitq', exp_name=exp_name)

    async def run(self):
        # Phase 1: Ingest all KG triples in a single batch
        all_triples = self.loader.iter_all_triples()
        if all_triples:
            print(
                f'[LiCoMemory/MultiTQ] Ingesting {len(all_triples)} triples...',
                flush=True,
            )
            await self.backend.ingest_all_triples(all_triples)
            print('[LiCoMemory/MultiTQ] All triples ingested.', flush=True)

        questions = list(self.loader.iter_questions())
        total = len(questions)
        print(f'[LiCoMemory/MultiTQ] Evaluating {total} questions...', flush=True)

        for idx, question in enumerate(questions, start=1):
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_question(
                question
            )
            self.evaluator.record(question, retrieval_metrics, llm_metrics, details=details)

            if idx % 50 == 0 or idx == total:
                hits1 = retrieval_metrics.get('hits@1', 0.0)
                mrr = retrieval_metrics.get('mrr', 0.0)
                print(
                    f'[LiCoMemory/MultiTQ] {idx}/{total} | '
                    f'hits@1={hits1:.3f} mrr={mrr:.3f}',
                    flush=True,
                )

        self.evaluator.log_summary()
