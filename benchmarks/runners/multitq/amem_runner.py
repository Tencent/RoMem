"""A-mem-sys runner for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import logging

from benchmarks.evaluators.multitq import MultiTQEvaluator
from benchmarks.loaders.multitq import MultiTQLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.multitq.amem_backend import MultiTQAMemBackend

logger = logging.getLogger(__name__)


class MultiTQAMemRunner(BenchmarkRunnerBase):
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
        model_name: str | None = None,
        llm_backend: str | None = None,
        llm_model: str | None = None,
    ):
        super().__init__('multitq', exp_name)
        self.loader = MultiTQLoader(
            root=data_path,
            eval_split=eval_split,
            max_time_ids=max_time_ids,
            max_examples=max_examples,
        )
        self.backend = MultiTQAMemBackend(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
        )
        self.evaluator = MultiTQEvaluator('multitq', exp_name=exp_name)

    async def run(self):
        # Phase 1: Ingest all KG triples in a single batch
        all_triples = self.loader.iter_all_triples()
        if all_triples:
            print(f'Ingesting {len(all_triples)} triples...', flush=True)
            await self.backend.ingest_all_triples(all_triples)
            print('All triples ingested.', flush=True)

        questions = list(self.loader.iter_questions())
        n = len(questions)
        print(f'Evaluating {n} questions...', flush=True)
        for idx, question in enumerate(questions, 1):
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_question(
                question
            )
            self.evaluator.record(question, retrieval_metrics, llm_metrics, details=details)
            if idx % 50 == 0 or idx == n:
                print(f'  Question {idx}/{n} done', flush=True)

        self.evaluator.log_summary()
