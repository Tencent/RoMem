"""A-mem-sys runner for the LoCoMo dataset."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from benchmarks.evaluators.locomo import LocomoEvaluator
from benchmarks.loaders.locomo import LocomoLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.locomo.amem_backend import LocomoAMemBackend
from benchmarks.runners.locomo.utils import filter_sample_episodes

logger = logging.getLogger(__name__)


class LocomoAMemRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        llm_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        question_type: str | None = None,
        amem_model_name: str | None = None,
        amem_llm_backend: str | None = None,
        amem_llm_model: str | None = None,
    ):
        super().__init__('locomo', exp_name)
        self.loader = LocomoLoader(
            data_path=data_path,
            max_examples=max_examples,
            question_type=question_type,
        )
        self.backend = LocomoAMemBackend(
            model_name=amem_model_name,
            llm_backend=amem_llm_backend,
            llm_model=amem_llm_model,
            search_limit=search_limit,
            llm_context_sizes=llm_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
        )
        self.evaluator = LocomoEvaluator('locomo', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        if self.loader.is_mc:
            iterator = self.loader.iter_mc_examples()
            current_group: str | None = None
            for idx, example in enumerate(iterator, start=1):
                group_id = example.question_id.split('_q', 1)[0]
                if group_id != current_group:
                    if current_group is not None:
                        await self.backend.clear_graph()
                    await self.backend.ingest_example(example)
                    current_group = group_id
                prediction, context_ids = await self.backend.answer_mc_question(example)
                self.evaluator.record_mc(example, prediction, context_ids)
            await self.backend.clear_graph()
            self.backend.finalize_usage_logging()
            self.evaluator.log_summary()
            print(f'Total questions: {self.evaluator.mc_total}', flush=True)
            return

        total_samples = self._max_examples or self.loader.sample_count
        q_done = 0
        for idx, sample in enumerate(self.loader.iter_samples(), start=1):
            sample = filter_sample_episodes(sample)
            await self.backend.ingest_sample(sample)
            for qa in sample.qa:
                prediction, context_ids = await self.backend.answer_question(sample, qa)
                self.evaluator.record(
                    sample, qa, prediction, context_ids,
                    context_docs=self.backend.last_context_docs,
                )
                q_done += 1
            await self.backend.clear_graph()
            n_q = len(sample.qa)
            f1_avg = self.evaluator.f1_sum / self.evaluator.total if self.evaluator.total else 0
            recall_avg = self.evaluator.recall_sum / self.evaluator.total if self.evaluator.total else 0
            print(
                f'[{idx}/{total_samples}] Sample {sample.sample_id} | '
                f'{n_q} questions (total {q_done}) | f1={f1_avg:.3f} recall={recall_avg:.3f}',
                flush=True,
            )

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
        print(f'Total questions: {self.evaluator.total}', flush=True)
