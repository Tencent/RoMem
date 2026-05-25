"""Graphiti runner for the LoCoMo dataset."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.locomo import LocomoEvaluator
from benchmarks.loaders.locomo import LocomoLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.locomo.graphiti_backend import LocomoGraphitiBackend
from benchmarks.runners.locomo.utils import filter_sample_episodes

logger = logging.getLogger(__name__)


class LocomoGraphitiRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        reset_mode: str = 'delete',
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        search_reranker: str | None = None,
        question_type: str | None = None,
    ):
        super().__init__('locomo', exp_name)
        self.loader = LocomoLoader(
            data_path=data_path,
            max_examples=max_examples,
            question_type=question_type,
        )
        self.backend = LocomoGraphitiBackend(
            uri,
            user,
            password,
            database=database,
            reset_mode=reset_mode,
            llm_model_override=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            search_reranker=search_reranker,
        )
        self.evaluator = LocomoEvaluator('locomo', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        if self.loader.is_mc:
            iterator = self.loader.iter_mc_examples()
            progress = tqdm(iterator, desc='LoCoMo-MC samples', total=self.loader.sample_count)
            current_group: str | None = None
            for idx, example in enumerate(progress, start=1):
                group_id = example.question_id.split('_q', 1)[0]
                if group_id != current_group:
                    if current_group is not None:
                        await self.backend.clear_graph()
                    await self.backend.ingest_example(example)
                    current_group = group_id
                prediction, context_ids = await self.backend.answer_mc_question(example)
                self.evaluator.record_mc(example, prediction, context_ids)
                self.backend.log_usage_summary(prefix=f'sample={idx}')
            progress.close()
            await self.backend.clear_graph()
            self.backend.finalize_usage_logging()
            self.evaluator.log_summary()
            print(f'Total questions: {self.evaluator.mc_total}')
            return

        iterator = self.loader.iter_samples()
        progress = tqdm(iterator, desc='LoCoMo samples', total=self.loader.sample_count)
        for idx, sample in enumerate(progress, start=1):
            sample = filter_sample_episodes(sample)
            await self.backend.ingest_sample(sample)
            for qa in sample.qa:
                prediction, context_ids = await self.backend.answer_question(sample, qa)
                self.evaluator.record(sample, qa, prediction, context_ids, context_docs=self.backend.last_context_docs)
            await self.backend.clear_graph()
            self.backend.log_usage_summary(prefix=f'sample={idx}')
        progress.close()

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
        print(f'Total questions: {self.evaluator.total}')
