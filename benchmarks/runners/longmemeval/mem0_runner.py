"""Mem0 runner for the LongMemEval benchmark."""

from __future__ import annotations

from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.longmemeval import LongmemevalEvaluator
from benchmarks.loaders.longmemeval import LongmemevalLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.longmemeval.mem0_backend import LongmemevalMem0Backend


class LongmemevalMem0Runner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        neo4j_uri: str | None,
        neo4j_user: str | None,
        neo4j_password: str | None,
        neo4j_database: str | None = None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        reset_mode: str = 'delete',
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        judge_llm_model: str | None = None,
        judge_llm_api_key: str | None = None,
        judge_llm_base_url: str | None = None,
        disable_judge: bool = False,
    ):
        super().__init__('longmemeval', exp_name)
        self.loader = LongmemevalLoader(data_path=data_path, max_examples=max_examples)
        self.backend = LongmemevalMem0Backend(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
            judge_llm_model=judge_llm_model,
            judge_llm_api_key=judge_llm_api_key,
            judge_llm_base_url=judge_llm_base_url,
            disable_judge=disable_judge,
        )
        self.evaluator = LongmemevalEvaluator('longmemeval', exp_name=exp_name)
        self._max_examples = max_examples
        self.reset_mode = reset_mode

    async def run(self):
        iterator = self.loader.iter_examples()
        progress = tqdm(iterator, desc='LongMemEval samples', total=self._max_examples)
        for example in progress:
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(example)
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)
            self.backend.log_usage_summary()
        progress.close()
        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
