"""Graphiti runner for the LongMemEval dataset."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.longmemeval import LongmemevalEvaluator
from benchmarks.loaders.longmemeval import LongmemevalLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.longmemeval.graphiti_backend import LongmemevalGraphitiBackend
from benchmarks.runners.shared_utils import usage_totals, usage_delta

logger = logging.getLogger(__name__)


class LongmemevalGraphitiRunner(BenchmarkRunnerBase):
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
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        judge_llm_model: str | None = None,
        judge_llm_api_key: str | None = None,
        judge_llm_base_url: str | None = None,
        disable_judge: bool = False,
        enable_official_comparison: bool = False,
        official_baseline_path: str | None = None,
        search_reranker: str | None = None,
    ):
        super().__init__('longmemeval', exp_name)
        self.loader = LongmemevalLoader(data_path=data_path, max_examples=max_examples)
        self.backend = LongmemevalGraphitiBackend(
            uri,
            user,
            password,
            database=database,
            reset_mode=reset_mode,
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
            judge_llm_model=judge_llm_model,
            judge_llm_api_key=judge_llm_api_key,
            judge_llm_base_url=judge_llm_base_url,
            disable_judge=disable_judge,
            enable_official_comparison=enable_official_comparison,
            official_baseline_path=official_baseline_path,
            search_reranker=search_reranker,
        )
        self.evaluator = LongmemevalEvaluator('longmemeval', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        iterator = self.loader.iter_examples()
        progress = tqdm(iterator, desc='LongMemEval samples', total=self._max_examples)
        for idx, example in enumerate(progress, start=1):
            before = usage_totals()
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(example)
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)
            after = usage_totals()
            delta = usage_delta(before, after)
            llm_parts = []
            answers = details.get('llm_answers', {})
            for k in self.backend.answer_context_sizes:
                llm_parts.append(
                    f'@{k}: acc={llm_metrics.get(f"llm@{k}_accuracy", 0.0):.2f} '
                    f'ans="{answers.get(k, "")}"'
                )
            llm_summary = '; '.join(llm_parts) if llm_parts else 'N/A'
            retrieval_summary = ', '.join(
                f'{key}={retrieval_metrics.get(key, 0.0):.3f}'
                for key in ('turn_recall_all@5', 'turn_ndcg_any@5', 'turn_recall_all@10', 'turn_ndcg_any@10', 'turn_recall_all@15', 'turn_ndcg_any@15')
                if key in retrieval_metrics
            )
            logger.info(
                'Sample %s | retrieval[%s] | LLM[%s] | fact="%s" | gold="%s" | tokens=+%s/+%s cost=+%.6f',
                example.question_id,
                retrieval_summary,
                llm_summary,
                details.get('matching_fact', ''),
                example.answer,
                int(delta['prompt_tokens']),
                int(delta['completion_tokens']),
                delta['cost'],
            )
            self.backend.log_usage_summary(prefix=f'sample={idx}')
        progress.close()

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
        logger.info('LongMemEval complete: %s samples processed.', self.evaluator.total)
