"""A-mem-sys runner for the DMR-MSC benchmark."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.dmr_msc import DmrMscEvaluator
from benchmarks.loaders.dmr_msc import DmrMscLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.dmr_msc.amem_backend import DmrMscAMemBackend

logger = logging.getLogger(__name__)


class DmrMscAMemRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        amem_model_name: str | None = None,
        amem_llm_backend: str | None = None,
        amem_llm_model: str | None = None,
    ):
        super().__init__('dmr_msc', exp_name)
        self.loader = DmrMscLoader(data_path=data_path, max_examples=max_examples)
        self.backend = DmrMscAMemBackend(
            model_name=amem_model_name,
            llm_backend=amem_llm_backend,
            llm_model=amem_llm_model,
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
        )
        self.evaluator = DmrMscEvaluator('dmr_msc', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        iterator = self.loader.iter_examples()
        progress = tqdm(iterator, desc='DMR-MSC samples (A-mem)', total=self.loader.sample_count)
        for idx, example in enumerate(progress, start=1):
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(example)
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)
            answers = details.get('llm_answers', {})
            llm_parts = []
            for k in self.backend.answer_context_sizes:
                llm_parts.append(
                    f'@{k}: exact={llm_metrics.get(f"llm@{k}_exact", 0.0):.0f} '
                    f'f1={llm_metrics.get(f"llm@{k}_f1", 0.0):.3f} '
                    f'acc={llm_metrics.get(f"llm@{k}_accuracy", 0.0):.0f} '
                    f'ans="{answers.get(k, "")}"'
                )
            llm_summary = '; '.join(llm_parts) if llm_parts else 'N/A'
            logger.info(
                'Sample %s | hit@1=%.0f hit@3=%.0f mrr=%.3f | '
                'LLM[%s] | fact="%s" | gold="%s"',
                example.example_id,
                retrieval_metrics.get('retrieval_hit@1', 0.0),
                retrieval_metrics.get('retrieval_hit@3', 0.0),
                retrieval_metrics.get('retrieval_mrr', 0.0),
                llm_summary,
                details.get('matching_fact') or details.get('top_fact', ''),
                example.answer,
            )
            self.backend.log_usage_summary(prefix=f'sample={idx}')
        progress.close()

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
        llm_summary = []
        for k, stats in sorted(self.evaluator.llm_totals.items()):
            count = stats.get('count', 0.0)
            if not count:
                continue
            llm_summary.append(
                f'@{k} exact={stats.get("exact", 0.0)/count:.3f} '
                f'f1={stats.get("f1", 0.0)/count:.3f} '
                f'acc={stats.get("accuracy", 0.0)/count:.3f}'
            )
        llm_overview = '; '.join(llm_summary) if llm_summary else 'n/a'
        logger.info(
            'DMR-MSC evaluation complete: %s samples | retrieval accuracy=%.3f | LLM=%s',
            self.evaluator.total,
            self.evaluator.hit_sum / self.evaluator.total if self.evaluator.total else 0.0,
            llm_overview,
        )
