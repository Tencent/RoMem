"""Mem0 runner for the FinTMMBench benchmark."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.fintmmbench import FinTMMBenchEvaluator
from benchmarks.loaders.fintmmbench import FinTMMBenchLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.fintmmbench.mem0_backend import FinTMMBenchMem0Backend

logger = logging.getLogger(__name__)


class FinTMMBenchMem0Runner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        uri: str | None,
        user: str | None,
        password: str | None,
        database: str | None = None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        sampled_tag: str | None = None,
    ):
        super().__init__('fintmmbench', exp_name)
        self.loader = FinTMMBenchLoader(data_dir=data_path, max_examples=max_examples, sampled_tag=sampled_tag)
        self.backend = FinTMMBenchMem0Backend(
            neo4j_uri=uri,
            neo4j_user=user,
            neo4j_password=password,
            neo4j_database=database,
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
        )
        self.evaluator = FinTMMBenchEvaluator('fintmmbench', exp_name=exp_name)

    async def run(self):
        # One-time corpus ingestion
        corpus = self.loader.get_corpus()
        await self.backend.ingest_corpus(corpus)

        # Evaluate each question
        iterator = self.loader.iter_examples()
        progress = tqdm(iterator, desc='FinTMMBench questions', total=self.loader.sample_count)
        for idx, example in enumerate(progress, start=1):
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(example)
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)
            answers = details.get('llm_answers', {})
            llm_parts = []
            for k in self.backend.answer_context_sizes:
                llm_parts.append(
                    f'@{k}: acc={llm_metrics.get(f"llm@{k}_accuracy", 0.0):.0f} '
                    f'ans="{answers.get(k, "")}"'
                )
            llm_summary = '; '.join(llm_parts) if llm_parts else 'N/A'
            recall_parts = ' '.join(
                f'R@{k}={retrieval_metrics.get(f"retrieval_recall@{k}", 0.0):.2f}'
                for k in [1, 3, 5, 10]
            )
            logger.info(
                'Q %s [%s] | %s mrr=%.3f | '
                'LLM[%s] | gold="%s"',
                example.uuid,
                example.question_type,
                recall_parts,
                retrieval_metrics.get('retrieval_mrr', 0.0),
                llm_summary,
                example.answer,
            )
            self.backend.log_usage_summary(prefix=f'question={idx}')
        progress.close()

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()
        n = self.evaluator.total
        if n:
            llm_summary_parts = []
            for k, stats in sorted(self.evaluator.llm_totals.items()):
                count = stats.get('count', 0.0)
                if not count:
                    continue
                llm_summary_parts.append(
                    f'@{k} acc={stats.get("accuracy", 0.0)/count:.3f}'
                )
            llm_overview = '; '.join(llm_summary_parts) if llm_summary_parts else 'n/a'
            recall_parts = ' '.join(
                f'R@{k}={self.evaluator.recall_at_k_sums[k]/n:.3f}'
                for k in [1, 3, 5, 10]
            )
            mrr_avg = self.evaluator.mrr_sum / n
            logger.info(
                'FinTMMBench evaluation complete: %s questions | '
                '%s mrr=%.3f | LLM=%s',
                n, recall_parts, mrr_avg, llm_overview,
            )
            print('\n' + '=' * 70)
            print('  FinTMMBench Aggregate Results')
            print('=' * 70)
            print(f'  Questions evaluated : {n}')
            for k in [1, 3, 5, 10]:
                print(f'  Retrieval recall@{k:<2} : {self.evaluator.recall_at_k_sums[k]/n:.4f}')
            print(f'  Retrieval MRR       : {mrr_avg:.4f}')
            for k, stats in sorted(self.evaluator.llm_totals.items()):
                count = stats.get('count', 0.0)
                if not count:
                    continue
                print(f'  LLM@{k} accuracy    : {stats.get("accuracy", 0.0)/count:.4f}')
            print('=' * 70 + '\n')
