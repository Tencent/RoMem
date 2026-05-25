"""LiCoMemory runner for the FinTMMBench temporal financial QA benchmark."""

from __future__ import annotations

import logging
import os

from benchmarks.evaluators.fintmmbench import FinTMMBenchEvaluator
from benchmarks.loaders.fintmmbench import FinTMMBenchLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.fintmmbench.licomem_backend import FinTMMBenchLiCoMemBackend

logger = logging.getLogger(__name__)


class FinTMMBenchLiCoMemRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None = None,
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
        sampled_tag: str | None = None,
    ):
        # Suppress verbose LiCoMemory logs
        os.environ.setdefault('LICOMEM_QUIET', '1')

        super().__init__('fintmmbench', exp_name)
        self.loader = FinTMMBenchLoader(
            data_dir=data_path,
            max_examples=max_examples,
            sampled_tag=sampled_tag,
        )
        self.backend = FinTMMBenchLiCoMemBackend(
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
        self.evaluator = FinTMMBenchEvaluator('fintmmbench', exp_name=exp_name)

    async def run(self):
        # Phase 1: One-time corpus ingestion
        corpus = self.loader.get_corpus()
        await self.backend.ingest_corpus(corpus)

        # Phase 2: Evaluate each question
        total = self.loader.sample_count
        print(
            f'[LiCoMemory/FinTMMBench] Evaluating {total} examples...',
            flush=True,
        )

        for idx, example in enumerate(self.loader.iter_examples(), start=1):
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(
                example
            )
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)

            if idx % 20 == 0 or idx == total:
                recall_parts = ' '.join(
                    f'R@{k}={retrieval_metrics.get(f"retrieval_recall@{k}", 0.0):.2f}'
                    for k in [1, 3, 5, 10]
                )
                mrr = retrieval_metrics.get('retrieval_mrr', 0.0)
                print(
                    f'[LiCoMemory/FinTMMBench] {idx}/{total} | '
                    f'{recall_parts} mrr={mrr:.3f}',
                    flush=True,
                )

        self.backend.finalize_usage_logging()
        self.evaluator.log_summary()

        n = self.evaluator.total
        if n:
            print('\n' + '=' * 70, flush=True)
            print('  FinTMMBench LiCoMemory Aggregate Results', flush=True)
            print('=' * 70, flush=True)
            print(f'  Questions evaluated : {n}', flush=True)
            for k in [1, 3, 5, 10]:
                recall_sum = self.evaluator.recall_at_k_sums.get(k, 0.0)
                print(
                    f'  Retrieval recall@{k:<2} : {recall_sum / n:.4f}',
                    flush=True,
                )
            mrr_avg = self.evaluator.mrr_sum / n
            print(f'  Retrieval MRR       : {mrr_avg:.4f}', flush=True)
            for k, stats in sorted(self.evaluator.llm_totals.items()):
                count = stats.get('count', 0.0)
                if not count:
                    continue
                print(
                    f'  LLM@{k} accuracy    : {stats.get("accuracy", 0.0) / count:.4f}',
                    flush=True,
                )
            print('=' * 70 + '\n', flush=True)
