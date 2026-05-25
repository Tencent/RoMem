"""LiCoMemory runner for the LoCoMo dataset."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.locomo import LocomoEvaluator
from benchmarks.loaders.locomo import LocomoLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.locomo.licomem_backend import LocomoLiCoMemBackend
from benchmarks.runners.locomo.utils import filter_sample_episodes

logger = logging.getLogger(__name__)


def _suppress_licomem_logs():
    """Suppress LiCoMemory INFO/DEBUG logs but keep WARNING/ERROR visible."""
    import os
    os.environ['LICOMEM_QUIET'] = '1'
    for name in ('DynamicMemory', 'GraphRAG', 'EntityExtractor', 'DialogueExtractor',
                 'SessionSummarizer', 'GraphBuilder', 'ChunkProcessor',
                 'QueryProcessor', 'TripleReranker', 'SummaryRetriever',
                 'EmbeddingManager', 'LLMManager'):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.handlers.clear()
        lg.propagate = True


class LocomoLiCoMemRunner(BenchmarkRunnerBase):
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
        licomem_llm_model: str | None = None,
        licomem_llm_api_key: str | None = None,
        licomem_llm_base_url: str | None = None,
        licomem_embed_model: str | None = None,
        licomem_embed_api_key: str | None = None,
        licomem_embed_api_type: str | None = None,
        licomem_embed_dimensions: int | None = None,
        licomem_base_dir: str | None = None,
        question_type: str | None = None,
    ):
        super().__init__('locomo', exp_name)
        self.loader = LocomoLoader(
            data_path=data_path,
            max_examples=max_examples,
            question_type=question_type,
        )
        self.backend = LocomoLiCoMemBackend(
            search_limit=search_limit,
            llm_context_sizes=llm_context_sizes,
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
        self.evaluator = LocomoEvaluator('locomo', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        _suppress_licomem_logs()
        if self.loader.is_mc:
            iterator = self.loader.iter_mc_examples()
            progress = tqdm(iterator, desc='LoCoMo-MC (LiCoMemory)', total=self._max_examples)
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

        total_samples = self._max_examples or self.loader.sample_count
        q_done = 0
        for idx, sample in enumerate(self.loader.iter_samples(), start=1):
            sample = filter_sample_episodes(sample)
            await self.backend.ingest_sample(sample)
            for qa in sample.qa:
                prediction, context_ids = await self.backend.answer_question(sample, qa)
                self.evaluator.record(sample, qa, prediction, context_ids, context_docs=self.backend.last_context_docs)
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
