"""HippoRAG runner for the LongMemEval dataset."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.longmemeval import LongmemevalEvaluator
from benchmarks.loaders.longmemeval import LongmemevalLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.longmemeval.hipporag_backend import LongmemevalHippoRAGBackend

logger = logging.getLogger(__name__)


class LongmemevalHippoRAGRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
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
        hippo_save_dir: str | None = None,
        hippo_llm_model: str | None = None,
        hippo_embedding_model: str | None = None,
        hippo_openie_mode: str | None = None,
    ):
        super().__init__('longmemeval', exp_name)
        self.loader = LongmemevalLoader(data_path=data_path, max_examples=max_examples)
        self.backend = LongmemevalHippoRAGBackend(
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
            hippo_save_dir=hippo_save_dir,
            hippo_llm_model=hippo_llm_model,
            hippo_embedding_model=hippo_embedding_model,
            hippo_openie_mode=hippo_openie_mode,
        )
        self.evaluator = LongmemevalEvaluator('longmemeval', exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        iterator = self.loader.iter_examples()
        progress = tqdm(iterator, desc='LongMemEval samples', total=self._max_examples)
        for example in progress:
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_example(example)
            self.evaluator.record(example, retrieval_metrics, llm_metrics, details=details)
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
                'Sample %s | retrieval[%s] | LLM[%s] | fact="%s" | gold="%s"',
                example.question_id,
                retrieval_summary,
                llm_summary,
                details.get('matching_fact', ''),
                example.answer,
            )
        progress.close()
        self.evaluator.log_summary()
        logger.info('LongMemEval complete: %s samples processed.', self.evaluator.total)
