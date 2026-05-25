"""HippoRAG runner for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import logging

from tqdm import tqdm

from benchmarks.evaluators.multitq import MultiTQEvaluator
from benchmarks.loaders.multitq import MultiTQLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.multitq.hipporag_backend import MultiTQHippoRAGBackend

logger = logging.getLogger(__name__)


class MultiTQHippoRAGRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None = None,
        eval_split: str = "test",
        max_time_ids: int | None = None,
        max_examples: int | None = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        hippo_save_dir: str | None = None,
        hippo_llm_model: str | None = None,
        hippo_embedding_model: str | None = None,
        hippo_openie_mode: str | None = None,
    ):
        super().__init__("multitq", exp_name)
        self.loader = MultiTQLoader(
            root=data_path, eval_split=eval_split,
            max_time_ids=max_time_ids, max_examples=max_examples,
        )
        self.backend = MultiTQHippoRAGBackend(
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
            hippo_save_dir=hippo_save_dir,
            hippo_llm_model=hippo_llm_model,
            hippo_embedding_model=hippo_embedding_model,
            hippo_openie_mode=hippo_openie_mode,
        )
        self.evaluator = MultiTQEvaluator("multitq", exp_name=exp_name)

    async def run(self):
        # Phase 1: Ingest all KG triples in a single batch
        all_triples = self.loader.iter_all_triples()
        if all_triples:
            logger.info("Ingesting %d triples...", len(all_triples))
            await self.backend.ingest_all_triples(all_triples)
            logger.info("All triples ingested.")

        questions = list(self.loader.iter_questions())
        progress = tqdm(questions, desc="MultiTQ questions")
        for question in progress:
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_question(question)
            self.evaluator.record(question, retrieval_metrics, llm_metrics, details=details)
        progress.close()
        self.evaluator.log_summary()
