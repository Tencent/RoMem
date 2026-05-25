"""RoMem runner for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import logging

from tqdm import tqdm

from benchmarks.evaluators.multitq import MultiTQEvaluator
from benchmarks.loaders.multitq import MultiTQLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.multitq.romem_backend import MultiTQRoMemBackend

logger = logging.getLogger(__name__)


class MultiTQRoMemRunner(BenchmarkRunnerBase):
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
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
    ):
        super().__init__("multitq", exp_name)
        self.loader = MultiTQLoader(
            root=data_path,
            eval_split=eval_split,
            max_time_ids=max_time_ids,
            max_examples=max_examples,
        )
        self.backend = MultiTQRoMemBackend(
            search_limit=search_limit,
            answer_context_sizes=answer_context_sizes,
            answer_llm_model=answer_llm_model,
            answer_llm_api_key=answer_llm_api_key,
            answer_llm_base_url=answer_llm_base_url,
            romem_config=romem_config,
            romem_save_dir=romem_save_dir,
            romem_llm_model=romem_llm_model,
            romem_embedding_model=romem_embedding_model,
            romem_openie_mode=romem_openie_mode,
            romem_temporal_awareness=romem_temporal_awareness,
            romem_enable_tkge_tunnel=romem_enable_tkge_tunnel,
            romem_tkge_verbose=romem_tkge_verbose,
            exp_name=exp_name,
        )
        self.evaluator = MultiTQEvaluator("multitq", exp_name=exp_name)
        self._max_examples = max_examples

    async def run(self):
        # Phase 1: Ingest all KG triples in a single batch, then train TKGE once.
        all_triples = self.loader.iter_all_triples()
        if all_triples:
            logger.info("Ingesting %d triples...", len(all_triples))
            await self.backend.ingest_all_triples(all_triples, train_tkge=False)
            loaded = self.backend.load_tkge_checkpoint()
            if loaded:
                logger.info("Loaded TKGE checkpoint (train_calls=%s). Continuing training...", loaded.get("train_calls"))
            else:
                logger.info("No TKGE checkpoint found. Training from scratch...")
            self.backend.train_tkge()
            self.backend.save_tkge_checkpoint()
            logger.info("TKGE training complete, checkpoint saved.")

        # Phase 2: Evaluate test questions
        questions = list(self.loader.iter_questions())
        progress = tqdm(questions, desc="MultiTQ questions")
        for question in progress:
            retrieval_metrics, llm_metrics, details = await self.backend.evaluate_question(question)
            self.evaluator.record(question, retrieval_metrics, llm_metrics, details=details)
            # Log progress
            llm_parts = []
            answers = details.get("llm_answers", {})
            for k in self.backend.answer_context_sizes:
                llm_parts.append(
                    f'@{k}: acc={llm_metrics.get(f"llm@{k}_accuracy", 0.0):.2f} '
                    f'ans="{answers.get(k, "")}"'
                )
            llm_summary = "; ".join(llm_parts) if llm_parts else "N/A"
            retrieval_summary = ", ".join(
                f"{key}={retrieval_metrics.get(key, 0.0):.3f}"
                for key in sorted(retrieval_metrics.keys())
                if key.startswith("answer_in_context")
            )
            logger.info(
                "Q%s [%s] | retrieval[%s] | LLM[%s] | gold=%s",
                question.quid,
                question.qtype,
                retrieval_summary,
                llm_summary,
                question.answers[:3],
            )
        progress.close()
        self.evaluator.log_summary()
        self.backend.finalize_usage_logging()
        logger.info("MultiTQ complete: %s questions processed.", self.evaluator.total)
