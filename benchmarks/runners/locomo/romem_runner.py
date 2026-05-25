"""RoMem runner for the LoCoMo dataset."""

from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from benchmarks.evaluators.locomo import LocomoEvaluator
from benchmarks.loaders.locomo import LocomoLoader
from benchmarks.runners.base_runner import BenchmarkRunnerBase
from benchmarks.runners.locomo.romem_backend import LocomoRoMemBackend
from benchmarks.evaluators.locomo import normalize_answer
from benchmarks.runners.locomo.utils import filter_sample_episodes

logger = logging.getLogger(__name__)


class LocomoRoMemRunner(BenchmarkRunnerBase):
    def __init__(
        self,
        data_path: str | None,
        max_examples: Optional[int] = None,
        exp_name: str | None = None,
        search_limit: int | None = None,
        llm_context_sizes: list[int] | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        question_type: str | None = None,
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
    ):
        super().__init__('locomo', exp_name)
        self.loader = LocomoLoader(
            data_path=data_path,
            max_examples=max_examples,
            question_type=question_type,
        )
        self.backend = LocomoRoMemBackend(
            search_limit=search_limit,
            llm_context_sizes=llm_context_sizes,
            llm_model_override=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            romem_config=romem_config,
            romem_save_dir=romem_save_dir,
            romem_llm_model=romem_llm_model,
            romem_embedding_model=romem_embedding_model,
            romem_openie_mode=romem_openie_mode,
            romem_temporal_awareness=romem_temporal_awareness,
            romem_enable_tkge_tunnel=romem_enable_tkge_tunnel,
            romem_tkge_verbose=romem_tkge_verbose,
        )
        self.evaluator = LocomoEvaluator('locomo', exp_name=exp_name)
        self._max_examples = max_examples
        self._debug_retrieval = (romem_tkge_verbose or 0) >= 3

    @staticmethod
    def _find_choice_rank(docs: list[str], choice: str) -> int | None:
        if not docs:
            return None
        normalized_choice = normalize_answer(choice)
        if not normalized_choice:
            return None
        for idx, doc in enumerate(docs, start=1):
            if normalized_choice in normalize_answer(doc or ''):
                return idx
        return None

    @staticmethod
    def _fact_matches_answer(fact_entry: dict, answer: str) -> bool:
        norm_answer = normalize_answer(answer)
        if not norm_answer:
            return False
        fact = fact_entry.get("fact")
        happen = fact_entry.get("happen_time", "")
        parts = []
        if fact:
            if isinstance(fact, (list, tuple)):
                parts.extend(str(x) for x in fact)
            else:
                parts.append(str(fact))
        if happen:
            parts.append(str(happen))
        hay = normalize_answer(" ".join(parts))
        return norm_answer in hay

    async def run(self):
        if self.loader.is_mc:
            iterator = self.loader.iter_mc_examples()
            progress = tqdm(iterator, desc='LoCoMo-MC samples', total=self._max_examples)
            current_group: str | None = None
            for idx, example in enumerate(progress, start=1):
                group_id = example.question_id.split('_q', 1)[0]
                if group_id != current_group:
                    if current_group is not None:
                        await self.backend.clear_graph()
                    await self.backend.ingest_example(example)
                    current_group = group_id
                prediction, context_ids = await self.backend.answer_mc_question(example)
                if self._debug_retrieval:
                    gold_choice = example.choices[example.correct_choice_index]
                    rank = self._find_choice_rank(self.backend._last_mc_docs, gold_choice)
                    snippet = ''
                    if rank is not None:
                        snippet = (self.backend._last_mc_docs[rank - 1] or '').replace('\n', ' ')[:200]
                    logger.info(
                        'LoCoMo-MC debug %s gold_choice_rank=%s gold_choice=%r snippet=%r',
                        example.question_id,
                        rank,
                        gold_choice,
                        snippet,
                    )
                self.evaluator.record_mc(example, prediction, context_ids)
            progress.close()
            await self.backend.clear_graph()
            self.evaluator.log_summary()
            print(f'Total questions: {self.evaluator.mc_total}')
            return

        iterator = self.loader.iter_samples()
        progress = tqdm(iterator, desc='LoCoMo samples', total=self._max_examples)
        for idx, sample in enumerate(progress, start=1):
            sample = filter_sample_episodes(sample)
            await self.backend.ingest_sample(sample)
            for qa in sample.qa:
                prediction, context_ids = await self.backend.answer_question(sample, qa)
                if self._debug_retrieval and self.backend.romem is not None:
                    debug = self.backend.romem.last_rerank_debug or {}
                    kge_debug = self.backend.romem.last_tkge_debug or {}
                    combined = {
                        "query": debug.get("query") or kge_debug.get("query"),
                        "gold_answer": qa.answer,
                        "query_time": debug.get("query_time"),
                        "time_request": debug.get("time_request"),
                        "time_mode": kge_debug.get("time_mode"),
                        "before": kge_debug.get("order_before"),
                        "after": kge_debug.get("order_after"),
                        "rank_deltas": kge_debug.get("rank_deltas"),
                        "time_recall": kge_debug.get("time_recall"),
                        "rotation_debug": kge_debug.get("rotation_debug"),
                        "facts": debug.get("facts"),
                    }
                    logger.info('LoCoMo-QA debug %s %s', sample.sample_id, combined)
                self.evaluator.record(sample, qa, prediction, context_ids, context_docs=self.backend.last_context_docs)
            await self.backend.clear_graph()
        progress.close()

        self.evaluator.log_summary()
        print(f'Total questions: {self.evaluator.total}')
