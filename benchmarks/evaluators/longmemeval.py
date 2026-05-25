"""
Evaluator for LongMemEval QA benchmark.
"""

from __future__ import annotations

from benchmarks.reports.logger import ExperimentLogger
from benchmarks.types import BenchmarkResult, LongmemevalExample


class LongmemevalEvaluator:
    def __init__(self, dataset: str, exp_name: str | None = None):
        self.dataset = dataset
        self.logger = ExperimentLogger(dataset, exp_name) if exp_name else None
        self.total = 0
        self.retrieval_sums: dict[str, float] = {}
        self.llm_totals: dict[int, dict[str, float]] = {}
        self.official_totals: dict[str, float] = {}

    def record(
        self,
        example: LongmemevalExample,
        retrieval_metrics: dict[str, float],
        llm_metrics: dict[str, float],
        details: dict | None = None,
    ) -> BenchmarkResult:
        self.total += 1
        for key, value in retrieval_metrics.items():
            if key.startswith('recall_') or key.startswith('ndcg_') or key.startswith('turn_'):
                self.retrieval_sums[key] = self.retrieval_sums.get(key, 0.0) + value

        grouped = self._group_llm_metrics(llm_metrics)
        for k, stats in grouped.items():
            bucket = self.llm_totals.setdefault(k, {'accuracy': 0.0, 'count': 0.0})
            bucket['accuracy'] += stats.get('accuracy', 0.0)
            bucket['count'] += 1

        metrics = {**retrieval_metrics, **llm_metrics}
        self._accumulate_official(metrics)
        result = BenchmarkResult(
            dataset=self.dataset,
            mode='qa',
            snapshot_label=example.question_id,
            window_id=None,
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            detail: dict = {
                'question_id': example.question_id,
                'question': example.question,
                'question_type': example.question_type,
                'gold_answer': example.answer,
                'predictions': (details or {}).get('llm_answers', {}),
                'matching_fact': (details or {}).get('matching_fact', ''),
                'retrieval_metrics': retrieval_metrics,
                'llm_metrics': llm_metrics,
            }
            if details:
                for key in ('context_docs', 'judge_verdicts'):
                    if key in details:
                        detail[key] = details[key]
            self.logger.log_detail(detail)
        return result

    def log_summary(self) -> None:
        if not self.logger or not self.total:
            return
        summary = {'total_questions': self.total}
        for key, value in self.retrieval_sums.items():
            summary[f'{key}_mean'] = value / self.total
        for name, value in self.official_totals.items():
            summary[f'{name}_mean'] = value / self.total
        for k, stats in sorted(self.llm_totals.items()):
            count = stats.get('count', 0.0)
            if not count:
                continue
            summary[f'llm@{k}_accuracy_mean'] = stats.get('accuracy', 0.0) / count
        self.logger.log(
            BenchmarkResult(
                dataset=self.dataset,
                mode='summary',
                snapshot_label='aggregate',
                window_id=None,
                metrics=summary,
            )
        )

    def _group_llm_metrics(self, metrics: dict[str, float]) -> dict[int, dict[str, float]]:
        grouped: dict[int, dict[str, float]] = {}
        for key, value in metrics.items():
            if not key.startswith('llm@'):
                continue
            remainder = key[4:]
            if '_' not in remainder:
                continue
            size_str, metric = remainder.split('_', 1)
            if not size_str.isdigit():
                continue
            k = int(size_str)
            bucket = grouped.setdefault(k, {})
            if metric == 'accuracy':
                bucket['accuracy'] = value
        return grouped

    def _accumulate_official(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if key.startswith('official'):
                self.official_totals[key] = self.official_totals.get(key, 0.0) + value
