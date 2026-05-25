"""
Evaluator for the FinTMMBench temporal financial QA benchmark.

Metrics:
  - retrieval_recall@k: fraction of gold source docs found in top-k results (k=1,3,5,10)
  - retrieval_mrr: mean reciprocal rank
  - llm@k_accuracy: LLM judge answer correctness at context size k
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmarks.types import BenchmarkResult, FinTMMBenchExample
from benchmarks.reports.logger import ExperimentLogger


RECALL_KS = [1, 3, 5, 10]


def _default_type_stats() -> dict[str, float]:
    d: dict[str, float] = {'count': 0.0, 'mrr': 0.0}
    for k in RECALL_KS:
        d[f'recall@{k}'] = 0.0
    return d


class FinTMMBenchEvaluator:
    def __init__(self, dataset: str = 'fintmmbench', exp_name: str | None = None):
        self.dataset = dataset
        self.logger = ExperimentLogger(dataset, exp_name) if exp_name else None
        self.total = 0
        self.recall_at_k_sums: dict[int, float] = {k: 0.0 for k in RECALL_KS}
        self.mrr_sum = 0.0
        self.type_stats: dict[str, dict[str, float]] = defaultdict(_default_type_stats)
        self.subtask_stats: dict[str, dict[str, float]] = defaultdict(_default_type_stats)
        self.llm_totals: dict[int, dict[str, float]] = {}

    def record(
        self,
        example: FinTMMBenchExample,
        retrieval_metrics: dict[str, float],
        llm_metrics: dict[str, float],
        details: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        self.total += 1
        mrr = retrieval_metrics.get('retrieval_mrr', 0.0)
        self.mrr_sum += mrr
        for k in RECALL_KS:
            self.recall_at_k_sums[k] += retrieval_metrics.get(f'retrieval_recall@{k}', 0.0)

        # Per question-type stats
        bucket = self.type_stats[example.question_type]
        bucket['count'] += 1
        bucket['mrr'] += mrr
        for k in RECALL_KS:
            bucket[f'recall@{k}'] += retrieval_metrics.get(f'retrieval_recall@{k}', 0.0)

        # Per subtask stats
        for subtask in example.subtasks:
            sb = self.subtask_stats[subtask]
            sb['count'] += 1
            sb['mrr'] += mrr
            for k in RECALL_KS:
                sb[f'recall@{k}'] += retrieval_metrics.get(f'retrieval_recall@{k}', 0.0)

        # LLM accuracy grouped by context size
        for key, value in llm_metrics.items():
            parts = key.split('_', 1)
            if not parts[0].startswith('llm@'):
                continue
            try:
                k = int(parts[0].replace('llm@', ''))
            except ValueError:
                continue
            metric_name = parts[1] if len(parts) > 1 else ''
            if metric_name != 'accuracy':
                continue
            agg = self.llm_totals.setdefault(k, {'count': 0.0, 'accuracy': 0.0})
            agg['accuracy'] += value
            agg['count'] = max(agg['count'], self.total)

        metrics = {**retrieval_metrics, **llm_metrics}
        result = BenchmarkResult(
            dataset=self.dataset,
            mode='qa',
            snapshot_label=example.uuid,
            window_id=example.question_type,
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            detail_payload = {
                'uuid': example.uuid,
                'question': example.question,
                'gold_answer': example.answer,
                'question_type': example.question_type,
                'subtasks': example.subtasks,
                'source_ids': example.source_ids,
            }
            if details:
                detail_payload.update(details)
            self.logger.log_detail(detail_payload)
        return result

    def log_summary(self) -> None:
        if not self.logger or self.total == 0:
            return

        overall: dict[str, float] = {
            'retrieval_mrr_mean': self.mrr_sum / self.total,
            'total_questions': float(self.total),
        }
        for k in RECALL_KS:
            overall[f'retrieval_recall@{k}_mean'] = self.recall_at_k_sums[k] / self.total

        for k, stats in sorted(self.llm_totals.items()):
            count = stats.get('count', 0.0)
            if not count:
                continue
            overall[f'llm@{k}_accuracy_mean'] = stats.get('accuracy', 0.0) / count

        self.logger.log(BenchmarkResult(
            dataset=self.dataset,
            mode='summary',
            snapshot_label='aggregate',
            window_id='fintmmbench',
            metrics=overall,
        ))

        # Per question-type breakdown
        for qtype, stats in sorted(self.type_stats.items()):
            n = stats['count']
            if n == 0:
                continue
            type_metrics: dict[str, float] = {
                'retrieval_mrr_mean': stats['mrr'] / n,
                'count': n,
            }
            for k in RECALL_KS:
                type_metrics[f'retrieval_recall@{k}_mean'] = stats[f'recall@{k}'] / n
            self.logger.log(BenchmarkResult(
                dataset=self.dataset,
                mode='summary',
                snapshot_label='by_type',
                window_id=qtype,
                metrics=type_metrics,
            ))

        # Per subtask breakdown
        for subtask, stats in sorted(self.subtask_stats.items()):
            n = stats['count']
            if n == 0:
                continue
            sub_metrics: dict[str, float] = {
                'retrieval_mrr_mean': stats['mrr'] / n,
                'count': n,
            }
            for k in RECALL_KS:
                sub_metrics[f'retrieval_recall@{k}_mean'] = stats[f'recall@{k}'] / n
            self.logger.log(BenchmarkResult(
                dataset=self.dataset,
                mode='summary',
                snapshot_label='by_subtask',
                window_id=subtask,
                metrics=sub_metrics,
            ))
