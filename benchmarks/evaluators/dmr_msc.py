"""
Evaluator for the DMR-MSC benchmark.
"""

from __future__ import annotations

from benchmarks.reports.logger import ExperimentLogger
from benchmarks.types import BenchmarkResult, DmrMscExample


class DmrMscEvaluator:
    def __init__(self, dataset: str, exp_name: str | None = None):
        self.dataset = dataset
        self.logger = ExperimentLogger(dataset, exp_name) if exp_name else None
        self.total = 0
        self.hit_sum = 0.0
        self.hit3_sum = 0.0
        self.mrr_sum = 0.0
        self.llm_totals: dict[int, dict[str, float]] = {}
        self.window_stats: dict[str, dict[str, float]] = {}

    def record(
        self,
        example: DmrMscExample,
        retrieval_metrics: dict[str, float],
        llm_metrics: dict[str, float],
        details: dict | None = None,
    ) -> BenchmarkResult:
        self.total += 1
        self.hit_sum += retrieval_metrics.get('retrieval_hit@1', 0.0)
        self.hit3_sum += retrieval_metrics.get('retrieval_hit@3', 0.0)
        self.mrr_sum += retrieval_metrics.get('retrieval_mrr', 0.0)
        grouped_llm = self._group_llm_metrics(llm_metrics)
        for k, stats in grouped_llm.items():
            agg = self.llm_totals.setdefault(k, {'exact': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'count': 0.0})
            agg['exact'] += stats.get('exact', 0.0)
            agg['f1'] += stats.get('f1', 0.0)
            agg['accuracy'] += stats.get('accuracy', 0.0)
            agg['count'] += 1

        metrics = {
            **retrieval_metrics,
            **llm_metrics,
        }
        window_id = example.window_id or 'dmr_msc'
        result = BenchmarkResult(
            dataset=self.dataset,
            mode='qa',
            snapshot_label=example.example_id,
            window_id=window_id,
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            detail: dict = {
                'example_id': example.example_id,
                'window_id': window_id,
                'question': example.question,
                'gold_answer': example.answer,
                'predictions': (details or {}).get('llm_answers', {}),
                'matching_fact': (details or {}).get('matching_fact', ''),
                'top_fact': (details or {}).get('top_fact', ''),
                'retrieval_metrics': retrieval_metrics,
                'llm_metrics': llm_metrics,
            }
            if details:
                for key in ('context_docs',):
                    if key in details:
                        detail[key] = details[key]
            self.logger.log_detail(detail)
        self._update_window_stats(window_id, retrieval_metrics, llm_metrics)
        return result

    def log_summary(self) -> None:
        if self.total == 0:
            return
        overall = {
            'retrieval_accuracy_mean': self.hit_sum / self.total,
            'retrieval_hit@3_mean': self.hit3_sum / self.total,
            'retrieval_mrr_mean': self.mrr_sum / self.total,
            'total_questions': self.total,
        }
        for k, stats in sorted(self.llm_totals.items()):
            count = stats.get('count', 0.0)
            if not count:
                continue
            overall[f'llm@{k}_exact_mean'] = stats.get('exact', 0.0) / count
            overall[f'llm@{k}_f1_mean'] = stats.get('f1', 0.0) / count
            overall[f'llm@{k}_accuracy_mean'] = stats.get('accuracy', 0.0) / count
        if self.logger:
            self.logger.log(
                BenchmarkResult(
                    dataset=self.dataset,
                    mode='summary',
                    snapshot_label='aggregate',
                    window_id='dmr_msc',
                    metrics=overall,
                )
            )
        for window_id, stats in self.window_stats.items():
            count = stats.get('count', 0.0)
            if count <= 1:
                continue
            window_metrics = {
                'retrieval_accuracy_mean': stats.get('hit', 0.0) / count,
                'retrieval_hit@3_mean': stats.get('hit3', 0.0) / count,
                'retrieval_mrr_mean': stats.get('mrr', 0.0) / count,
                'total_questions': count,
            }
            llm_bucket = stats.get('llm', {})
            for k, bucket in llm_bucket.items():
                sub_count = bucket.get('count', 0.0)
                if not sub_count:
                    continue
                window_metrics[f'llm@{k}_exact_mean'] = bucket.get('exact', 0.0) / sub_count
                window_metrics[f'llm@{k}_f1_mean'] = bucket.get('f1', 0.0) / sub_count
                window_metrics[f'llm@{k}_accuracy_mean'] = bucket.get('accuracy', 0.0) / sub_count
            if self.logger:
                self.logger.log(
                    BenchmarkResult(
                        dataset=self.dataset,
                        mode='summary',
                        snapshot_label='aggregate',
                        window_id=window_id,
                        metrics=window_metrics,
                    )
                )
        # If no logger, emit a quick stdout summary for convenience.
        if not self.logger:
            print('DMR-MSC summary (no logger configured):', overall)

    def _update_window_stats(
        self,
        window_id: str,
        retrieval_metrics: dict[str, float],
        llm_metrics: dict[str, float],
    ) -> None:
        stats = self.window_stats.setdefault(
            window_id,
            {
                'count': 0.0,
                'hit': 0.0,
                'hit3': 0.0,
                'mrr': 0.0,
                'llm': {},
            },
        )
        stats['count'] += 1
        stats['hit'] += retrieval_metrics.get('retrieval_hit@1', 0.0)
        stats['hit3'] += retrieval_metrics.get('retrieval_hit@3', 0.0)
        stats['mrr'] += retrieval_metrics.get('retrieval_mrr', 0.0)
        grouped_llm = self._group_llm_metrics(llm_metrics)
        for k, sub in grouped_llm.items():
            bucket = stats['llm'].setdefault(k, {'exact': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'count': 0.0})
            bucket['exact'] += sub.get('exact', 0.0)
            bucket['f1'] += sub.get('f1', 0.0)
            bucket['accuracy'] += sub.get('accuracy', 0.0)
            bucket['count'] += 1

    def _group_llm_metrics(self, llm_metrics: dict[str, float]) -> dict[int, dict[str, float]]:
        grouped: dict[int, dict[str, float]] = {}
        for key, value in llm_metrics.items():
            if not key.startswith('llm@'):
                continue
            remainder = key[4:]
            if '_' not in remainder:
                continue
            size_str, metric_name = remainder.split('_', 1)
            if not size_str.isdigit():
                continue
            k = int(size_str)
            entry = grouped.setdefault(k, {})
            if metric_name == 'exact':
                entry['exact'] = value
            elif metric_name == 'f1':
                entry['f1'] = value
            elif metric_name == 'accuracy':
                entry['accuracy'] = value
        return grouped
