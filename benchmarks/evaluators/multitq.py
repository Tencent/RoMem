"""
Evaluator for the MultiTQ temporal KGQA benchmark.

Tracks retrieval (hits@k, MRR, answer_in_context@k) and LLM QA accuracy
with per-category breakdowns by qtype, time_level, answer_type, and qlabel.
"""

from __future__ import annotations

from benchmarks.reports.logger import ExperimentLogger
from benchmarks.types import BenchmarkResult, MultiTQQuestion


class MultiTQEvaluator:
    def __init__(self, dataset: str = "multitq", exp_name: str | None = None):
        self.dataset = dataset
        self.logger = ExperimentLogger(dataset, exp_name) if exp_name else None
        self.total = 0
        self.retrieval_sums: dict[str, float] = {}
        self.llm_totals: dict[int, dict[str, float]] = {}
        # Per-category stats: category_type -> category_value -> stats
        self._category_stats: dict[str, dict[str, dict]] = {}

    def record(
        self,
        question: MultiTQQuestion,
        retrieval_metrics: dict[str, float],
        llm_metrics: dict[str, float],
        details: dict | None = None,
    ) -> BenchmarkResult:
        self.total += 1

        # Accumulate retrieval metrics
        for key, value in retrieval_metrics.items():
            if key.startswith(("answer_in_context", "hits@")) or key == "mrr":
                self.retrieval_sums[key] = self.retrieval_sums.get(key, 0.0) + value

        # Accumulate LLM metrics by context size
        grouped = self._group_llm_metrics(llm_metrics)
        for k, stats in grouped.items():
            bucket = self.llm_totals.setdefault(k, {"accuracy": 0.0, "count": 0.0})
            bucket["accuracy"] += stats.get("accuracy", 0.0)
            bucket["count"] += 1

        # Per-category breakdowns
        categories = {
            "qtype": question.qtype,
            "time_level": question.time_level,
            "answer_type": question.answer_type,
            "qlabel": question.qlabel,
        }
        for cat_type, cat_value in categories.items():
            self._accumulate_category(cat_type, cat_value, retrieval_metrics, grouped)

        # Build combined metrics and log
        metrics = {**retrieval_metrics, **llm_metrics}
        window_id = question.qtype or "multitq"
        result = BenchmarkResult(
            dataset=self.dataset,
            mode="qa",
            snapshot_label=str(question.quid),
            window_id=window_id,
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            detail: dict = {
                "quid": question.quid,
                "question": question.question,
                "answers": question.answers,
                "answer_type": question.answer_type,
                "time_level": question.time_level,
                "qtype": question.qtype,
                "qlabel": question.qlabel,
                "predictions": (details or {}).get("llm_answers", {}),
                "retrieval_metrics": retrieval_metrics,
                "llm_metrics": llm_metrics,
            }
            if details:
                for key in ("context_docs", "matching_fact", "match_strategies"):
                    if key in details:
                        detail[key] = details[key]
            self.logger.log_detail(detail)
        return result

    def log_summary(self) -> None:
        if not self.total:
            return
        summary: dict[str, float] = {"total_questions": float(self.total)}

        # Aggregate retrieval means
        for key, value in self.retrieval_sums.items():
            summary[f"{key}_mean"] = value / self.total

        # Aggregate LLM means
        for k, stats in sorted(self.llm_totals.items()):
            count = stats.get("count", 0.0)
            if not count:
                continue
            summary[f"llm@{k}_accuracy_mean"] = stats.get("accuracy", 0.0) / count

        # Print summary
        retrieval_parts = []
        for key in sorted(self.retrieval_sums.keys()):
            retrieval_parts.append(f"{key}={summary.get(f'{key}_mean', 0.0):.4f}")
        llm_parts = []
        for k in sorted(self.llm_totals.keys()):
            llm_parts.append(
                f"llm@{k}_acc={summary.get(f'llm@{k}_accuracy_mean', 0.0):.4f}"
            )
        print(
            f"MultiTQ summary: n={self.total} "
            + " ".join(retrieval_parts)
            + " "
            + " ".join(llm_parts)
        )

        if self.logger:
            self.logger.log(
                BenchmarkResult(
                    dataset=self.dataset,
                    mode="summary",
                    snapshot_label="aggregate",
                    window_id="multitq",
                    metrics=summary,
                )
            )
            # Per-category breakdown summaries
            for cat_type, cat_values in sorted(self._category_stats.items()):
                for cat_value, stats in sorted(cat_values.items()):
                    count = stats.get("count", 0.0)
                    if not count:
                        continue
                    cat_metrics: dict[str, float] = {"total_questions": count}
                    for key, val in stats.get("retrieval", {}).items():
                        cat_metrics[f"{key}_mean"] = val / count
                    for k, bucket in stats.get("llm", {}).items():
                        denom = bucket.get("count", 0.0)
                        if denom:
                            cat_metrics[f"llm@{k}_accuracy_mean"] = (
                                bucket.get("accuracy", 0.0) / denom
                            )
                    self.logger.log(
                        BenchmarkResult(
                            dataset=self.dataset,
                            mode="summary",
                            snapshot_label="aggregate",
                            window_id=f"{cat_type}={cat_value}",
                            metrics=cat_metrics,
                        )
                    )

    def _accumulate_category(
        self,
        cat_type: str,
        cat_value: str,
        retrieval_metrics: dict[str, float],
        llm_grouped: dict[int, dict[str, float]],
    ) -> None:
        cat_stats = self._category_stats.setdefault(cat_type, {})
        stats = cat_stats.setdefault(
            cat_value, {"count": 0.0, "retrieval": {}, "llm": {}}
        )
        stats["count"] += 1
        for key, value in retrieval_metrics.items():
            if key.startswith(("answer_in_context", "hits@")) or key == "mrr":
                section = stats.setdefault("retrieval", {})
                section[key] = section.get(key, 0.0) + value
        for k, values in llm_grouped.items():
            bucket = stats["llm"].setdefault(k, {"accuracy": 0.0, "count": 0.0})
            bucket["accuracy"] += values.get("accuracy", 0.0)
            bucket["count"] += 1

    def _group_llm_metrics(
        self, metrics: dict[str, float]
    ) -> dict[int, dict[str, float]]:
        grouped: dict[int, dict[str, float]] = {}
        for key, value in metrics.items():
            if not key.startswith("llm@"):
                continue
            remainder = key[4:]
            if "_" not in remainder:
                continue
            size_str, metric = remainder.split("_", 1)
            if not size_str.isdigit():
                continue
            k = int(size_str)
            bucket = grouped.setdefault(k, {})
            if metric == "accuracy":
                bucket["accuracy"] = value
        return grouped
