"""
Evaluator for LoCoMo QA questions (official scoring).
"""

from __future__ import annotations

import logging
import string
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

from benchmarks.reports.logger import ExperimentLogger
from benchmarks.types import BenchmarkResult, LocomoExample, LocomoQA, LocomoSample

logger = logging.getLogger(__name__)
ps = PorterStemmer()

CATEGORY_LABELS = {
    1: 'single_hop',
    2: 'multi_hop',
    3: 'temporal_reasoning',
    4: 'open_domain',
    5: 'adversarial',
}


def normalize_answer(text: str) -> str:
    text = text.replace(',', '')
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in set(string.punctuation))
    for article in (' a ', ' an ', ' the ', ' and '):
        text = text.replace(article, ' ')
    return ' '.join(text.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    if not predictions or not ground_truths:
        return 0.0
    scores = []
    for gt in ground_truths:
        scores.append(max(f1_score(pred, gt) for pred in predictions))
    return sum(scores) / len(scores)


def score_answer(prediction: str, qa: LocomoQA) -> float:
    answer = qa.answer
    if qa.category == 3:
        answer = answer.split(';')[0].strip()
    if qa.category in (2, 3, 4):
        return f1_score(prediction, answer)
    if qa.category == 1:
        return f1_multi(prediction, answer)
    if qa.category == 5:
        if 'no information available' in prediction.lower() or 'not mentioned' in prediction.lower():
            return 1.0
        return 0.0
    return 0.0


def recall_from_context(evidence: list[str], context_ids: list[str]) -> float:
    if not evidence:
        return 1.0
    if not context_ids:
        return 0.0
    if context_ids[0].startswith('S'):
        sessions = [cid[1:] for cid in context_ids]
        hits = sum(ev.split(':')[0][1:] in sessions for ev in evidence)
        return hits / len(evidence)
    hits = sum(ev in context_ids for ev in evidence)
    return hits / len(evidence)


class LocomoEvaluator:
    def __init__(self, dataset: str, exp_name: str | None = None):
        self.dataset = dataset
        self.logger = ExperimentLogger(dataset, exp_name) if exp_name else None
        self.total = 0
        self.f1_sum = 0.0
        self.recall_sum = 0.0
        self.category_stats: dict[int, dict[str, float]] = defaultdict(
            lambda: {'count': 0.0, 'f1': 0.0, 'recall': 0.0}
        )
        self.qa_type_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {'count': 0.0, 'f1': 0.0, 'recall': 0.0}
        )
        self.mc_total = 0
        self.mc_correct = 0
        self.mc_type_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {'count': 0.0, 'accuracy': 0.0}
        )

    def record(
        self,
        sample: LocomoSample,
        qa: LocomoQA,
        prediction: str,
        context_ids: list[str],
        context_docs: list[str] | None = None,
    ) -> BenchmarkResult:
        f1 = score_answer(prediction or '', qa)
        recall = recall_from_context(qa.evidence, context_ids) if context_ids is not None else 1.0
        question_type = qa.question_type or CATEGORY_LABELS.get(qa.category)

        self.total += 1
        self.f1_sum += f1
        self.recall_sum += recall

        bucket = self.category_stats[qa.category]
        bucket['count'] += 1
        bucket['f1'] += f1
        bucket['recall'] += recall
        if question_type:
            type_bucket = self.qa_type_stats[question_type]
            type_bucket['count'] += 1
            type_bucket['f1'] += f1
            type_bucket['recall'] += recall

        metrics = {
            'f1': f1,
            'recall': recall,
            'category': float(qa.category),
        }
        result = BenchmarkResult(
            dataset=self.dataset,
            mode='qa',
            snapshot_label=sample.sample_id,
            window_id=question_type or f'cat_{qa.category}',
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            gold = qa.answer
            if qa.category == 3:
                gold = gold.split(';')[0].strip()
            detail: dict = {
                'sample_id': sample.sample_id,
                'question': qa.question,
                'question_type': question_type,
                'category': qa.category,
                'gold_answer': gold,
                'prediction': prediction or '',
                'evidence': qa.evidence,
                'context_ids': context_ids,
                'f1': round(f1, 4),
                'evidence_recall': round(recall, 4),
            }
            if context_docs is not None:
                detail['context_docs'] = context_docs
            self.logger.log_detail(detail)
        return result

    def record_mc(
        self,
        example: LocomoExample,
        predicted_index: int,
        context_ids: list[str] | None = None,
    ) -> BenchmarkResult:
        correct = int(predicted_index == example.correct_choice_index)
        self.mc_total += 1
        self.mc_correct += correct

        bucket = self.mc_type_stats[example.question_type or 'unknown']
        bucket['count'] += 1
        bucket['accuracy'] += correct

        metrics = {
            'accuracy': float(correct),
        }
        result = BenchmarkResult(
            dataset=self.dataset,
            mode='mc',
            snapshot_label=example.question_id,
            window_id=example.question_type or 'unknown',
            metrics=metrics,
        )
        if self.logger:
            self.logger.log(result)
            self.logger.log_detail({
                'question_id': example.question_id,
                'question': example.question,
                'question_type': example.question_type,
                'choices': example.choices,
                'correct_choice_index': example.correct_choice_index,
                'predicted_index': predicted_index,
                'correct': bool(correct),
                'context_ids': context_ids,
            })
        return result

    def log_summary(self) -> None:
        if not self.logger:
            return
        if self.total:
            aggregate = {
                'f1_mean': self.f1_sum / self.total,
                'recall_mean': self.recall_sum / self.total,
                'total_questions': self.total,
            }
            self.logger.log(
                BenchmarkResult(
                    dataset=self.dataset,
                    mode='summary',
                    snapshot_label='aggregate',
                    window_id='locomo',
                    metrics=aggregate,
                )
            )

            for category, stats in self.category_stats.items():
                total = stats.get('count', 0.0)
                if not total:
                    continue
                window_metrics = {
                    'f1_mean': stats['f1'] / total,
                    'recall_mean': stats['recall'] / total,
                    'total_questions': total,
                }
                self.logger.log(
                    BenchmarkResult(
                        dataset=self.dataset,
                        mode='summary',
                        snapshot_label='aggregate',
                        window_id=f'cat_{category}',
                        metrics=window_metrics,
                    )
                )
                label = CATEGORY_LABELS.get(category)
                if label and label not in self.qa_type_stats:
                    self.logger.log(
                        BenchmarkResult(
                            dataset=self.dataset,
                            mode='summary',
                            snapshot_label='aggregate',
                            window_id=label,
                            metrics=window_metrics,
                        )
                    )
            for question_type, stats in self.qa_type_stats.items():
                total = stats.get('count', 0.0)
                if not total:
                    continue
                window_metrics = {
                    'f1_mean': stats['f1'] / total,
                    'recall_mean': stats['recall'] / total,
                    'total_questions': total,
                }
                self.logger.log(
                    BenchmarkResult(
                        dataset=self.dataset,
                        mode='summary',
                        snapshot_label='aggregate',
                        window_id=question_type,
                        metrics=window_metrics,
                    )
                )

        if self.mc_total:
            mc_aggregate = {
                'accuracy_mean': self.mc_correct / self.mc_total,
                'total_questions': self.mc_total,
            }
            self.logger.log(
                BenchmarkResult(
                    dataset=self.dataset,
                    mode='summary',
                    snapshot_label='aggregate',
                    window_id='locomo_mc',
                    metrics=mc_aggregate,
                )
            )
            for question_type, stats in self.mc_type_stats.items():
                total = stats.get('count', 0.0)
                if not total:
                    continue
                window_metrics = {
                    'accuracy_mean': stats['accuracy'] / total,
                    'total_questions': total,
                }
                self.logger.log(
                    BenchmarkResult(
                        dataset=self.dataset,
                        mode='summary',
                        snapshot_label='aggregate',
                        window_id=question_type,
                        metrics=window_metrics,
                    )
                )
