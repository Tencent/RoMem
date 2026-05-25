"""
Shared dataclasses and type aliases for benchmark pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class EpisodePayload:
    """
    Canonical representation of an episode to ingest into a knowledge graph.
    """

    content: str
    reference_time: datetime
    metadata: dict[str, Any]


@dataclass
class ProbeSample:
    """
    Test triple used for evaluation.
    """

    window_id: str
    subject: str
    relation: str
    object: str
    changed: bool


@dataclass
class SnapshotBatch:
    """
    A batch of episodes to ingest for a given snapshot/timestamp.
    """

    label: str
    episodes: list[EpisodePayload]


@dataclass
class BenchmarkResult:
    """
    Metrics emitted by evaluators for a particular probe window.
    """

    dataset: str
    mode: str
    snapshot_label: str
    window_id: str | None
    metrics: dict[str, float]


@dataclass
class LocomoExample:
    """
    Represents a single LoCoMo multiple-choice question and its supporting episodes.
    """

    question_id: str
    question: str
    choices: list[str]
    correct_choice_index: int
    question_type: str
    episodes: list[EpisodePayload]


@dataclass
class LocomoQA:
    """
    Represents a single LoCoMo QA item from the official dataset.
    """

    question: str
    answer: str
    evidence: list[str]
    category: int
    question_type: str | None = None


@dataclass
class LocomoSample:
    """
    Represents a LoCoMo conversation sample with multiple QA items.
    """

    sample_id: str
    conversation: dict[str, Any]
    qa: list[LocomoQA]
    episodes: list[EpisodePayload]


@dataclass
class DmrMscExample:
    """
    Represents a DMR-MSC conversation sample with persona/context episodes.
    """

    example_id: str
    question: str
    answer: str
    episodes: list[EpisodePayload]
    window_id: str
    metadata: dict[str, Any]


@dataclass
class LongmemevalExample:
    """
    Represents a LongMemEval QA sample (oracle subset).
    """

    question_id: str
    question: str
    answer: str
    question_type: str
    episodes: list[EpisodePayload]
    metadata: dict[str, Any]


@dataclass
class TemporalTriple:
    """
    A single (head, relation, tail, time) triple from a temporal knowledge graph.
    """

    head: str
    relation: str
    tail: str
    time_id: int
    timestamp: datetime | None = None


@dataclass
class FinTMMBenchExample:
    """
    Represents a single FinTMMBench temporal financial QA example.
    """

    uuid: str
    question: str
    answer: str
    explanation: str
    source_ids: list[str]
    question_type: str        # NewsSentiment, StockPrice, FinancialTable, NewsEvent, MultiPriceTable, Chart
    subtasks: list[str]       # Extraction, Calculation, Sentiment, Trend, etc.
    episodes: list[EpisodePayload]


@dataclass
class MultiTQQuestion:
    """
    Represents a single MultiTQ temporal KGQA question.
    """

    quid: int
    question: str
    answers: list[str]
    answer_type: str   # "entity" or "time"
    time_level: str    # "day", "month", "year"
    qtype: str         # "equal", "before_after", "after_first", etc.
    qlabel: str        # "Single" or "Multiple"
