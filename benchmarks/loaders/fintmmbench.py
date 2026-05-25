"""
Loader for the FinTMMBench temporal financial QA benchmark.

Dataset: 5,676 QA pairs across NASDAQ-100 companies spanning 4 modalities:
  - FinancialTable: quarterly/annual financial indicators
  - StockPrice: daily stock prices
  - News: Reuters financial news articles
  - Chart: technical stock charts (skipped — requires vision)

Each question references source documents via UUIDs and includes temporal context
(specific dates or date ranges).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from benchmarks.types import EpisodePayload, FinTMMBenchExample

DEFAULT_DATA_DIR = Path('dataset/FinTMMBench')

# Chart questions require image understanding — skip by default
SKIP_TYPES = frozenset({'Chart'})


@dataclass
class FinTMMBenchLoader:
    data_dir: Path
    max_examples: int | None
    skip_types: frozenset[str]
    sampled_tag: str | None

    def __init__(
        self,
        data_dir: str | Path | None = None,
        max_examples: int | None = None,
        skip_types: frozenset[str] | None = None,
        sampled_tag: str | None = None,
    ):
        resolved = Path(
            data_dir
            or os.getenv('FINTMMBENCH_DATA_DIR')
            or DEFAULT_DATA_DIR
        )
        if not resolved.exists():
            raise FileNotFoundError(f'FinTMMBench data directory not found: {resolved}')
        self.data_dir = resolved
        env_limit = os.getenv('FINTMMBENCH_MAX_EXAMPLES')
        self.max_examples = max_examples or (int(env_limit) if env_limit else None)
        self.skip_types = skip_types if skip_types is not None else SKIP_TYPES
        self.sampled_tag = sampled_tag or os.getenv('FINTMMBENCH_SAMPLED_TAG')
        self._qa_cache: list[dict] | None = None
        self._corpus_cache: dict[str, dict] | None = None

    def _load_qa(self) -> list[dict]:
        if self._qa_cache is not None:
            return self._qa_cache
        if self.sampled_tag:
            qa_path = self.data_dir / 'sampled' / f'QA_{self.sampled_tag}.json'
        else:
            qa_path = self.data_dir / 'QA.json'
        with qa_path.open() as f:
            self._qa_cache = json.load(f)
        return self._qa_cache

    def _load_corpus(self) -> dict[str, dict]:
        """Load all corpus documents keyed by UUID."""
        if self._corpus_cache is not None:
            return self._corpus_cache
        if self.sampled_tag:
            return self._load_sampled_corpus()
        corpus: dict[str, dict] = {}
        for name in ('News.json', 'FinancialTable.json', 'StockPrice.json'):
            path = self.data_dir / 'data' / name
            if not path.exists():
                continue
            with path.open() as f:
                items = json.load(f)
            for item in items:
                uid = item.get('uuid', '')
                if uid:
                    corpus[uid] = item
        self._corpus_cache = corpus
        return corpus

    def _load_sampled_corpus(self) -> dict[str, dict]:
        """Load pre-sampled corpus from sampled/corpus_{tag}.json."""
        corpus_path = self.data_dir / 'sampled' / f'corpus_{self.sampled_tag}.json'
        with corpus_path.open() as f:
            items = json.load(f)
        corpus = {}
        for item in items:
            uid = item.get('uuid', '')
            if uid:
                corpus[uid] = item
        self._corpus_cache = corpus
        return corpus

    @property
    def sample_count(self) -> int:
        qa = self._load_qa()
        filtered = [e for e in qa if e.get('type') not in self.skip_types]
        total = len(filtered)
        if self.max_examples and self.max_examples > 0:
            return min(total, self.max_examples)
        return total

    def iter_examples(self) -> Iterable[FinTMMBenchExample]:
        qa = self._load_qa()
        corpus = self._load_corpus()
        count = 0
        for item in qa:
            if item.get('type') in self.skip_types:
                continue
            if self.max_examples and self.max_examples > 0 and count >= self.max_examples:
                break
            yield self._build_example(item, corpus)
            count += 1

    def _build_example(self, item: dict, corpus: dict[str, dict]) -> FinTMMBenchExample:
        answers_list = item.get('answers', [])
        first_answer = answers_list[0] if answers_list else {}
        answer_text = first_answer.get('answer', '')
        explanation = first_answer.get('explanation', '')
        source_ids = first_answer.get('source', [])

        # Build episodes from source documents
        episodes = self._build_episodes(source_ids, corpus)

        return FinTMMBenchExample(
            uuid=item.get('uuid', ''),
            question=item.get('question', ''),
            answer=answer_text,
            explanation=explanation,
            source_ids=source_ids,
            question_type=item.get('type', ''),
            subtasks=item.get('subtask', []),
            episodes=episodes,
        )

    @staticmethod
    def _build_episodes(source_ids: list[str], corpus: dict[str, dict]) -> list[EpisodePayload]:
        episodes: list[EpisodePayload] = []
        for sid in source_ids:
            doc = corpus.get(sid)
            if not doc:
                continue
            content = _doc_to_text(doc)
            if not content:
                continue
            ref_time = _parse_date(doc.get('Date', ''))
            episodes.append(EpisodePayload(
                content=content,
                reference_time=ref_time,
                metadata={
                    'uuid': sid,
                    'type': doc.get('type', ''),
                    'company': doc.get('Company', ''),
                    'symbol': doc.get('Symbol', ''),
                },
            ))
        return episodes

    def get_corpus(self) -> dict[str, dict]:
        return self._load_corpus()


# Roughly 4 chars per token; keep well under the 8192-token embedding limit.
_MAX_TEXT_CHARS = 28_000


def _doc_to_text(doc: dict) -> str:
    """Convert a corpus document to a plain-text representation."""
    doc_type = doc.get('type', '')
    company = doc.get('Company', '')
    symbol = doc.get('Symbol', '')
    date = doc.get('Date', '')

    if doc_type == 'News':
        text = doc.get('Text', '')
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + '...'
        return f"[News] {date} | {company} ({symbol}): {text}"

    if doc_type == 'FinancialTable':
        indicator = doc.get('indicator_name', '')
        value = doc.get('indicator_value', '')
        unit = doc.get('unit', '')
        interval = doc.get('Interval', '')
        return (
            f"[FinancialTable] {date} | {company} ({symbol}) | "
            f"{interval} {indicator} = {value} {unit}"
        )

    if doc_type == 'StockPrice':
        indicator = doc.get('indicator_name', '')
        value = doc.get('indicator_value', '')
        unit = doc.get('unit', '')
        return (
            f"[StockPrice] {date} | {company} ({symbol}) | "
            f"{indicator} = {value} {unit}"
        )

    # Fallback
    return json.dumps(doc, default=str)


def _parse_date(date_str: str) -> datetime:
    if not date_str:
        return datetime(2022, 1, 1)
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime(2022, 1, 1)
