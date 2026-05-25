"""
Loader for the LoCoMo QA benchmark (official evaluation format).
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Iterable

from benchmarks.types import EpisodePayload, LocomoExample, LocomoQA, LocomoSample

DEFAULT_DATA_PATH = Path('dataset/locomo/code/locomo/data/locomo10.json')
logger = logging.getLogger(__name__)


@dataclass
class LocomoLoader:
    data_path: Path = DEFAULT_DATA_PATH
    max_examples: int | None = None
    question_type: str | None = None
    question_type_map_path: Path | None = None
    _cached_data: list[dict] | None = None
    _is_mc: bool | None = None
    _qa_question_type_map: dict[tuple[str, str], str] | None = None
    _qa_question_type_fallback: dict[str, str] | None = None

    def __init__(
        self,
        data_path: str | Path | None = None,
        max_examples: int | None = None,
        question_type: str | None = None,
        question_type_map_path: str | Path | None = None,
    ):
        resolved_path = Path(
            data_path
            or os.getenv('LOCOMO_DATA_PATH')
            or DEFAULT_DATA_PATH
        )
        if not resolved_path.exists():
            raise FileNotFoundError(f'LoCoMo data file not found: {resolved_path}')
        self.data_path = resolved_path
        self.max_examples = None if max_examples == 0 else max_examples
        self.question_type = question_type.strip() if isinstance(question_type, str) and question_type.strip() else None
        map_path = (
            question_type_map_path
            or os.getenv('LOCOMO_QT_MAP_PATH')
            or os.getenv('LOCOMO_MC_PATH')
            or Path('dataset/locomo/data/locomo_mc10.json')
        )
        self.question_type_map_path = Path(map_path) if map_path else None
        self._cached_data = None
        self._is_mc = None
        self._qa_question_type_map = None
        self._qa_question_type_fallback = None

    def iter_samples(self) -> Iterable[LocomoSample]:
        data = self._load_dataset()
        if self.is_mc:
            return iter(())
        count = 0
        for item in data:
            if self.max_examples is not None and count >= self.max_examples:
                break
            sample = self._build_sample(item)
            if not sample.qa:
                continue
            yield sample
            count += 1

    def iter_examples(self) -> Iterable[LocomoSample]:
        # Backwards-compatible alias.
        yield from self.iter_samples()

    def iter_mc_examples(self) -> Iterable[LocomoExample]:
        data = self._load_dataset()
        if not self.is_mc:
            return iter(())
        count = 0
        for item in data:
            if self.max_examples is not None and count >= self.max_examples:
                break
            if not self._should_include_mc(item):
                continue
            example = self._build_mc_example(item)
            if example is None:
                continue
            yield example
            count += 1

    def _load_dataset(self) -> list[dict]:
        if self._cached_data is not None:
            return self._cached_data
        items: list[dict] = []
        with self.data_path.open(encoding='utf-8') as f:
            probe = f.read(4096)
            first_nonspace = next((ch for ch in probe if not ch.isspace()), '')
            f.seek(0)
            if first_nonspace == '[':
                data = json.load(f)
                if isinstance(data, list):
                    items = data
            else:
                for line_num, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning('Skipping malformed LoCoMo record at line %s.', line_num)
        self._cached_data = items
        self._is_mc = self._detect_mc_format(items)
        return items

    @property
    def is_mc(self) -> bool:
        if self._is_mc is None:
            self._load_dataset()
        return bool(self._is_mc)

    @property
    def sample_count(self) -> int:
        data = self._load_dataset()
        total = len(data)
        if self.max_examples is not None:
            total = min(total, self.max_examples)
        return total

    def _detect_mc_format(self, data: list[dict]) -> bool:
        if not data:
            return False
        sample = data[0]
        return isinstance(sample, dict) and 'choices' in sample and 'question_id' in sample

    # Map known question_type names to the LoCoMo category numbers that can
    # contain them.  Category 3 holds *both* temporal_reasoning and
    # compound_answer, so a coarse category filter alone is not enough — we
    # still need the resolved string for a fine-grained match.
    _QUESTION_TYPE_TO_CATEGORIES: ClassVar[dict[str, set[int]]] = {
        'single_hop': {1},
        'multi_hop': {2},
        'temporal_reasoning': {3},
        'compound_answer': {3},
        'open_domain': {4},
        'adversarial': {5},
    }

    def _normalize_question_type_filter(self) -> str | None:
        if not self.question_type:
            return None
        raw = self.question_type.strip().lower().replace('-', '_').replace(' ', '_')
        return raw or None

    def _coarse_category_match(self, category: int) -> bool:
        """Quick pre-filter: does the QA's category *possibly* match the
        requested question_type?  Returns True when unsure (no mapping)."""
        target = self._normalize_question_type_filter()
        if not target:
            return True
        cats = self._QUESTION_TYPE_TO_CATEGORIES.get(target)
        if cats is None:
            # Unknown type name — can't exclude by category.
            return True
        return category in cats

    def _matches_question_type(self, resolved_type: str | None) -> bool:
        """Fine-grained string match after question_type has been resolved."""
        target = self._normalize_question_type_filter()
        if not target:
            return True
        if not resolved_type:
            # Could not resolve — exclude when a specific type is requested.
            return False
        qt = resolved_type.strip().lower().replace('-', '_').replace(' ', '_')
        return qt == target

    def _should_include_mc(self, item: dict) -> bool:
        if not self.question_type:
            return True
        raw = (item.get('question_type') or '').strip().lower().replace('-', '_').replace(' ', '_')
        target = self.question_type.lower().replace('-', '_').replace(' ', '_')
        return raw == target

    def _build_sample(self, item: dict) -> LocomoSample:
        conversation = item.get('conversation', {})
        episodes = self._build_episodes(item)
        sample_id = str(item.get('sample_id', 'locomo'))
        qa_items: list[LocomoQA] = []
        for qa in item.get('qa', []):
            category = int(qa.get('category', 0))
            # Stage 1: coarse category pre-filter (cheap, no lookup needed).
            if not self._coarse_category_match(category):
                continue
            # Resolve question_type string (may need lookup from MC file).
            question_type = qa.get('question_type')
            if isinstance(question_type, str):
                question_type = question_type.strip()
            else:
                question_type = self._lookup_question_type(
                    sample_id,
                    str(qa.get('question', '')),
                )
            # Stage 2: fine-grained string match on resolved question_type.
            if not self._matches_question_type(question_type):
                continue
            qa_items.append(
                LocomoQA(
                    question=qa.get('question', ''),
                    answer=str(qa.get('answer', '')),
                    evidence=[str(ev) for ev in qa.get('evidence', []) if ev],
                    category=category,
                    question_type=question_type or None,
                )
            )
        return LocomoSample(
            sample_id=sample_id,
            conversation=conversation,
            qa=qa_items,
            episodes=episodes,
        )

    def _build_mc_example(self, item: dict) -> LocomoExample | None:
        question = (item.get('question') or '').strip()
        choices = item.get('choices') or []
        if not question or not isinstance(choices, list) or not choices:
            return None
        episodes = self._build_mc_episodes(item)
        return LocomoExample(
            question_id=str(item.get('question_id', 'locomo')),
            question=question,
            choices=[str(choice) for choice in choices],
            correct_choice_index=int(item.get('correct_choice_index', -1)),
            question_type=str(item.get('question_type', '')),
            episodes=episodes,
        )

    @staticmethod
    def _normalize_question(text: str) -> str:
        return ' '.join(text.lower().split())

    def _lookup_question_type(self, sample_id: str, question: str) -> str | None:
        self._load_question_type_map()
        if not self._qa_question_type_map:
            return None
        norm = self._normalize_question(question)
        return (
            self._qa_question_type_map.get((sample_id, norm))
            or (self._qa_question_type_fallback or {}).get(norm)
        )

    def _load_question_type_map(self) -> None:
        if self._qa_question_type_map is not None:
            return
        self._qa_question_type_map = {}
        self._qa_question_type_fallback = {}
        path = self.question_type_map_path
        if not path or not path.exists():
            return
        try:
            with path.open(encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    question = (item.get('question') or '').strip()
                    question_type = (item.get('question_type') or '').strip()
                    question_id = (item.get('question_id') or '').strip()
                    if not question or not question_type or not question_id:
                        continue
                    conv_id = question_id.split('_q', 1)[0]
                    norm = self._normalize_question(question)
                    self._qa_question_type_map[(conv_id, norm)] = question_type
                    if norm not in self._qa_question_type_fallback:
                        self._qa_question_type_fallback[norm] = question_type
        except OSError:
            return

    def _build_episodes(self, item: dict) -> list[EpisodePayload]:
        conversation = item.get('conversation', {})
        episodes: list[EpisodePayload] = []
        session_summary = item.get('session_summary') or {}
        if isinstance(session_summary, dict) and session_summary:
            for idx in range(1, 50):
                key = f'session_{idx}_summary'
                summary = session_summary.get(key)
                if not isinstance(summary, str) or not summary.strip():
                    continue
                date_key = f'session_{idx}_date_time'
                date_text = conversation.get(date_key, '')
                timestamp = self._parse_session_time(date_text)
                content = summary.strip()
                if date_text:
                    content = f'({date_text})\n{content}'
                metadata = {
                    'session_id': f'S{idx}',
                    'session_date': date_text,
                    'source': 'session_summary',
                }
                episodes.append(
                    EpisodePayload(
                        content=content,
                        reference_time=timestamp,
                        metadata=metadata,
                    )
                )
            if episodes:
                return episodes
        event_summary = item.get('event_summary') or {}
        if isinstance(event_summary, dict) and event_summary:
            for idx in range(1, 50):
                key = f'events_session_{idx}'
                session = event_summary.get(key)
                if not isinstance(session, dict):
                    continue
                date_text = session.get('date', '') if isinstance(session.get('date', ''), str) else ''
                timestamp = self._parse_event_time(date_text) or datetime.utcnow()
                lines: list[str] = []
                for speaker, events in session.items():
                    if speaker == 'date':
                        continue
                    if not events:
                        continue
                    for event in events:
                        if not event:
                            continue
                        lines.append(f'{speaker}: {event}'.strip())
                if not lines:
                    continue
                content = '\n'.join(lines)
                if date_text:
                    content = f'({date_text})\n{content}'
                metadata = {
                    'session_id': f'S{idx}',
                    'session_date': date_text,
                    'source': 'events',
                }
                episodes.append(
                    EpisodePayload(
                        content=content,
                        reference_time=timestamp,
                        metadata=metadata,
                    )
                )
            if episodes:
                return episodes
        haystack_summaries = item.get('haystack_session_summaries') or []
        haystack_dates = item.get('haystack_session_datetimes') or []
        if haystack_summaries:
            for idx, summary in enumerate(haystack_summaries, start=1):
                if not summary:
                    continue
                date_text = ''
                timestamp = None
                if idx - 1 < len(haystack_dates):
                    date_text = haystack_dates[idx - 1] or ''
                    timestamp = self._parse_iso_time(date_text)
                if timestamp is None:
                    timestamp = datetime.utcnow()
                content = summary.strip()
                if date_text:
                    content = f'({date_text})\n{content}'
                metadata = {
                    'session_id': f'S{idx}',
                    'session_date': date_text,
                    'source': 'summary',
                }
                episodes.append(
                    EpisodePayload(
                        content=content,
                        reference_time=timestamp,
                        metadata=metadata,
                    )
                )
            if episodes:
                return episodes
        for idx in range(1, 50):
            key = f'session_{idx}'
            if key not in conversation or not conversation.get(key):
                continue
            date_key = f'{key}_date_time'
            date_text = conversation.get(date_key, '')
            timestamp = self._parse_session_time(date_text)
            turns = []
            for turn in conversation.get(key, []):
                speaker = turn.get('speaker', '')
                text = turn.get('text') or turn.get('compressed_text') or ''
                if not text:
                    continue
                line = f'{speaker}: {text}'.strip()
                if turn.get('img_file'):
                    caption = turn.get('blip_caption', 'an image')
                    line += f' [shares {caption}]'
                turns.append(line)
            if not turns:
                continue
            content = '\n'.join(turns)
            if date_text:
                content = f'({date_text})\n{content}'
            metadata = {
                'session_id': f'S{idx}',
                'session_date': date_text,
            }
            episodes.append(
                EpisodePayload(
                    content=content,
                    reference_time=timestamp,
                    metadata=metadata,
                )
                )
        return episodes

    def _build_mc_episodes(self, item: dict) -> list[EpisodePayload]:
        episodes: list[EpisodePayload] = []
        summaries = item.get('haystack_session_summaries') or []
        datetimes = item.get('haystack_session_datetimes') or []
        for idx, summary in enumerate(summaries, start=1):
            if not summary:
                continue
            date_text = datetimes[idx - 1] if idx - 1 < len(datetimes) else ''
            timestamp = self._parse_iso_time(date_text) or datetime.utcnow()
            content = summary.strip()
            if date_text:
                content = f'({date_text})\n{content}'
            metadata = {
                'session_id': f'S{idx}',
                'session_date': date_text,
                'source': 'summary',
            }
            episodes.append(
                EpisodePayload(
                    content=content,
                    reference_time=timestamp,
                    metadata=metadata,
                )
            )
        return episodes

    def _parse_session_time(self, date_text: str) -> datetime:
        if not date_text:
            return datetime.utcnow()
        for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
            try:
                return datetime.strptime(date_text, fmt)
            except ValueError:
                continue
        return datetime.utcnow()

    def _parse_iso_time(self, date_text: str) -> datetime | None:
        if not date_text:
            return None
        try:
            normalized = date_text.replace('Z', '+00:00')
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def _parse_event_time(self, date_text: str) -> datetime | None:
        if not date_text:
            return None
        cleaned = date_text.replace(',', '').strip()
        for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None
