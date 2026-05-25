"""
Loader for LongMemEval oracle dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from benchmarks.types import EpisodePayload, LongmemevalExample

DEFAULT_DATA_PATH = Path('dataset/longmemeval/longmemeval_oracle.json')
DATE_FORMAT = '%Y/%m/%d (%a) %H:%M'


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.utcnow().replace(tzinfo=timezone.utc)
    # Input like "2023/04/10 (Mon) 17:50"
    parsed = datetime.strptime(value, DATE_FORMAT)
    return parsed.replace(tzinfo=timezone.utc)


@dataclass
class LongmemevalLoader:
    data_path: Path
    max_examples: int | None

    def __init__(self, data_path: str | Path | None = None, max_examples: int | None = None):
        resolved_path = Path(
            data_path
            or os.getenv('LONGMEMEVAL_DATA_PATH')
            or DEFAULT_DATA_PATH
        )
        if not resolved_path.exists():
            raise FileNotFoundError(f'LongMemEval data file not found: {resolved_path}')
        self.data_path = resolved_path
        env_limit = os.getenv('LONGMEMEVAL_MAX_EXAMPLES')
        self.max_examples = max_examples or (int(env_limit) if env_limit else None)

    def iter_examples(self) -> Iterable[LongmemevalExample]:
        payload = json.loads(self.data_path.read_text())
        count = 0
        for item in payload:
            if self.max_examples is not None and count >= self.max_examples:
                break
            yield self._build_example(item)
            count += 1

    def _build_example(self, item: dict) -> LongmemevalExample:
        question_id = item.get('question_id', '')
        question_type = item.get('question_type', 'longmemeval')
        episodes: list[EpisodePayload] = []
        sessions = item.get('haystack_sessions', [])
        session_ids = item.get('haystack_session_ids', [])
        session_dates = item.get('haystack_dates', [])
        answer_session_ids = set(item.get('answer_session_ids') or [])

        for session_idx, session in enumerate(sessions):
            session_id = (
                session_ids[session_idx]
                if session_idx < len(session_ids)
                else f'{question_id}_session_{session_idx}'
            )
            timestamp = _parse_timestamp(
                session_dates[session_idx] if session_idx < len(session_dates) else ''
            )
            for message_idx, message in enumerate(session):
                role = message.get('role', 'user')
                content = message.get('content', '')
                has_answer = bool(message.get('has_answer'))
                text = f'{role}: {content}'
                reference_time = timestamp + timedelta(seconds=message_idx)
                episodes.append(
                    EpisodePayload(
                        content=text,
                        reference_time=reference_time,
                        metadata={
                            'question_id': question_id,
                            'question_type': question_type,
                            'session_id': session_id,
                            'has_answer': has_answer,
                            'is_answer_session': session_id in answer_session_ids,
                        },
                    )
                )

        return LongmemevalExample(
            question_id=question_id,
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            question_type=question_type,
            episodes=episodes,
            metadata={
                'question_date': item.get('question_date'),
                'answer_session_ids': list(answer_session_ids),
            },
        )
