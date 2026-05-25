"""
Loader for the DMR-MSC conversational benchmark.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.types import DmrMscExample, EpisodePayload

DEFAULT_DATA_PATH = Path('dataset/dmr_msc/msc_self_instruct.jsonl')
TIME_BACK_PATTERN = re.compile(r'(\\d+)\\s+(day|days|hour|hours)', re.IGNORECASE)


def _flatten_statements(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        items: list[str] = []
        for entry in value:
            items.extend(_flatten_statements(entry))
        return items
    return []


def _normalize_statements(statements: Any) -> list[str]:
    flat = _flatten_statements(statements)
    return [stmt.strip() for stmt in flat if stmt and stmt.strip()]


def _format_persona(statements: Sequence[str], speaker_label: str, prefix: str) -> str:
    normalized = _normalize_statements(statements)
    if not normalized:
        return ''
    lines = [f'{speaker_label}: {statement}' for statement in normalized]
    return f'{prefix} for {speaker_label}:\n' + '\n'.join(lines)


def _format_dialog(turns: Sequence[dict], default_speakers: tuple[str, str] = ('Speaker 1', 'Speaker 2')) -> str:
    lines: list[str] = []
    for idx, turn in enumerate(turns):
        text = (turn.get('text') or '').strip()
        if not text:
            continue
        speaker = turn.get('id') or default_speakers[idx % len(default_speakers)]
        lines.append(f'{speaker}: {text}')
    return '\n'.join(lines)


def _parse_time_back(value: str | None) -> timedelta:
    if not value:
        return timedelta()
    total = timedelta()
    for number, unit in TIME_BACK_PATTERN.findall(value):
        count = int(number)
        if 'day' in unit.lower():
            total += timedelta(days=count)
        else:
            total += timedelta(hours=count)
    return total


@dataclass
class DmrMscLoader:
    data_path: Path
    max_examples: int | None

    def __init__(self, data_path: str | Path | None = None, max_examples: int | None = None):
        resolved_path = Path(
            data_path
            or os.getenv('DMR_MSC_DATA_PATH')
            or DEFAULT_DATA_PATH
        )
        if not resolved_path.exists():
            raise FileNotFoundError(f'DMR-MSC data file not found: {resolved_path}')
        self.data_path = resolved_path
        env_limit = os.getenv('DMR_MSC_MAX_EXAMPLES')
        self.max_examples = max_examples or (int(env_limit) if env_limit else None)

    @property
    def sample_count(self) -> int:
        """Return the total number of examples that will be yielded."""
        total = sum(1 for line in self.data_path.open() if line.strip())
        if self.max_examples is not None and self.max_examples > 0:
            return min(total, self.max_examples)
        return total

    def iter_examples(self) -> Iterable[DmrMscExample]:
        count = 0
        with self.data_path.open() as f:
            for row_index, line in enumerate(f):
                if not line.strip():
                    continue
                if self.max_examples is not None and count >= self.max_examples:
                    break
                payload = json.loads(line)
                yield self._build_example(payload, row_index)
                count += 1

    def _build_example(self, item: dict, row_index: int) -> DmrMscExample:
        metadata = item.get('metadata', {})
        source_id = metadata.get('initial_data_id', f'dmr_{row_index}')
        example_id = f'{source_id}_sample_{row_index}'
        base_time = datetime.utcnow()
        episodes: list[EpisodePayload] = []

        def add_episode(content: str, reference_time: datetime, meta: dict[str, str]) -> None:
            content = (content or '').strip()
            if not content:
                return
            episode_meta = {
                'example_id': example_id,
                **meta,
            }
            episodes.append(
                EpisodePayload(
                    content=content,
                    reference_time=reference_time,
                    metadata=episode_meta,
                )
            )

        init_personas = item.get('init_personas') or item.get('personas') or []
        for speaker_idx, statements in enumerate(init_personas, start=1):
            persona_content = _format_persona(
                statements,
                speaker_label=f'Speaker {speaker_idx}',
                prefix='Initial persona',
            )
            add_episode(
                persona_content,
                base_time - timedelta(days=60 + (len(init_personas) - speaker_idx)),
                {
                    'type': 'persona',
                    'segment': 'initial',
                    'speaker': f'Speaker {speaker_idx}',
                },
            )

        previous_dialogs = item.get('previous_dialogs', [])
        ordered_previous = sorted(
            enumerate(previous_dialogs),
            key=lambda pair: _parse_time_back(pair[1].get('time_back')),
            reverse=True,
        )
        for prev_index, prev in ordered_previous:
            offset = _parse_time_back(prev.get('time_back'))
            if offset == timedelta():
                offset = timedelta(days=14 + prev_index)
            reference_time = base_time - offset
            personas_snapshot = prev.get('personas') or []
            for speaker_idx, statements in enumerate(personas_snapshot, start=1):
                add_episode(
                    _format_persona(
                        statements,
                        speaker_label=f'Speaker {speaker_idx}',
                        prefix='Historical persona',
                    ),
                    reference_time - timedelta(hours=1),
                    {
                        'type': 'persona',
                        'segment': 'history',
                        'speaker': f'Speaker {speaker_idx}',
                        'history_index': str(prev_index),
                    },
                )
            add_episode(
                _format_dialog(prev.get('dialog', [])),
                reference_time,
                {
                    'type': 'dialog',
                    'segment': 'history',
                    'history_index': str(prev_index),
                    'time_back': prev.get('time_back', ''),
                },
            )

        persona_updates = [
            ('personas_update1', 'Speaker 1'),
            ('personas_update2', 'Speaker 2'),
        ]
        for field, speaker in persona_updates:
            statements = _normalize_statements(item.get(field, []))
            if not statements:
                continue
            add_episode(
                _format_persona(statements, speaker_label=speaker, prefix='Persona update'),
                base_time - timedelta(hours=2),
                {
                    'type': 'persona',
                    'segment': 'update',
                    'speaker': speaker,
                },
            )

        summaries = [
            ('summary_speaker_1', 'Speaker 1'),
            ('summary_speaker_2', 'Speaker 2'),
        ]
        for field, speaker in summaries:
            summary_lines = _normalize_statements(item.get(field, []))
            if not summary_lines:
                continue
            summary_text = '\n'.join(summary_lines)
            add_episode(
                f'Summary for {speaker}:\n{summary_text}',
                base_time - timedelta(hours=1),
                {
                    'type': 'summary',
                    'segment': 'current',
                    'speaker': speaker,
                },
            )

        add_episode(
            _format_dialog(item.get('dialog', [])),
            base_time,
            {
                'type': 'dialog',
                'segment': 'current',
            },
        )

        question = (item.get('self_instruct', {}).get('B') or '').strip()
        answer = (item.get('self_instruct', {}).get('A') or '').strip()

        return DmrMscExample(
            example_id=example_id,
            question=question,
            answer=answer,
            episodes=episodes,
            window_id=str(source_id),
            metadata={
                'initial_data_id': metadata.get('initial_data_id'),
                'session_id': metadata.get('session_id'),
                'row_index': row_index,
            },
        )
