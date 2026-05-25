"""
Loader for the MultiTQ temporal knowledge graph QA dataset.

Data layout (dataset/MultiTQ/):
  full_fixed.txt   – all 461K KG triples  (head\trelation\ttail\tdate)
  test.json        – 54K test questions
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List

from benchmarks.types import TemporalTriple, MultiTQQuestion

DEFAULT_ROOT = Path("dataset/MultiTQ")


class MultiTQLoader:
    def __init__(
        self,
        root: str | Path | None = None,
        eval_split: str = "test",
        max_time_ids: int | None = None,
        max_examples: int | None = None,
    ) -> None:
        self.root = Path(root or DEFAULT_ROOT)
        if not self.root.exists():
            raise FileNotFoundError(f"MultiTQ root not found: {self.root}")
        self.eval_split = eval_split
        self.max_time_ids = max_time_ids
        self.max_examples = max_examples

        kg_path = self.root / "full_fixed.txt"
        if not kg_path.exists():
            raise FileNotFoundError(f"KG file not found: {kg_path}")

        self._date_map, self.time_map = self._build_time_index(kg_path)
        self._all_by_time = self._load_triples(kg_path)
        self._all_time_ids = sorted(self._all_by_time.keys())

    # ── KG triple loading ─────────────────────────────────────────

    def _build_time_index(
        self, path: Path
    ) -> tuple[Dict[str, int], Dict[int, datetime]]:
        dates: dict[str, datetime] = {}
        with path.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                date_str = parts[3].strip()
                if not date_str or date_str in dates:
                    continue
                try:
                    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                dates[date_str] = dt
        ordered = sorted(dates.items(), key=lambda item: item[1])
        date_map: Dict[str, int] = {}
        time_map: Dict[int, datetime] = {}
        for idx, (date_str, dt) in enumerate(ordered):
            date_map[date_str] = idx
            time_map[idx] = dt
        return date_map, time_map

    def _load_triples(self, path: Path) -> DefaultDict[int, List[TemporalTriple]]:
        by_time: DefaultDict[int, List[TemporalTriple]] = defaultdict(list)
        with path.open() as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                head, relation, tail, date_str = (p.strip() for p in parts[:4])
                if not head or not relation or not tail or not date_str:
                    continue
                time_id = self._date_map.get(date_str)
                if time_id is None:
                    continue
                timestamp = self.time_map.get(time_id)
                if timestamp is None:
                    continue
                triple = TemporalTriple(
                    head=head.replace("_", " "),
                    relation=relation.replace("_", " "),
                    tail=tail.replace("_", " "),
                    time_id=time_id,
                    timestamp=timestamp,
                )
                by_time[time_id].append(triple)
        return by_time

    def iter_time_ids(self) -> List[int]:
        limit = self._effective_limit_env("MULTITQ_MAX_TIMES")
        if limit is not None:
            return self._all_time_ids[:limit]
        return list(self._all_time_ids)

    def get_all_triples(self, time_id: int) -> List[TemporalTriple]:
        return self._all_by_time.get(time_id, [])

    def iter_all_triples(self) -> List[TemporalTriple]:
        """Return all KG triples across all time ids as a flat list."""
        time_ids = self.iter_time_ids()
        all_triples: List[TemporalTriple] = []
        for tid in time_ids:
            all_triples.extend(self._all_by_time.get(tid, []))
        return all_triples

    def format_window_label(self, time_id: int) -> str:
        timestamp = self.time_map.get(time_id)
        return timestamp.strftime("%Y-%m-%d") if timestamp else f"time_{time_id}"

    # ── Question loading ──────────────────────────────────────────

    def iter_questions(self) -> Iterator[MultiTQQuestion]:
        question_file = self.root / f"{self.eval_split}.json"
        if not question_file.exists():
            # Fallback: try questions/ subdirectory
            question_file = self.root / "questions" / f"{self.eval_split}.json"
        if not question_file.exists():
            raise FileNotFoundError(f"Question file not found: {question_file}")

        with question_file.open() as f:
            data = json.load(f)

        count = 0
        for item in data:
            if self.max_examples is not None and count >= self.max_examples:
                break
            answers_raw = item.get("answers") or item.get("answer") or []
            if isinstance(answers_raw, str):
                answers_raw = [answers_raw]
            answers = [str(a).replace("_", " ") for a in answers_raw]

            question = MultiTQQuestion(
                quid=item.get("quid", count),
                question=item.get("question", ""),
                answers=answers,
                answer_type=item.get("answer_type", "entity"),
                time_level=item.get("time_level", "year"),
                qtype=item.get("qtype", "unknown"),
                qlabel=item.get("qlabel", "Single"),
            )
            yield question
            count += 1

    # ── Helpers ───────────────────────────────────────────────────

    def _effective_limit_env(self, env_var: str) -> int | None:
        if self.max_time_ids is not None:
            return self.max_time_ids
        env_val = int(os.getenv(env_var, "0"))
        return env_val if env_val > 0 else None
