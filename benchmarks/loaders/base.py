"""
Abstract loader interface for benchmark datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterable

from benchmarks.types import ProbeSample, SnapshotBatch


class SnapshotLoader(ABC):
    @abstractmethod
    async def iter_snapshots(self) -> AsyncIterator[SnapshotBatch]:
        """Yield SnapshotBatch objects in chronological order."""

    @abstractmethod
    def iter_probes(self, window_id: str) -> Iterable[ProbeSample]:
        """Return probe samples for a given window identifier."""

    @property
    @abstractmethod
    def window_ids(self) -> list[str]:
        """Chronological list of probe window identifiers."""
