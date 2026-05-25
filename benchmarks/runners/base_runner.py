"""
Shared runner base classes for benchmarks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BenchmarkRunnerBase(ABC):
    def __init__(self, dataset: str, exp_name: str | None = None):
        self.dataset = dataset
        self.exp_name = exp_name

    @abstractmethod
    async def run(self):
        """Execute the benchmark."""
        raise NotImplementedError
