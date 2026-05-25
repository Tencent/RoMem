"""
Utilities for persisting benchmark results to disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from benchmarks.types import BenchmarkResult


class ExperimentLogger:
    def __init__(self, dataset: str, exp_name: str, base_dir: str | Path = 'benchmarks/reports'):
        self.dataset = dataset
        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / dataset / exp_name
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.run_dir = self.exp_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.run_dir / 'results.jsonl'
        self.details_path = self.run_dir / 'details.jsonl'

    def log(self, result: BenchmarkResult) -> None:
        payload = asdict(result)
        if payload.get('window_id') is None:
            del payload['window_id']
        payload['timestamp'] = datetime.utcnow().isoformat()
        with self.results_path.open('a') as f:
            json.dump(payload, f)
            f.write('\n')

    def log_detail(self, detail: dict) -> None:
        """Log a per-item detail record (predictions, gold answers, etc.)."""
        detail['timestamp'] = datetime.utcnow().isoformat()
        with self.details_path.open('a') as f:
            json.dump(detail, f, ensure_ascii=False)
            f.write('\n')
