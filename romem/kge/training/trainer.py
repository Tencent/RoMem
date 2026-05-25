"""
Trainer and tester wrappers adapted from TKGE.
"""

from __future__ import annotations

from ..utils import *
from ..model.model_process import DevBatchProcessor, TrainBatchProcessor


class Trainer:
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        self.train_processor = TrainBatchProcessor(args, kg)
        self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self):
        self.args.valid = True
        loss = self.train_processor.process_epoch(self.model, self.optimizer)
        res = self.valid_processor.process_epoch(self.model)
        self.args.valid = False
        return loss, res


class Tester:
    def __init__(self, args, kg, model) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = DevBatchProcessor(args, kg)

    def test(self):
        self.args.valid = False
        return self.test_processor.process_epoch(self.model)
