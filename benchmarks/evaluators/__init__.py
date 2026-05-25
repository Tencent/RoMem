"""
Evaluator registry per dataset.
"""

from .locomo import LocomoEvaluator  # noqa: F401
from .dmr_msc import DmrMscEvaluator  # noqa: F401
from .longmemeval import LongmemevalEvaluator  # noqa: F401
from .fintmmbench import FinTMMBenchEvaluator  # noqa: F401
