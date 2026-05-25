"""
Model components for the TKGE adapter.
"""

try:
    from .base import BaseModel  # noqa: F401
    from .lora_layers import LoraKGE_Layers, TransE  # noqa: F401
except Exception:  # pragma: no cover
    # Optional heavy deps (e.g., torch_scatter) are not required for RoMem adapter paths.
    BaseModel = None  # type: ignore[assignment]
    LoraKGE_Layers = None  # type: ignore[assignment]
    TransE = None  # type: ignore[assignment]

from .distmult import DistMultModel  # noqa: F401
from .romem_distmult import RoMemDistMultModel  # noqa: F401
from .romem_chronor import RoMemChronoRModel  # noqa: F401
