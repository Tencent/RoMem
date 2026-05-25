"""
Knowledge Graph Embedding (KGE) subsystem.

Provides the TKGE-based temporal encoder, RoMem scoring model,
retriever interface, and supporting utilities.
"""

from .config import TKGEConfig
from .encoder import TKGEEncoder
from .retriever import TKGERetriever

__all__ = [
    "TKGEConfig",
    "TKGEEncoder",
    "TKGERetriever",
]
