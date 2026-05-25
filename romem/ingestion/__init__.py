"""
Ingestion layer: information extraction (OpenIE) and graph construction.

Keep imports lightweight at module import time. Offline backends and
graph construction components are imported lazily.
"""

from .openie_openai import OpenIE
from .ops import MemoryOpsManager

__all__ = [
    "OpenIE",
    "MemoryOpsManager",
    "GraphBuilder",
    "GraphManager",
    "Neo4jWriter",
]


def __getattr__(name: str):
    if name == "GraphManager":
        from .graph_manager import GraphManager
        return GraphManager
    if name == "Neo4jWriter":
        from .neo4j_writer import Neo4jWriter
        return Neo4jWriter
    if name == "GraphBuilder":
        from .builder import GraphBuilder
        return GraphBuilder
    if name == "VLLMOfflineOpenIE":
        from .openie_vllm_offline import VLLMOfflineOpenIE
        return VLLMOfflineOpenIE
    raise AttributeError(name)
