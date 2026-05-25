"""
Feature and node-ordering utilities adapted from TKGE preprocessing scripts.
"""

from __future__ import annotations

from typing import Iterable



def rank_new_entities(entity_ids: Iterable[int]) -> list[int]:
    """
    Placeholder helper that will rank new entity identifiers according to RoMem heuristics.

    Parameters
    ----------
    entity_ids:
        Iterable of entity identifiers to be ranked.

    Returns
    -------
    list[int]
        Currently returns the input order.
    """

    return list(entity_ids)
