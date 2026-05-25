"""Shared helpers for RoMem benchmark backends."""

from __future__ import annotations

from typing import Mapping, Any


def apply_romem_config(config: Any, overrides: Mapping[str, Any] | None) -> None:
    """Apply config overrides from a mapping onto a BaseConfig instance."""
    if not overrides:
        return
    # Map legacy or dataset-level keys onto BaseConfig fields.
    if "romem_tkge_verbose" in overrides and hasattr(config, "tkge_verbose"):
        config.tkge_verbose = overrides["romem_tkge_verbose"]
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
