"""
Shared utility functions used across benchmark runner backends.

Centralised here to avoid duplication across per-benchmark backend files.
"""

from __future__ import annotations

import os
import re
import string

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    value = (value or '').lower()
    value = re.sub(r'\s+', ' ', value).strip()
    return value.translate(str.maketrans('', '', string.punctuation))


def token_overlap(a: str, b: str) -> float:
    """Fraction of shared tokens (by min-set size)."""
    tokens_a = set(filter(None, a.split()))
    tokens_b = set(filter(None, b.split()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_bool(value: str | None, default: bool) -> bool:
    """Parse a string config value as a boolean."""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_optional_bool(value: str | None) -> bool | None:
    """Parse a string config value as an optional boolean."""
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def initialize_answer_llm(
    model: str | None,
    api_key: str | None,
    base_url: str | None,
):
    """Create an LLM client for answer generation."""
    from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
    from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
    from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    if not model:
        model = os.getenv('OPENAI_MODEL') or 'gpt-4o-mini'
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    base_url = base_url or os.getenv('OPENAI_BASE_URL')
    if not api_key:
        return None
    config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
    name = (config.model or '').lower()
    if not name or name.startswith('gpt'):
        return OpenAIClient(config=config)
    return OpenAIGenericClient(config=config)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AnswerResponse(BaseModel):
    """Standard answer response schema for LLM answer generation."""
    answer: str


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------

def usage_totals() -> dict[str, float]:
    """Snapshot of current Graphiti LLM usage totals."""
    from baselines.graphiti.graphiti_core.llm_client.usage_tracker import usage_tracker
    return usage_tracker.summary().get('totals', {}).copy()


def usage_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    """Compute token/cost deltas between two usage snapshots."""
    return {
        'prompt_tokens': after.get('prompt_tokens', 0.0) - before.get('prompt_tokens', 0.0),
        'completion_tokens': after.get('completion_tokens', 0.0) - before.get('completion_tokens', 0.0),
        'cost': after.get('cost', 0.0) - before.get('cost', 0.0),
    }
