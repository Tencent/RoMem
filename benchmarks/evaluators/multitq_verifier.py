"""
Rule-based answer verification for MultiTQ, matching MemoTime's multi-strategy approach.

Strategies applied in order (first match wins):
  1. exact            – normalized entity equality
  2. contain          – bidirectional substring
  3. advanced_normalize – strip prefixes/brackets/punctuation, then substring
  4. time_format      – year-month level time matching
  5. multi_answer     – any comma-separated part matches
  6. semantic         – word overlap > 50%
  7. loose            – remove all spaces/underscores, then substring
"""

from __future__ import annotations

import re
from typing import List, Tuple


def verify_answer(
    predicted: str,
    gold_answers: list[str],
    answer_type: str = "entity",
) -> Tuple[bool, str]:
    """
    Verify a predicted answer against gold answers using MemoTime's strategies.

    Returns:
        (is_correct, match_strategy)  e.g. (True, "contain")
    """
    if not predicted or not predicted.strip():
        return False, "empty"
    if not gold_answers:
        return False, "no_gold"

    # Extract candidate answers from the prediction
    candidates = _extract_answer(predicted)

    strategies = [
        ("exact", _match_exact),
        ("contain", _match_contain),
        ("advanced_normalize", _match_advanced),
        ("time_format", _match_time_format),
        ("multi_answer", _match_multi_answer),
        ("semantic", _match_semantic),
        ("loose", _match_loose),
    ]

    for strategy_name, match_fn in strategies:
        for cand in candidates:
            for gold in gold_answers:
                if match_fn(cand, gold):
                    return True, strategy_name

    return False, "none"


# ── Matching strategies ──────────────────────────────────────────


def _match_exact(predicted: str, golden: str) -> bool:
    return _normalize_entity(predicted) == _normalize_entity(golden)


def _match_contain(predicted: str, golden: str) -> bool:
    p = _normalize_entity(predicted)
    g = _normalize_entity(golden)
    return g in p or p in g


def _match_advanced(predicted: str, golden: str) -> bool:
    p = _normalize_advanced(predicted)
    g = _normalize_advanced(golden)
    if not p or not g:
        return False
    return p == g or g in p or p in g


def _match_time_format(predicted: str, golden: str) -> bool:
    p = _normalize_entity(predicted)
    g = _normalize_entity(golden)

    # Year-month match
    pred_ym = re.search(r"(\d{4})[-\s]*(\d{1,2})", p)
    gold_ym = re.search(r"(\d{4})[-\s]*(\d{1,2})", g)
    if pred_ym and gold_ym:
        pred_str = f"{pred_ym.group(1)}-{pred_ym.group(2).zfill(2)}"
        gold_str = f"{gold_ym.group(1)}-{gold_ym.group(2).zfill(2)}"
        return pred_str == gold_str

    # Month name matching
    month_names = {
        "january": "01", "jan": "01", "february": "02", "feb": "02",
        "march": "03", "mar": "03", "april": "04", "apr": "04",
        "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
        "august": "08", "aug": "08", "september": "09", "sep": "09", "sept": "09",
        "october": "10", "oct": "10", "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }
    for name, num in month_names.items():
        if name in p and num in g:
            return True

    # Prefix-based time containment (e.g. "2005" matches "2005-01-15")
    p_norm, _ = _normalize_time(p)
    g_norm, _ = _normalize_time(g)
    if p_norm and g_norm:
        if p_norm == g_norm or p_norm.startswith(g_norm) or g_norm.startswith(p_norm):
            return True

    return False


def _match_multi_answer(predicted: str, golden: str) -> bool:
    p = _normalize_entity(predicted)
    g = _normalize_entity(golden)

    if "," in g:
        for part in g.split(","):
            part = part.strip()
            if part and (p in part or part in p):
                return True
    if "," in p:
        for part in p.split(","):
            part = part.strip()
            if part and (part in g or g in part):
                return True
    return False


def _match_semantic(predicted: str, golden: str) -> bool:
    p = _normalize_entity(predicted)
    g = _normalize_entity(golden)

    # Remove common country modifiers for fairer comparison
    for tag in ("(china)", "(japan)", "(thailand)", "(south korea)", "(india)"):
        p = p.replace(tag, "").strip()
        g = g.replace(tag, "").strip()

    if p and g and (p in g or g in p):
        return True

    p_words = set(p.split())
    g_words = set(g.split())
    if p_words and g_words:
        overlap = len(p_words & g_words)
        total = len(p_words | g_words)
        if total > 0 and overlap / total > 0.5:
            return True
    return False


def _match_loose(predicted: str, golden: str) -> bool:
    p = predicted.replace("_", "").replace(" ", "").lower()
    g = golden.replace("_", "").replace(" ", "").lower()
    return p in g or g in p


# ── Normalization helpers ────────────────────────────────────────


def _normalize_entity(entity: str) -> str:
    if not entity:
        return ""
    normalized = re.sub(r"[_/\'\"()\[\]{}<>.,;:!?@#$%^&*+=|\\~`-]", " ", entity)
    normalized = normalized.lower()
    return " ".join(normalized.split())


def _normalize_advanced(answer: str) -> str:
    if not answer:
        return ""
    normalized = answer.strip()

    # Remove common prefixes
    for prefix in ("So the answer is:", "The answer is:", "Answer:", "Answer is:"):
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):].strip()

    # Remove date in parentheses
    normalized = re.sub(r"\s*\(\d{4}[-/]\d{2}[-/]\d{2}\)\s*", "", normalized)
    normalized = re.sub(r"\s*\(\d{4}[-/]\d{2}\)\s*", "", normalized)
    normalized = re.sub(r"\s*\(\d{4}\)\s*", "", normalized)
    # Remove trailing "on YYYY-MM-DD"
    normalized = re.sub(r"\s+on\s+\d{4}[-/]\d{2}[-/]\d{2}.*$", "", normalized)

    # Standard normalization
    normalized = normalized.replace("_", " ").replace("/", " ")
    normalized = re.sub(r"[.,!?]", "", normalized)
    normalized = " ".join(normalized.split()).lower()
    return normalized


def _normalize_time(time_str: str) -> Tuple[str | None, str]:
    if not time_str:
        return None, "unknown"
    time_str = str(time_str).strip()

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", time_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "day"
    m = re.match(r"(\d{4})-(\d{2})", time_str)
    if m:
        return f"{m.group(1)}-{m.group(2)}", "month"
    m = re.match(r"(\d{4})", time_str)
    if m:
        return m.group(1), "year"
    return time_str, "unknown"


def _extract_answer(answer_text: str) -> List[str]:
    """Extract candidate answers from a prediction string."""
    if not answer_text:
        return []

    main = answer_text
    # Strip "the answer is" prefix
    m = re.search(r"(?:So )?[Tt]he answer is[:\s]+(.+)", answer_text, re.DOTALL)
    if m:
        main = m.group(1)

    # Remove trailing date qualifiers
    main = re.sub(r"\s+on\s+\d{4}[-/]\d{2}[-/]\d{2}[^\s,]*", "", main)

    parts = re.split(r"\s*,\s*(?:or\s+)?|\s+or\s+", main)
    candidates: list[str] = []
    seen: set[str] = set()
    for part in parts:
        cleaned = part.strip()
        if re.match(r"^on\s+\d{4}", cleaned, re.IGNORECASE):
            continue
        cleaned = re.sub(r"\s*\[.*?\]\s*", " ", cleaned).strip()
        cleaned = re.sub(r"\s*\(\d{4}[-/]\d{2}[-/]\d{2}\)\s*", "", cleaned)
        cleaned = re.sub(r"\s*\(\d{4}[-/]\d{2}\)\s*", "", cleaned)
        cleaned = re.sub(r"\s*\(\d{4}\)\s*", "", cleaned)
        cleaned = re.sub(r"\s+on\s+\d{4}[-/]\d{2}[-/]\d{2}.*$", "", cleaned)
        cleaned = " ".join(cleaned.split())

        key = cleaned.lower().replace("_", " ")
        if cleaned and key not in {"or", "and", "the"} and key not in seen:
            candidates.append(cleaned)
            seen.add(key)

    if not candidates:
        cleaned = answer_text.strip().split(",")[0].strip()
        if cleaned:
            candidates.append(cleaned)
    return candidates or [answer_text.strip()]
