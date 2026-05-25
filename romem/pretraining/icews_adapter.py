#!/usr/bin/env python3
"""
Adapter to convert ICEWS05-15 (or similar tab-separated temporal KG files)
into the transition_observations format used by gate pretraining.

ICEWS format:  head \t relation \t tail \t YYYY-MM-DD

ICEWS relations are explicitly functional (event-level), making them
high-quality supervision for the semantic gate alpha_r.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass
class IcewsFact:
    h: str
    r: str
    t: str
    date_str: str
    timestamp: float  # unix seconds


# ---------------------------------------------------------------------------
# Text normalisation (mirrors build_gate_pretrain_data)
# ---------------------------------------------------------------------------

REL_STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "at", "for", "of", "by",
    "from", "with", "and", "or", "into", "onto", "over", "under",
    "upon", "within",
}


def _norm_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_stem(tok: str) -> str:
    if tok in {"is", "are", "was", "were", "be", "been", "being"}:
        return "be"
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ied"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _norm_relation(rel: str) -> str:
    rel = _norm_text(rel)
    if rel.endswith(" inv"):
        rel = rel[:-4].strip()
    if rel.endswith("_inv"):
        rel = rel[:-4].strip()
    return rel


def canonical_relation(rel_surface: str) -> str:
    rel = _norm_relation(rel_surface)
    toks = [t for t in rel.split() if t]
    reduced = [_simple_stem(t) for t in toks if t not in REL_STOPWORDS]
    if not reduced:
        reduced = [_simple_stem(t) for t in toks]
    can = " ".join(reduced).strip()
    return can or rel


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_icews_facts(path: str | Path) -> list[IcewsFact]:
    """Load tab-separated temporal KG file (head \\t rel \\t tail \\t date)."""
    facts: list[IcewsFact] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            h, r, t, date_str = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            if not (h and r and t and date_str):
                continue
            try:
                dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            facts.append(IcewsFact(
                h=_norm_text(h),
                r=_norm_relation(r),
                t=_norm_text(t),
                date_str=date_str,
                timestamp=dt.timestamp(),
            ))
    return facts


# ---------------------------------------------------------------------------
# Slot statistics
# ---------------------------------------------------------------------------

def _slot_stats(events: list[tuple[float, str]]) -> dict:
    if not events:
        return {"n_obs": 0, "n_unique": 0, "n_transitions": 0, "span_days": 0.0}
    ordered = sorted(events, key=lambda x: (x[0], x[1]))
    unique_vals = {v for _, v in ordered}
    n_transitions = sum(1 for (_, a), (_, b) in zip(ordered, ordered[1:]) if a != b)
    span_days = max(0.0, (ordered[-1][0] - ordered[0][0]) / 86400.0)
    return {
        "n_obs": len(ordered),
        "n_unique": len(unique_vals),
        "n_transitions": n_transitions,
        "span_days": span_days,
    }


# ---------------------------------------------------------------------------
# Mining transition observations
# ---------------------------------------------------------------------------

def mine_icews_transitions(
    facts: Iterable[IcewsFact],
    *,
    min_slot_support: int = 3,
    max_pairs_per_slot: int = 256,
    max_delta_days: float = 18250.0,
    seed: int = 13,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Mine transition observations from ICEWS facts.

    Returns:
        (transition_rows, relation_nodes, relation_activity)
    """
    rng = random.Random(seed)

    # Collect per-slot events (HR slots only -- ICEWS relations are functional on the tail side)
    hr_slot_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rel_counts: dict[str, int] = defaultdict(int)
    rel_surface_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for f in facts:
        can = canonical_relation(f.r)
        rel_counts[can] += 1
        rel_surface_counts[can][f.r] += 1
        hr_slot_events[(f.h, can)].append({
            "ts": f.timestamp,
            "counterpart": f.t,
            "h": f.h,
            "t": f.t,
            "r_surface": f.r,
        })

    # Build relation nodes
    relation_nodes: list[dict] = []
    for can in sorted(rel_counts.keys()):
        variants = [
            {"relation_text": surf, "count": cnt, "timed_count": cnt}
            for surf, cnt in sorted(rel_surface_counts[can].items(), key=lambda x: -x[1])
        ]
        relation_nodes.append({
            "canonical_relation": can,
            "total_facts": rel_counts[can],
            "timed_facts": rel_counts[can],
            "variant_count": len(variants),
            "variants": variants,
        })

    # Mine transition pairs
    transition_rows: list[dict] = []
    rel_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"pairs": 0.0, "changed": 0.0, "delta_days_sum": 0.0}
    )

    for (h, can), events in hr_slot_events.items():
        stats = _slot_stats([(e["ts"], e["counterpart"]) for e in events])
        if stats["n_obs"] < min_slot_support:
            continue

        ordered = sorted(events, key=lambda x: (x["ts"], x["counterpart"]))
        if len(ordered) < 2:
            continue

        pairs = list(zip(ordered[:-1], ordered[1:]))
        if max_pairs_per_slot > 0 and len(pairs) > max_pairs_per_slot:
            rng.shuffle(pairs)
            pairs = pairs[:max_pairs_per_slot]

        for prev, cur in pairs:
            if cur["ts"] <= prev["ts"]:
                continue
            delta_days = (cur["ts"] - prev["ts"]) / 86400.0
            if delta_days > max_delta_days:
                continue

            changed = int(prev["counterpart"] != cur["counterpart"])
            transition_rows.append({
                "slot_type": "hr",
                "canonical_relation": can,
                "relation_text_prev": prev["r_surface"],
                "relation_text_cur": cur["r_surface"],
                "head_prev": h,
                "head_cur": h,
                "tail_prev": prev["counterpart"],
                "tail_cur": cur["counterpart"],
                "changed": changed,
                "time_prev": _iso(prev["ts"]),
                "time_cur": _iso(cur["ts"]),
                "delta_days": round(delta_days, 4),
                "slot_stats": {
                    "n_obs": stats["n_obs"],
                    "n_unique_counterparts": stats["n_unique"],
                    "n_transitions": stats["n_transitions"],
                    "span_days": round(stats["span_days"], 4),
                },
                "source_prev": {"file": "icews05-15", "chunk_id": ""},
                "source_cur": {"file": "icews05-15", "chunk_id": ""},
            })

            st = rel_stats[can]
            st["pairs"] += 1.0
            st["changed"] += float(changed)
            st["delta_days_sum"] += delta_days

    # Build relation activity
    relation_activity: list[dict] = []
    for can, st in sorted(rel_stats.items()):
        p = max(1.0, st["pairs"])
        relation_activity.append({
            "canonical_relation": can,
            "num_pairs": int(st["pairs"]),
            "change_rate": st["changed"] / p,
            "mean_delta_days": st["delta_days_sum"] / p,
        })

    return transition_rows, relation_nodes, relation_activity
