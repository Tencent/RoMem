#!/usr/bin/env python3
"""
Build unsupervised/self-supervised pretraining data for RoMem gate alpha_r.

No relation-level labels are required.
We mine two artifacts from timed OpenIE caches **and** (optionally) temporal
KG datasets such as ICEWS05-15:

1) relation_nodes.jsonl
   Canonical relations with surface-form variants and support counts.

2) transition_observations.jsonl
   Adjacent temporal observations from the same slot:
   - HR slots: key=(h, r), counterpart=t
   - RT slots: key=(r, t), counterpart=h
   Each row stores whether the counterpart changed across time (changed=0/1),
   giving a purely self-supervised signal without predefined static/dynamic labels.

Improvements over scripts/romem_pretrain/build_gate_pretrain_data.py:
- Integrates ICEWS05-15 (or similar tab-separated TKG) via --icews-dir
- Filters one-to-many (non-functional) slots that produce misleading changed=1
- Caps max delta_days between consecutive observations
- Higher default min_slot_support (3 vs 2)
- Removes outdated pseudo-label files from earlier pipeline versions
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from romem.pretraining.icews_adapter import (
    load_icews_facts,
    mine_icews_transitions,
)

# ---------------------------------------------------------------------------
# Text normalisation (shared with icews_adapter)
# ---------------------------------------------------------------------------

REL_STOPWORDS = {
    "a", "an", "the", "to", "in", "on", "at", "for", "of", "by",
    "from", "with", "and", "or", "into", "onto", "over", "under",
    "upon", "within",
}


@dataclass
class TimedFact:
    h: str
    r_surface: str
    r_canonical: str
    t: str
    happen_ts: float | None
    obs_ts: float | None
    source_file: str
    chunk_id: str


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


def _norm_relation_surface(rel: str) -> str:
    rel = _norm_text(rel)
    if rel.endswith(" inv"):
        rel = rel[:-4].strip()
    if rel.endswith("_inv"):
        rel = rel[:-4].strip()
    return rel


def _canonical_relation(rel_surface: str) -> str:
    rel = _norm_relation_surface(rel_surface)
    toks = [t for t in rel.split() if t]
    reduced = [_simple_stem(t) for t in toks if t not in REL_STOPWORDS]
    if not reduced:
        reduced = [_simple_stem(t) for t in toks]
    can = " ".join(reduced).strip()
    return can or rel


def _parse_dt(text: str) -> datetime | None:
    s = "" if text is None else str(text).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _iso(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Loading OpenIE caches
# ---------------------------------------------------------------------------

def _load_timed_facts(path: str) -> list[TimedFact]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data.get("docs", [])
    out: list[TimedFact] = []

    for doc in docs:
        chunk_id = str(doc.get("idx", ""))
        timed = doc.get("extracted_timed_triples") or []

        if not timed:
            triples = doc.get("extracted_triples") or []
            for triple in triples:
                if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                    continue
                h, r, t = triple
                r_surface = _norm_relation_surface(str(r))
                out.append(
                    TimedFact(
                        h=_norm_text(h),
                        r_surface=r_surface,
                        r_canonical=_canonical_relation(r_surface),
                        t=_norm_text(t),
                        happen_ts=None,
                        obs_ts=None,
                        source_file=path,
                        chunk_id=chunk_id,
                    )
                )
            continue

        for item in timed:
            triple = item.get("triple")
            if not (isinstance(triple, (list, tuple)) and len(triple) == 3):
                continue
            h, r, t = triple
            r_surface = _norm_relation_surface(str(r))
            happen_dt = _parse_dt(item.get("happen_time", ""))
            obs_dt = _parse_dt(item.get("system_time", ""))
            out.append(
                TimedFact(
                    h=_norm_text(h),
                    r_surface=r_surface,
                    r_canonical=_canonical_relation(r_surface),
                    t=_norm_text(t),
                    happen_ts=float(happen_dt.timestamp()) if happen_dt is not None else None,
                    obs_ts=float(obs_dt.timestamp()) if obs_dt is not None else None,
                    source_file=path,
                    chunk_id=chunk_id,
                )
            )

    return out


def _dedupe_facts(facts: Iterable[TimedFact]) -> list[TimedFact]:
    seen: set[tuple] = set()
    out: list[TimedFact] = []
    for f in facts:
        day_key = int(f.happen_ts // 86400) if f.happen_ts is not None else None
        key = (f.h, f.r_surface, f.t, day_key, f.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


# ---------------------------------------------------------------------------
# Slot statistics + one-to-many detection
# ---------------------------------------------------------------------------

def _slot_stats(events: list[tuple[float, str]]) -> dict:
    if not events:
        return {"n_obs": 0, "n_unique": 0, "n_transitions": 0, "span_days": 0.0}

    ordered = sorted(events, key=lambda x: (x[0], x[1]))
    n_obs = len(ordered)
    unique_vals = sorted({v for _, v in ordered})

    n_transitions = 0
    prev = ordered[0][1]
    for _, cur in ordered[1:]:
        if cur != prev:
            n_transitions += 1
            prev = cur

    span_days = max(0.0, (ordered[-1][0] - ordered[0][0]) / 86400.0)
    return {
        "n_obs": n_obs,
        "n_unique": len(unique_vals),
        "n_transitions": n_transitions,
        "span_days": span_days,
    }


def _is_one_to_many_slot(stats: dict, *, max_functional_ratio: float = 0.5) -> bool:
    """Heuristic: a slot is one-to-many if almost every observation has a
    different counterpart, meaning the relation is non-functional (like
    ``disney_original_series``).

    Functional (president_of): n_unique << n_obs (many re-observations of same entity).
    One-to-many (member_of_cast): n_unique ≈ n_obs (every observation is a new entity).

    We use the ratio n_unique / n_obs. If it exceeds max_functional_ratio
    AND n_unique > 2, we treat the slot as non-functional.
    """
    n_obs = stats.get("n_obs", 0)
    n_unique = stats.get("n_unique", 0)
    if n_obs < 3 or n_unique <= 2:
        return False
    ratio = n_unique / n_obs
    return ratio > max_functional_ratio


# ---------------------------------------------------------------------------
# Transition pair mining
# ---------------------------------------------------------------------------

def _emit_transition_pairs(
    slot_type: str,
    key: tuple[str, str],
    events: list[dict],
    *,
    min_slot_support: int,
    max_pairs_per_slot: int,
    max_delta_days: float,
    min_pair_delta_days: float,
    filter_one_to_many: bool,
    max_functional_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], dict]:
    """Mine transition pairs from a single slot.

    Returns (rows, rel_stat_delta) where rel_stat_delta has keys
    {'pairs': float, 'changed': float, 'delta_days_sum': float}.
    """
    stats = _slot_stats([(e["ts"], e["counterpart"]) for e in events])
    if stats["n_obs"] < min_slot_support:
        return [], {}

    # Filter non-functional (one-to-many) slots
    if filter_one_to_many and _is_one_to_many_slot(stats, max_functional_ratio=max_functional_ratio):
        return [], {}

    ordered = sorted(events, key=lambda x: (x["ts"], x["counterpart"]))
    if len(ordered) < 2:
        return [], {}

    pairs = list(zip(ordered[:-1], ordered[1:]))
    if max_pairs_per_slot > 0 and len(pairs) > max_pairs_per_slot:
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs_per_slot]

    can = key[1] if slot_type == "hr" else key[0]
    rows: list[dict] = []
    rel_delta = {"pairs": 0.0, "changed": 0.0, "delta_days_sum": 0.0}

    for prev, cur in pairs:
        if cur["ts"] <= prev["ts"]:
            continue
        delta_days = (cur["ts"] - prev["ts"]) / 86400.0
        if delta_days < min_pair_delta_days:
            continue
        if delta_days > max_delta_days:
            continue

        changed = int(prev["counterpart"] != cur["counterpart"])
        if slot_type == "hr":
            head_prev = key[0]
            head_cur = key[0]
            tail_prev = prev["counterpart"]
            tail_cur = cur["counterpart"]
        else:
            head_prev = prev["counterpart"]
            head_cur = cur["counterpart"]
            tail_prev = key[1]
            tail_cur = key[1]

        rows.append(
            {
                "slot_type": slot_type,
                "canonical_relation": can,
                "relation_text_prev": prev["r_surface"],
                "relation_text_cur": cur["r_surface"],
                "head_prev": head_prev,
                "head_cur": head_cur,
                "tail_prev": tail_prev,
                "tail_cur": tail_cur,
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
                "source_prev": {"file": prev.get("source_file", ""), "chunk_id": prev.get("chunk_id", "")},
                "source_cur": {"file": cur.get("source_file", ""), "chunk_id": cur.get("chunk_id", "")},
            }
        )

        rel_delta["pairs"] += 1.0
        rel_delta["changed"] += float(changed)
        rel_delta["delta_days_sum"] += delta_days

    return rows, rel_delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unsupervised alpha_r pretraining data from timed OpenIE caches and temporal KGs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["outputs/**/openie_results_ner_*.json"],
        help="Glob patterns for OpenIE JSON caches.",
    )
    parser.add_argument("--out-dir", default="outputs/pretrain_gate_data", help="Output directory.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files.")

    # --- ICEWS integration ---
    parser.add_argument(
        "--icews-dir",
        default=None,
        help="Path to ICEWS05-15 dataset directory (containing train.txt, valid.txt, test.txt). "
             "If provided, ICEWS facts are loaded and mined for transition observations.",
    )
    parser.add_argument(
        "--icews-splits",
        nargs="+",
        default=["train.txt", "valid.txt", "test.txt"],
        help="Which ICEWS split files to include.",
    )

    # --- Slot filtering ---
    parser.add_argument("--min-slot-support", type=int, default=3,
                        help="Minimum observations per slot to mine transitions (default: 3, was 2).")
    parser.add_argument("--min-pair-delta-days", type=float, default=0.0)
    parser.add_argument("--max-delta-days", type=float, default=18250.0,
                        help="Maximum delta_days between consecutive observations (default: 18250 = ~50 years).")
    parser.add_argument("--max-pairs-per-slot", type=int, default=256)

    # --- One-to-many filtering ---
    parser.add_argument("--filter-one-to-many", action="store_true", default=True,
                        help="Filter out one-to-many (non-functional) slots (default: enabled).")
    parser.add_argument("--no-filter-one-to-many", action="store_false", dest="filter_one_to_many",
                        help="Disable one-to-many slot filtering.")
    parser.add_argument("--max-functional-ratio", type=float, default=0.5,
                        help="Slots with n_unique/n_obs > this are treated as one-to-many (default: 0.5).")

    # --- Slot type selection ---
    parser.add_argument("--slot-types", nargs="+", default=["hr"],
                        help="Slot types to mine: hr (head-relation), rt (relation-tail). "
                             "Default: hr only. ICEWS relations are functional on the tail side, "
                             "so HR slots are most informative.")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Remove outdated pseudo-label files from earlier pipeline versions.
    for old in (
        "relation_gate_pretrain.jsonl",
        "slot_evidence.jsonl",
        "dynamic_pairs.jsonl",
        "static_invariance_pairs.jsonl",
        "semantic_edges.jsonl",
    ):
        old_path = os.path.join(args.out_dir, old)
        if os.path.exists(old_path):
            os.remove(old_path)

    # -----------------------------------------------------------------------
    # 1) Load OpenIE facts
    # -----------------------------------------------------------------------
    files: list[str] = []
    for pat in args.inputs:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    if args.max_files > 0:
        files = files[: args.max_files]

    all_facts: list[TimedFact] = []
    for p in files:
        try:
            all_facts.extend(_load_timed_facts(p))
        except Exception as exc:
            print(f"[WARN] skip {p}: {exc}")

    all_facts = _dedupe_facts(all_facts)
    all_facts = [f for f in all_facts if f.h and f.r_surface and f.r_canonical and f.t]
    print(f"[OpenIE] {len(files)} input files → {len(all_facts)} deduplicated facts")

    # -----------------------------------------------------------------------
    # 2) Load ICEWS facts (optional)
    # -----------------------------------------------------------------------
    icews_facts_loaded = 0
    icews_transition_rows: list[dict] = []
    icews_relation_nodes: list[dict] = []
    icews_relation_activity: list[dict] = []

    if args.icews_dir:
        icews_root = Path(args.icews_dir)
        if not icews_root.exists():
            print(f"[WARN] ICEWS dir not found: {icews_root}, skipping ICEWS integration.")
        else:
            all_icews_facts = []
            for split_name in args.icews_splits:
                split_path = icews_root / split_name
                if split_path.exists():
                    facts = load_icews_facts(split_path)
                    all_icews_facts.extend(facts)
                    print(f"[ICEWS] {split_name}: {len(facts)} facts")
                else:
                    print(f"[WARN] ICEWS split not found: {split_path}")

            icews_facts_loaded = len(all_icews_facts)
            print(f"[ICEWS] Total: {icews_facts_loaded} facts loaded")

            if all_icews_facts:
                icews_transition_rows, icews_relation_nodes, icews_relation_activity = mine_icews_transitions(
                    all_icews_facts,
                    min_slot_support=args.min_slot_support,
                    max_pairs_per_slot=args.max_pairs_per_slot,
                    max_delta_days=args.max_delta_days,
                    seed=args.seed,
                )
                print(
                    f"[ICEWS] Mined {len(icews_transition_rows)} transition observations "
                    f"from {len(icews_relation_activity)} relations"
                )

    # -----------------------------------------------------------------------
    # 3) Build relation node index from OpenIE facts
    # -----------------------------------------------------------------------
    rel_variant_counts: dict[str, Counter] = defaultdict(Counter)
    rel_variant_timed_counts: dict[str, Counter] = defaultdict(Counter)
    rel_total_counts: Counter = Counter()
    rel_timed_counts: Counter = Counter()

    hr_slot_events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rt_slot_events: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for f in all_facts:
        can = f.r_canonical
        surf = f.r_surface
        rel_total_counts[can] += 1
        rel_variant_counts[can][surf] += 1

        if f.happen_ts is None:
            continue

        rel_timed_counts[can] += 1
        rel_variant_timed_counts[can][surf] += 1

        hr_slot_events[(f.h, can)].append(
            {
                "ts": float(f.happen_ts),
                "counterpart": f.t,
                "h": f.h,
                "t": f.t,
                "r_surface": surf,
                "source_file": f.source_file,
                "chunk_id": f.chunk_id,
            }
        )
        rt_slot_events[(can, f.t)].append(
            {
                "ts": float(f.happen_ts),
                "counterpart": f.h,
                "h": f.h,
                "t": f.t,
                "r_surface": surf,
                "source_file": f.source_file,
                "chunk_id": f.chunk_id,
            }
        )

    # 3a) Write relation_nodes.jsonl (OpenIE-derived)
    openie_relation_nodes: list[dict] = []
    for can in sorted(rel_total_counts.keys()):
        variants = []
        for surf, cnt in rel_variant_counts[can].most_common():
            variants.append(
                {
                    "relation_text": surf,
                    "count": int(cnt),
                    "timed_count": int(rel_variant_timed_counts[can].get(surf, 0)),
                }
            )
        openie_relation_nodes.append(
            {
                "canonical_relation": can,
                "total_facts": int(rel_total_counts[can]),
                "timed_facts": int(rel_timed_counts[can]),
                "variant_count": len(variants),
                "variants": variants,
            }
        )

    # -----------------------------------------------------------------------
    # 4) Mine OpenIE transition pairs
    # -----------------------------------------------------------------------
    openie_transition_rows: list[dict] = []
    relation_transition_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"pairs": 0.0, "changed": 0.0, "delta_days_sum": 0.0}
    )

    slot_types = set(args.slot_types)

    if "hr" in slot_types:
        for key, events in hr_slot_events.items():
            rows, delta = _emit_transition_pairs(
                "hr", key, events,
                min_slot_support=args.min_slot_support,
                max_pairs_per_slot=args.max_pairs_per_slot,
                max_delta_days=args.max_delta_days,
                min_pair_delta_days=args.min_pair_delta_days,
                filter_one_to_many=args.filter_one_to_many,
                max_functional_ratio=args.max_functional_ratio,
                rng=rng,
            )
            openie_transition_rows.extend(rows)
            if delta:
                can = key[1]
                st = relation_transition_stats[can]
                for k in ("pairs", "changed", "delta_days_sum"):
                    st[k] += delta[k]

    if "rt" in slot_types:
        for key, events in rt_slot_events.items():
            rows, delta = _emit_transition_pairs(
                "rt", key, events,
                min_slot_support=args.min_slot_support,
                max_pairs_per_slot=args.max_pairs_per_slot,
                max_delta_days=args.max_delta_days,
                min_pair_delta_days=args.min_pair_delta_days,
                filter_one_to_many=args.filter_one_to_many,
                max_functional_ratio=args.max_functional_ratio,
                rng=rng,
            )
            openie_transition_rows.extend(rows)
            if delta:
                can = key[0]
                st = relation_transition_stats[can]
                for k in ("pairs", "changed", "delta_days_sum"):
                    st[k] += delta[k]

    print(
        f"[OpenIE] Mined {len(openie_transition_rows)} transition observations "
        f"from {len(relation_transition_stats)} relations"
    )

    # -----------------------------------------------------------------------
    # 5) Merge OpenIE + ICEWS outputs
    # -----------------------------------------------------------------------

    # Merge relation_nodes: combine by canonical_relation
    merged_nodes_map: dict[str, dict] = {}
    for node in openie_relation_nodes:
        can = node["canonical_relation"]
        merged_nodes_map[can] = node

    for node in icews_relation_nodes:
        can = node["canonical_relation"]
        if can in merged_nodes_map:
            # Merge counts and variants
            existing = merged_nodes_map[can]
            existing["total_facts"] += node["total_facts"]
            existing["timed_facts"] += node["timed_facts"]
            # Merge variants
            existing_surf = {v["relation_text"] for v in existing["variants"]}
            for v in node["variants"]:
                if v["relation_text"] not in existing_surf:
                    existing["variants"].append(v)
                    existing_surf.add(v["relation_text"])
                else:
                    for ev in existing["variants"]:
                        if ev["relation_text"] == v["relation_text"]:
                            ev["count"] += v["count"]
                            ev["timed_count"] += v["timed_count"]
                            break
            existing["variant_count"] = len(existing["variants"])
        else:
            merged_nodes_map[can] = node

    relation_nodes = [merged_nodes_map[can] for can in sorted(merged_nodes_map.keys())]

    # Merge transition rows
    transition_rows = openie_transition_rows + icews_transition_rows

    # Merge relation activity stats
    for row in icews_relation_activity:
        can = row["canonical_relation"]
        st = relation_transition_stats[can]
        st["pairs"] += float(row["num_pairs"])
        st["changed"] += float(row["change_rate"] * row["num_pairs"])
        st["delta_days_sum"] += float(row["mean_delta_days"] * row["num_pairs"])

    # -----------------------------------------------------------------------
    # 6) Write outputs
    # -----------------------------------------------------------------------
    relation_nodes_path = os.path.join(args.out_dir, "relation_nodes.jsonl")
    transition_path = os.path.join(args.out_dir, "transition_observations.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    with open(relation_nodes_path, "w", encoding="utf-8") as f:
        for row in relation_nodes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(transition_path, "w", encoding="utf-8") as f:
        for row in transition_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compute merged relation activity
    relation_activity: list[dict] = []
    for can, st in sorted(relation_transition_stats.items()):
        p = max(1.0, float(st["pairs"]))
        relation_activity.append(
            {
                "canonical_relation": can,
                "num_pairs": int(st["pairs"]),
                "change_rate": float(st["changed"] / p),
                "mean_delta_days": float(st["delta_days_sum"] / p),
            }
        )

    relation_activity_path = os.path.join(args.out_dir, "relation_activity.jsonl")
    with open(relation_activity_path, "w", encoding="utf-8") as f:
        for row in relation_activity:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    # 7) Summary
    # -----------------------------------------------------------------------
    n_changed = sum(1 for r in transition_rows if r.get("changed") == 1)
    n_unchanged = sum(1 for r in transition_rows if r.get("changed") == 0)

    summary = {
        "inputs": files,
        "num_input_files": len(files),
        "icews_dir": args.icews_dir,
        "icews_facts_loaded": icews_facts_loaded,
        "num_all_openie_facts_after_dedupe": len(all_facts),
        "num_relation_nodes": len(relation_nodes),
        "num_transition_observations": len(transition_rows),
        "num_openie_transition_observations": len(openie_transition_rows),
        "num_icews_transition_observations": len(icews_transition_rows),
        "num_relations_with_transition_signal": len(relation_activity),
        "changed_count": n_changed,
        "unchanged_count": n_unchanged,
        "changed_ratio": round(n_changed / max(1, n_changed + n_unchanged), 4),
        "filter_one_to_many": args.filter_one_to_many,
        "max_functional_ratio": args.max_functional_ratio,
        "max_delta_days": args.max_delta_days,
        "min_slot_support": args.min_slot_support,
        "slot_types": args.slot_types,
        "args": vars(args),
        "output_files": {
            "relation_nodes": relation_nodes_path,
            "transition_observations": transition_path,
            "relation_activity": relation_activity_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
