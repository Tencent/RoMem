#!/usr/bin/env python3
"""
Pretrain the semantic speed gate (alpha_r) from temporal transition observations.

Can be used as:
  - CLI:  python -m romem.pretraining.pretrain_alpha_r --pretrain-data-dir ...
  - API:  from romem.pretraining.pretrain_alpha_r import pretrain_gate_from_data
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch

from romem.embedding_model import _get_embedding_model_class
from romem.utils.config_utils import BaseConfig
from romem.kge.config import TKGEConfig
from romem.kge.encoder import TKGEEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    return out


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _choose_relation_texts(pretrain_data_dir: str, max_relations: int, seed: int) -> list[tuple[str, str]]:
    """
    Returns list of (canonical_relation, representative_surface_relation_text).
    Only uses relations that appear in transition observations.
    """
    transition_path = os.path.join(pretrain_data_dir, "transition_observations.jsonl")
    relation_nodes_path = os.path.join(pretrain_data_dir, "relation_nodes.jsonl")

    transition_rows = _read_jsonl(transition_path)
    if not transition_rows:
        raise RuntimeError(f"No transition rows found: {transition_path}")

    target_canonical = sorted(
        {
            str(r.get("canonical_relation", "")).strip()
            for r in transition_rows
            if str(r.get("canonical_relation", "")).strip()
        }
    )
    if not target_canonical:
        raise RuntimeError("No canonical relations found in transition observations.")

    node_rows = _read_jsonl(relation_nodes_path)
    can_to_surface: dict[str, str] = {}
    for row in node_rows:
        can = str(row.get("canonical_relation", "")).strip()
        if not can:
            continue
        variants = row.get("variants") or []
        best_text = can
        best_count = -1
        best_timed = -1
        for v in variants:
            text = str(v.get("relation_text", "")).strip()
            if not text:
                continue
            cnt = int(v.get("count", 0) or 0)
            timed = int(v.get("timed_count", 0) or 0)
            key = (timed, cnt)
            if key > (best_timed, best_count):
                best_timed, best_count = key
                best_text = text
        can_to_surface[can] = best_text

    pairs = [(can, can_to_surface.get(can, can)) for can in target_canonical]
    if max_relations > 0 and len(pairs) > max_relations:
        random.Random(seed).shuffle(pairs)
        pairs = pairs[:max_relations]
        pairs = sorted(pairs, key=lambda x: x[0])
    return pairs


# ---------------------------------------------------------------------------
# Core function (shared by CLI and convenience wrapper)
# ---------------------------------------------------------------------------

def pretrain_gate_from_data(
    pretrain_data_dir: str,
    embedding: str,
    output: str | None = None,
    epochs: int = 100,
    learning_rate: float = 5e-4,
    embedding_dim: int = 64,
    embedding_base_url: str | None = None,
    azure_embedding_endpoint: str | None = None,
    embedding_batch_size: int = 64,
    changed_weight: float | None = None,
    unchanged_weight: float | None = None,
    max_rows: int = 300_000,
    max_relations: int = 0,
    inspect_top_k: int = 20,
    seed: int = 13,
    verbose: int = 1,
    debug_dir: str | None = None,
) -> str:
    """Pretrain the semantic speed gate MLP from pre-mined transition data.

    Args:
        pretrain_data_dir: Directory containing ``transition_observations.jsonl``
            and ``relation_nodes.jsonl``.
        embedding: Text embedding model name (determines MLP input dimension).
        output: Path to save the checkpoint. Auto-generated if ``None``.
        epochs: Training epochs.
        learning_rate: Learning rate for the gate MLP.
        embedding_dim: KGE embedding dimension (not text embedding dim).
        embedding_base_url: Custom API base URL for embeddings.
        azure_embedding_endpoint: Azure embedding endpoint.
        embedding_batch_size: Batch size for relation text embeddings.
        changed_weight: Optional class weight for changed=1 samples.
        unchanged_weight: Optional class weight for changed=0 samples.
        max_rows: Maximum transition observations to use.
        max_relations: Maximum relations (0 = all).
        inspect_top_k: Number of relations to show in diagnostics.
        seed: Random seed.
        verbose: 0=silent, 1=progress, 2=debug.
        debug_dir: Directory for debug artifacts. ``None`` to auto-create.

    Returns:
        Path to the saved gate checkpoint.
    """
    random.seed(seed)

    if not os.path.isdir(pretrain_data_dir):
        raise FileNotFoundError(f"Pretraining data directory not found: {pretrain_data_dir}")

    model_tag = embedding.replace("/", "_").replace("\\", "_")
    checkpoint_path = output or os.path.join(pretrain_data_dir, f"alpha_r_pretrained_{model_tag}.pt")

    if verbose:
        logger.info("Pretraining gate: embedding=%s, data=%s", embedding, pretrain_data_dir)

    # Relation texts
    rel_pairs = _choose_relation_texts(pretrain_data_dir=pretrain_data_dir, max_relations=max_relations, seed=seed)
    rel_texts = [surface for _, surface in rel_pairs]
    if not rel_texts:
        raise RuntimeError("No relation texts available for pretraining.")

    if verbose:
        logger.info("Found %d relations", len(rel_texts))

    facts = [(f"__gate_h_{i}", rel_text, f"__gate_t_{i}") for i, rel_text in enumerate(rel_texts)]

    # Embedding model
    base_cfg = BaseConfig()
    base_cfg.embedding_model_name = embedding
    base_cfg.embedding_batch_size = embedding_batch_size
    if embedding_base_url:
        base_cfg.embedding_base_url = embedding_base_url
    if azure_embedding_endpoint:
        base_cfg.azure_embedding_endpoint = azure_embedding_endpoint

    EmbCls = _get_embedding_model_class(embedding)
    emb_model = EmbCls(global_config=base_cfg, embedding_model_name=embedding)

    fast_cfg = TKGEConfig(
        temporal_mode="romem",
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        steps_per_update=0,
        batch_size=min(256, max(1, len(facts))),
        use_lora=False,
        use_time_contrastive=False,
    )

    encoder = TKGEEncoder.from_facts(
        facts=facts,
        config=fast_cfg,
        verbose=verbose,
        relation_embedder=lambda texts: emb_model.batch_encode(texts),
    )

    result = encoder.pretrain_time_gate_from_artifacts(
        pretrain_data_dir=pretrain_data_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        changed_weight=changed_weight,
        unchanged_weight=unchanged_weight,
        max_rows=max_rows,
        checkpoint_path=checkpoint_path,
    )

    if verbose:
        logger.info("Saved gate checkpoint to %s", checkpoint_path)
        encoder.inspect_time_gate(top_k=inspect_top_k, print_report=True)

    # Debug artifacts
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
        _write_debug_artifacts(encoder, rel_pairs, result, debug_dir, embedding, checkpoint_path, inspect_top_k)

    return checkpoint_path


def _write_debug_artifacts(
    encoder: TKGEEncoder,
    rel_pairs: list[tuple[str, str]],
    result: dict,
    debug_dir: str,
    embedding_model: str,
    checkpoint_path: str,
    inspect_top_k: int,
) -> None:
    """Write detailed debug artifacts for inspection."""
    inspect_rows = encoder.inspect_time_gate(top_k=inspect_top_k, print_report=False)

    encoder._ensure_relation_text_embeddings()
    assert encoder.relation_text_embeddings is not None
    assert encoder.model is not None
    device = torch.device("cpu")
    with torch.no_grad():
        rel_text = encoder.relation_text_embeddings.to(device)
        alpha = encoder.model.time_gate_alpha(rel_text).squeeze(-1).detach().cpu().tolist()
        scale = (
            float(encoder.model.log_time_scale.exp().detach().cpu())
            if hasattr(encoder.model, "log_time_scale")
            else (1.0 / 86400.0)
        )

    alpha_rows: list[dict] = []
    for rid, a in enumerate(alpha):
        rel = encoder.kg.id2relation.get(rid, str(rid))
        eff_scale = scale * max(float(a), 1e-12)
        period_days = (2.0 * 3.141592653589793) / eff_scale / 86400.0
        alpha_rows.append({
            "relation_id": int(rid),
            "relation": rel,
            "alpha_r": float(a),
            "effective_period_days": float(period_days),
            "effective_period_years": float(period_days / 365.25),
        })

    alpha_sorted = sorted(alpha_rows, key=lambda x: x["alpha_r"])
    k = max(1, inspect_top_k)
    debug_payload = {
        "checkpoint_out": checkpoint_path,
        "pretrain_result": result,
        "embedding_model": embedding_model,
        "num_relation_texts": len(rel_pairs),
        "alpha_stats": {
            "min": float(alpha_sorted[0]["alpha_r"]),
            "max": float(alpha_sorted[-1]["alpha_r"]),
            "mean": float(sum(x["alpha_r"] for x in alpha_sorted) / max(1, len(alpha_sorted))),
        },
        "slowest_k": alpha_sorted[:k],
        "fastest_k": list(reversed(alpha_sorted[-k:])),
        "inspect_rows": inspect_rows,
    }

    _write_json(os.path.join(debug_dir, "pretrain_debug_summary.json"), debug_payload)
    _write_jsonl(os.path.join(debug_dir, "alpha_all_relations.jsonl"), alpha_rows)
    _write_jsonl(
        os.path.join(debug_dir, "relation_vocab_used.jsonl"),
        [{"canonical_relation": can, "relation_text": surf} for can, surf in rel_pairs],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain RoMem alpha_r from transition observations.")
    parser.add_argument("--pretrain-data-dir", default="outputs/pretrain_gate_data")
    parser.add_argument("--checkpoint-out", default="")
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--azure-embedding-endpoint", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64, help="KGE embedding dim (not text embedding dim).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--changed-weight", type=float, default=None)
    parser.add_argument("--unchanged-weight", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--max-relations", type=int, default=0)
    parser.add_argument("--inspect-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    run_tag = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    debug_root = args.debug_dir or os.path.join(args.pretrain_data_dir, "debug_gate_pretrain")
    debug_dir = os.path.join(debug_root, run_tag)

    checkpoint_path = pretrain_gate_from_data(
        pretrain_data_dir=args.pretrain_data_dir,
        embedding=args.embedding_model,
        output=args.checkpoint_out or None,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        embedding_base_url=args.embedding_base_url,
        azure_embedding_endpoint=args.azure_embedding_endpoint,
        embedding_batch_size=args.embedding_batch_size,
        changed_weight=args.changed_weight,
        unchanged_weight=args.unchanged_weight,
        max_rows=args.max_rows,
        max_relations=args.max_relations,
        inspect_top_k=args.inspect_top_k,
        seed=args.seed,
        verbose=args.verbose,
        debug_dir=debug_dir,
    )

    print(f"\n=== Pretraining complete ===")
    print(f"  embedding_model : {args.embedding_model}")
    print(f"  checkpoint      : {checkpoint_path}")
    print(f"  debug_dir       : {debug_dir}")


if __name__ == "__main__":
    main()
