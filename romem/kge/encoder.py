from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Callable

import math
import datetime
import random
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib

from .config import TKGEConfig
from .knowledge_graph import KnowledgeGraph
from .model.distmult import DistMultModel
from .model.romem_distmult import RoMemDistMultModel, _rotate
from .model.romem_chronor import RoMemChronoRModel
from .time_utils import parse_time_text, time_to_scalar, QueryTime
from ..utils.misc_utils import text_processing

_SECONDS_PER_YEAR = 365.25 * 86400.0

_REL_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "in",
    "on",
    "at",
    "for",
    "of",
    "by",
    "from",
    "with",
    "and",
    "or",
    "into",
    "onto",
    "over",
    "under",
    "upon",
    "within",
}

def _to_triple(item) -> Tuple[str, str, str]:
    def _norm(v: str) -> str:
        return text_processing(str(v))

    if hasattr(item, "triple"):
        triple = getattr(item, "triple")
        if isinstance(triple, (list, tuple)) and len(triple) == 3:
            return _norm(triple[0]), _norm(triple[1]), _norm(triple[2])
    if isinstance(item, (list, tuple)) and len(item) == 3:
        return _norm(item[0]), _norm(item[1]), _norm(item[2])
    if isinstance(item, str):
        cleaned = item.strip().strip("[]()")
        parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
        if len(parts) >= 3:
            return _norm(parts[0]), _norm(parts[1]), _norm(parts[2])
    raise ValueError(f"Cannot convert to triple: {item}")


def _to_triple_time(item, time_source: str) -> Tuple[Tuple[str, str, str], float, bool]:
    """
    Convert an input item to (triple, time_scalar, has_happen_time).
    - For TimedTriple-like objects: uses happen_time/system_time according to time_source.
    - For plain triples: returns time 0.0.
    """
    triple = _to_triple(item)
    happen_time = getattr(item, "happen_time", "")
    system_time = getattr(item, "system_time", "")
    dt = parse_time_text(happen_time=happen_time, obs_time=system_time, mode=time_source)
    return triple, time_to_scalar(dt), bool(str(happen_time).strip())


def _norm_relation_surface(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith(" inv"):
        text = text[:-4].strip()
    if text.endswith("_inv"):
        text = text[:-4].strip()
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


def _canonical_relation_for_gate(rel_surface: str) -> str:
    rel = _norm_relation_surface(rel_surface)
    toks = [t for t in rel.split() if t]
    reduced = []
    for t in toks:
        if t in _REL_STOPWORDS:
            continue
        reduced.append(_simple_stem(t))
    if not reduced:
        reduced = [_simple_stem(t) for t in toks]
    can = " ".join(reduced).strip()
    return can or rel


@dataclass
class TKGEEncoder:
    """
    Lightweight fact encoder using a DistMult model.

    NOTE: This is a minimal training loop (margin-based) to demonstrate integration.
    Replace/extend with the full TKGE trainer/loss as needed.
    """

    config: TKGEConfig
    kg: KnowledgeGraph
    model: Optional[DistMultModel] = None
    verbose: int = 0
    verbose_epoch_interval: int = 5
    optimizer: Optional[torch.optim.Optimizer] = None
    lora_ent: Optional[loralib.Embedding] = None
    lora_rel: Optional[loralib.Embedding] = None
    train_calls: int = 0
    relation_embedder: Optional[Callable[[List[str]], "np.ndarray"]] = None
    relation_text_embeddings: Optional[torch.Tensor] = None
    relation_text_dim: Optional[int] = None
    _time_gate_stage: Optional[str] = None

    @classmethod
    def from_facts(
        cls,
        facts: List[object],
        config: Optional[TKGEConfig] = None,
        verbose: int = 0,
        verbose_epoch_interval: int = 5,
        relation_embedder: Optional[Callable[[List[str]], "np.ndarray"]] = None,
    ) -> "TKGEEncoder":
        cfg = config or TKGEConfig()
        triples_with_time = [_to_triple_time(f, cfg.time_source) for f in facts]
        triples = [t for t, _, _ in triples_with_time]
        kg = KnowledgeGraph.from_triples(triples=triples, config=cfg)
        # If temporal, register times as well (for training).
        if cfg.temporal_mode == "romem" and triples_with_time:
            kg.fact_times = [ts for _, ts, _ in triples_with_time]
            kg.fact_has_happen = [hh for _, _, hh in triples_with_time]
        encoder = cls(config=cfg, kg=kg, verbose=verbose, verbose_epoch_interval=verbose_epoch_interval, relation_embedder=relation_embedder)
        if not triples:
            # Defer model initialization until we have relations to embed.
            return encoder
        if cfg.temporal_mode == "romem":
            encoder._ensure_relation_text_embeddings()
        encoder._maybe_init_model()
        encoder._train()
        return encoder

    def _relation_text(self, rel: str) -> str:
        rel = str(rel).strip()
        if rel.endswith("_inv"):
            rel = rel[:-4]
        return rel.replace("_", " ").strip()

    def _ensure_relation_text_embeddings(self) -> None:
        if self.config.temporal_mode != "romem":
            return
        if self.relation_embedder is None:
            raise RuntimeError("Pure semantic gate requires a relation_embedder.")
        num_rel = max(0, self.kg.num_relations)
        if num_rel == 0:
            return
        if self.relation_text_embeddings is None:
            start = 0
            existing = None
        else:
            start = int(self.relation_text_embeddings.shape[0])
            existing = self.relation_text_embeddings
        if start >= num_rel:
            return
        texts = [self._relation_text(self.kg.id2relation[i]) for i in range(start, num_rel)]
        if not texts:
            return
        emb = self.relation_embedder(texts)
        if emb is None:
            raise RuntimeError("relation_embedder returned None.")
        emb_t = torch.tensor(emb, dtype=torch.float32)
        if emb_t.ndim == 1:
            emb_t = emb_t.unsqueeze(0)
        if self.relation_text_dim is None:
            self.relation_text_dim = int(emb_t.shape[1])
        if existing is None:
            self.relation_text_embeddings = emb_t
        else:
            self.relation_text_embeddings = torch.cat([existing, emb_t], dim=0)

    def _relation_text_for_ids(self, r_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.config.temporal_mode != "romem":
            raise RuntimeError("Relation text embeddings requested outside RoMem mode.")
        self._ensure_relation_text_embeddings()
        if self.relation_text_embeddings is None:
            raise RuntimeError("Relation text embeddings not initialized.")
        r_cpu = r_idx.detach().cpu()
        rel_text = self.relation_text_embeddings.index_select(0, r_cpu)
        return rel_text.to(device)

    @staticmethod
    def _read_jsonl(path: str) -> list[dict]:
        rows: list[dict] = []
        if not os.path.exists(path):
            return rows
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
                    rows.append(row)
        return rows

    def _relation_id_to_canonical(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for rid in range(int(self.kg.num_relations)):
            rel_raw = self.kg.id2relation.get(rid, str(rid))
            rel_text = self._relation_text(rel_raw)
            out[rid] = _canonical_relation_for_gate(rel_text)
        return out

    def _resolve_relation_ids(self, relation_query: str) -> list[int]:
        if relation_query is None:
            return []
        query_norm = self._relation_text(text_processing(str(relation_query)))
        query_can = _canonical_relation_for_gate(query_norm)
        matched: list[int] = []
        for rid in range(int(self.kg.num_relations)):
            rel_raw = self.kg.id2relation.get(rid, str(rid))
            rel_text = self._relation_text(rel_raw)
            rel_can = _canonical_relation_for_gate(rel_text)
            if rel_text == query_norm or rel_can == query_can:
                matched.append(rid)
        return sorted(set(matched))

    def save_time_gate_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        if self.config.temporal_mode != "romem":
            raise RuntimeError("Time-gate checkpoint is only available in RoMem mode.")
        if self.model is None or not hasattr(self.model, "speed_mlp"):
            raise RuntimeError("Temporal gate is not initialized.")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        payload = {
            "speed_mlp": {k: v.detach().cpu() for k, v in self.model.speed_mlp.state_dict().items()},
            "relation_text_dim": int(self.relation_text_dim or 0),
            "created_at_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        torch.save(payload, checkpoint_path)
        return {
            "checkpoint_path": checkpoint_path,
            "relation_text_dim": int(self.relation_text_dim or 0),
            "num_gate_params": int(sum(v.numel() for v in payload["speed_mlp"].values())),
        }

    def load_time_gate_checkpoint(self, checkpoint_path: str, strict_dim: bool = True) -> dict[str, Any]:
        if self.config.temporal_mode != "romem":
            raise RuntimeError("Time-gate checkpoint is only available in RoMem mode.")
        self._maybe_init_model()
        if self.model is None or not hasattr(self.model, "speed_mlp"):
            raise RuntimeError("Temporal gate is not initialized.")
        obj = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(obj, dict) and "speed_mlp" in obj:
            state = obj["speed_mlp"]
            ckpt_dim = int(obj.get("relation_text_dim", 0) or 0)
        else:
            state = obj
            ckpt_dim = 0
        cur_dim = int(self.relation_text_dim or 0)
        if strict_dim and ckpt_dim > 0 and cur_dim > 0 and ckpt_dim != cur_dim:
            raise ValueError(
                f"relation_text_dim mismatch: checkpoint={ckpt_dim}, current={cur_dim}. "
                "Use strict_dim=False to ignore."
            )
        self.model.speed_mlp.load_state_dict(state, strict=True)
        return {
            "checkpoint_path": checkpoint_path,
            "checkpoint_relation_text_dim": ckpt_dim,
            "current_relation_text_dim": cur_dim,
        }

    # ── Full model checkpoint (for continuous training) ────────────

    def save_full_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Save complete TKGE model state to disk for later resumption."""
        if self.model is None:
            raise RuntimeError("Model is not initialized — nothing to save.")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        payload: dict[str, Any] = {
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "train_calls": self.train_calls,
            "num_entities": self.kg.num_entities,
            "num_relations": self.kg.num_relations,
            "created_at_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        }
        if self.lora_ent is not None:
            payload["lora_ent"] = {k: v.detach().cpu() for k, v in self.lora_ent.state_dict().items()}
        if self.lora_rel is not None:
            payload["lora_rel"] = {k: v.detach().cpu() for k, v in self.lora_rel.state_dict().items()}
        torch.save(payload, checkpoint_path)
        num_params = sum(v.numel() for v in payload["model_state_dict"].values())
        return {
            "checkpoint_path": checkpoint_path,
            "num_params": num_params,
            "train_calls": self.train_calls,
        }

    def load_full_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        """Load complete TKGE model state from a previous checkpoint."""
        self._maybe_init_model()
        if self.model is None:
            raise RuntimeError("Model is not initialized — cannot load checkpoint.")
        obj = torch.load(checkpoint_path, map_location="cpu")
        ckpt_ent = obj.get("num_entities", 0)
        ckpt_rel = obj.get("num_relations", 0)
        if ckpt_ent != self.kg.num_entities or ckpt_rel != self.kg.num_relations:
            raise ValueError(
                f"KG dimension mismatch: checkpoint has {ckpt_ent} entities / {ckpt_rel} relations, "
                f"current KG has {self.kg.num_entities} / {self.kg.num_relations}. "
                "Cannot load checkpoint from a different graph."
            )
        self.model.load_state_dict(obj["model_state_dict"])
        self.train_calls = obj.get("train_calls", 0)
        if "lora_ent" in obj and self.lora_ent is not None:
            self.lora_ent.load_state_dict(obj["lora_ent"])
        if "lora_rel" in obj and self.lora_rel is not None:
            self.lora_rel.load_state_dict(obj["lora_rel"])
        return {
            "checkpoint_path": checkpoint_path,
            "train_calls": self.train_calls,
            "num_entities": ckpt_ent,
            "num_relations": ckpt_rel,
        }

    def pretrain_time_gate_from_artifacts(
        self,
        pretrain_data_dir: str,
        epochs: int = 100,
        learning_rate: float | None = None,
        changed_weight: float | None = None,
        unchanged_weight: float | None = None,
        max_rows: int = 300000,
        checkpoint_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Pretrain alpha_r (speed_mlp) from purely self-supervised transition artifacts.

        Input:
        - transition_observations.jsonl with per-slot adjacent observations and changed=0/1
          where changed is whether the counterpart switched between t_prev and t_cur.

        Objective:
        - theta = alpha_r * omega * Δt
        - p_change = 1 - exp(-theta)
        - BCE(y=changed, p_change), weighted for class balance
        """
        if self.config.temporal_mode != "romem":
            return {"status": "skipped", "reason": "temporal_mode!=romem"}
        self._maybe_init_model()
        if self.model is None or not hasattr(self.model, "speed_mlp"):
            return {"status": "skipped", "reason": "temporal gate not initialized"}
        self._ensure_relation_text_embeddings()
        if self.relation_text_embeddings is None or self.kg.num_relations <= 0:
            return {"status": "skipped", "reason": "no relation embeddings available"}

        transition_rows = self._read_jsonl(os.path.join(pretrain_data_dir, "transition_observations.jsonl"))
        if not transition_rows:
            return {
                "status": "skipped",
                "reason": "transition_observations.jsonl not found or empty",
                "pretrain_data_dir": pretrain_data_dir,
            }

        rel_can = self._relation_id_to_canonical()
        can_to_rel_ids: dict[str, list[int]] = {}
        for rid, can in rel_can.items():
            can_to_rel_ids.setdefault(can, []).append(rid)

        obs_rel_ids: list[int] = []
        obs_delta_seconds: list[float] = []
        obs_changed: list[float] = []
        obs_base_weights: list[float] = []

        # Read rows directly (pure self-supervision: no predefined static/dynamic classes).
        for row in transition_rows:
            can = _canonical_relation_for_gate(str(row.get("canonical_relation", "")))
            rel_ids = can_to_rel_ids.get(can, [])
            if not rel_ids:
                continue
            delta_days = float(row.get("delta_days", 0.0) or 0.0)
            delta_sec = max(0.01, delta_days)  # keep in days, not seconds
            y = 1.0 if int(row.get("changed", 0) or 0) != 0 else 0.0
            slot_obs = 0.0
            slot_stats = row.get("slot_stats") or {}
            try:
                slot_obs = float(slot_stats.get("n_obs", 0) or 0)
            except Exception:
                slot_obs = 0.0
            w = math.log1p(max(1.0, slot_obs))
            for rid in rel_ids:
                obs_rel_ids.append(int(rid))
                obs_delta_seconds.append(delta_sec)
                obs_changed.append(y)
                obs_base_weights.append(w)

        if max_rows > 0 and len(obs_rel_ids) > max_rows:
            idxs = list(range(len(obs_rel_ids)))
            random.shuffle(idxs)
            idxs = idxs[: max_rows]
            obs_rel_ids = [obs_rel_ids[i] for i in idxs]
            obs_delta_seconds = [obs_delta_seconds[i] for i in idxs]
            obs_changed = [obs_changed[i] for i in idxs]
            obs_base_weights = [obs_base_weights[i] for i in idxs]

        if not obs_rel_ids:
            return {
                "status": "skipped",
                "reason": "no overlapping pretrain relations",
                "num_kg_relations": int(self.kg.num_relations),
                "num_observations": 0,
            }

        device = torch.device("cpu")
        self.model.to(device)

        obs_id_t = torch.tensor(obs_rel_ids, dtype=torch.long, device=device)
        obs_dt_t = torch.tensor(obs_delta_seconds, dtype=torch.float32, device=device)  # in days
        obs_y_t = torch.tensor(obs_changed, dtype=torch.float32, device=device)
        obs_base_w_t = torch.tensor(obs_base_weights, dtype=torch.float32, device=device)

        changed_count = float((obs_y_t > 0.5).sum().item())
        unchanged_count = float((obs_y_t <= 0.5).sum().item())
        if changed_weight is None:
            changed_weight = (changed_count + unchanged_count) / max(1.0, 2.0 * changed_count)
        if unchanged_weight is None:
            unchanged_weight = (changed_count + unchanged_count) / max(1.0, 2.0 * unchanged_count)
        class_w_t = torch.where(
            obs_y_t > 0.5,
            torch.full_like(obs_y_t, float(changed_weight)),
            torch.full_like(obs_y_t, float(unchanged_weight)),
        )
        obs_w_t = obs_base_w_t * class_w_t

        prev_requires_grad = {name: bool(param.requires_grad) for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("speed_mlp.")

        gate_params = [p for p in self.model.speed_mlp.parameters() if p.requires_grad]
        if not gate_params:
            for name, param in self.model.named_parameters():
                param.requires_grad = prev_requires_grad.get(name, True)
            return {"status": "skipped", "reason": "no trainable gate params"}

        # Learnable pretrain-time scale (in log-space).
        # Initialised so that alpha=0.5 at median_delta_days gives p_change≈0.5:
        #   ln(2) / (0.5 * median_dt) ≈ initial scale
        median_dt = float(obs_dt_t.median().clamp(min=1.0).item())
        init_scale = math.log(2.0) / (0.5 * median_dt)
        log_pretrain_scale = torch.nn.Parameter(
            torch.tensor(math.log(max(init_scale, 1e-8)), dtype=torch.float32, device=device)
        )

        lr = float(learning_rate if learning_rate is not None else self.config.learning_rate)
        optimizer = torch.optim.Adam(list(gate_params) + [log_pretrain_scale], lr=lr)
        loss_history: list[float] = []

        try:
            for epoch in range(max(1, int(epochs))):
                optimizer.zero_grad()
                rel_text = self._relation_text_for_ids(obs_id_t, device)
                alpha = self.model.time_gate_alpha(rel_text).squeeze(-1)
                pretrain_scale = log_pretrain_scale.exp()
                theta = alpha * pretrain_scale * obs_dt_t  # obs_dt_t in days
                theta = theta.clamp(min=0.0, max=50.0)
                p_change = 1.0 - torch.exp(-theta)
                p_change = p_change.clamp(min=1e-6, max=1.0 - 1e-6)
                per_row_bce = -(obs_y_t * torch.log(p_change) + (1.0 - obs_y_t) * torch.log(1.0 - p_change))
                total_loss = (per_row_bce * obs_w_t).sum() / obs_w_t.sum().clamp(min=1.0)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(gate_params) + [log_pretrain_scale], max_norm=1.0)
                optimizer.step()

                loss_val = float(total_loss.detach().cpu())
                loss_history.append(loss_val)
                if self.verbose and self.verbose >= 2 and ((epoch + 1) % 10 == 0):
                    print(
                        f"[TKGE][gate_pretrain] epoch={epoch+1}/{epochs} loss={loss_val:.6f} "
                        f"scale={float(pretrain_scale.detach()):.6f} "
                        f"rows={obs_id_t.numel()} changed={changed_count:.0f} unchanged={unchanged_count:.0f}"
                    )
        finally:
            for name, param in self.model.named_parameters():
                param.requires_grad = prev_requires_grad.get(name, True)
            self.optimizer = self._create_optimizer()

        ckpt_meta = None
        if checkpoint_path:
            ckpt_meta = self.save_time_gate_checkpoint(checkpoint_path)

        final_pretrain_scale = float(log_pretrain_scale.exp().detach().cpu())
        diagnostics: dict[str, float] = {}
        with torch.no_grad():
            rel_text_eval = self._relation_text_for_ids(obs_id_t, device)
            alpha_eval = self.model.time_gate_alpha(rel_text_eval).squeeze(-1)
            theta_eval = (alpha_eval * final_pretrain_scale * obs_dt_t).clamp(min=0.0, max=50.0)
            p_eval = (1.0 - torch.exp(-theta_eval)).clamp(min=1e-6, max=1.0 - 1e-6)
            bce_eval = -(obs_y_t * torch.log(p_eval) + (1.0 - obs_y_t) * torch.log(1.0 - p_eval))
            if obs_y_t.numel() > 0:
                diagnostics["mean_pred_change_prob"] = float(p_eval.mean().cpu())
                diagnostics["mean_bce"] = float(bce_eval.mean().cpu())
                changed_mask = obs_y_t > 0.5
                unchanged_mask = ~changed_mask
                if changed_mask.any().item():
                    diagnostics["mean_pred_change_prob_when_changed"] = float(p_eval[changed_mask].mean().cpu())
                    diagnostics["mean_bce_when_changed"] = float(bce_eval[changed_mask].mean().cpu())
                if unchanged_mask.any().item():
                    diagnostics["mean_pred_change_prob_when_unchanged"] = float(p_eval[unchanged_mask].mean().cpu())
                    diagnostics["mean_bce_when_unchanged"] = float(bce_eval[unchanged_mask].mean().cpu())

        return {
            "status": "ok",
            "pretrain_data_dir": pretrain_data_dir,
            "epochs": int(max(1, int(epochs))),
            "learning_rate": lr,
            "pretrain_scale_init": init_scale,
            "pretrain_scale_final": final_pretrain_scale,
            "num_kg_relations": int(self.kg.num_relations),
            "num_observations": int(len(obs_rel_ids)),
            "num_changed_observations": int(changed_count),
            "num_unchanged_observations": int(unchanged_count),
            "changed_weight": float(changed_weight),
            "unchanged_weight": float(unchanged_weight),
            "loss_initial": float(loss_history[0]) if loss_history else None,
            "loss_final": float(loss_history[-1]) if loss_history else None,
            "loss_history": [float(x) for x in loss_history],
            "diagnostics": diagnostics,
            "checkpoint": ckpt_meta,
        }

    def inspect_time_gate(
        self,
        relations: Optional[List[str]] = None,
        top_k: int = 10,
        print_report: bool = True,
    ) -> list[dict[str, Any]]:
        if self.config.temporal_mode != "romem":
            raise RuntimeError("Time-gate inspection is only available in RoMem mode.")
        self._maybe_init_model()
        self._ensure_relation_text_embeddings()
        if self.model is None or self.relation_text_embeddings is None:
            return []

        device = torch.device("cpu")
        self.model.to(device)
        with torch.no_grad():
            rel_text = self.relation_text_embeddings.to(device)
            alpha = self.model.time_gate_alpha(rel_text).squeeze(-1).detach().cpu().tolist()
            scale = (
                float(self.model.log_time_scale.exp().detach().cpu())
                if hasattr(self.model, "log_time_scale")
                else 1.0 / 86400.0
            )
        rows: list[dict[str, Any]] = []
        for rid, a in enumerate(alpha):
            rel_raw = self.kg.id2relation.get(rid, str(rid))
            rel_text = self._relation_text(rel_raw)
            rel_can = _canonical_relation_for_gate(rel_text)
            eff_scale = scale * max(float(a), 1e-12)
            period_days = (2.0 * math.pi) / eff_scale / 86400.0
            rows.append(
                {
                    "relation_id": int(rid),
                    "relation": rel_raw,
                    "relation_text": rel_text,
                    "canonical_relation": rel_can,
                    "alpha_r": float(a),
                    "effective_period_days": float(period_days),
                    "effective_period_years": float(period_days / 365.25),
                }
            )

        selected: list[dict[str, Any]] = []
        if relations:
            seen_ids = set()
            for q in relations:
                for rid in self._resolve_relation_ids(q):
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    selected.append(rows[rid])
        else:
            k = max(1, int(top_k))
            sorted_rows = sorted(rows, key=lambda x: x["alpha_r"])
            selected = sorted_rows[:k] + list(reversed(sorted_rows[-k:]))

        if print_report:
            print("[TKGE][gate_inspect] relation alpha_r effective_period_years")
            for row in selected:
                print(
                    "[TKGE][gate_inspect] "
                    f"rel={row['relation']} alpha_r={row['alpha_r']:.6f} "
                    f"period_years={row['effective_period_years']:.3f} "
                    f"canonical={row['canonical_relation']}"
                )
        return selected

    def _create_optimizer(self, params=None) -> torch.optim.Optimizer:
        """Create backbone-dependent optimizer: Adagrad for ChronoR, Adam for DistMult."""
        if params is None:
            params = self._trainable_parameters()
        backbone = getattr(self.config, "temporal_backbone", "distmult")
        if self.config.temporal_mode == "romem" and backbone == "chronor":
            return torch.optim.Adagrad(params, lr=self.config.learning_rate)
        return torch.optim.Adam(params, lr=self.config.learning_rate)

    def _maybe_init_model(self) -> None:
        num_rel = max(1, self.kg.num_relations)
        num_ent = max(1, self.kg.num_entities)
        emb_dim = getattr(self.config, "embedding_dim", 64)
        gamma = float(getattr(self.config, "gamma", 200.0))
        rebuild_optimizer = False
        if self.kg.num_relations == 0 or self.kg.num_entities == 0:
            # Nothing to initialize yet; wait until relations/entities exist.
            return
        if self.model is None:
            if self.config.temporal_mode == "romem":
                self._ensure_relation_text_embeddings()
                if self.relation_text_dim is None:
                    raise RuntimeError("Relation text dimension is required for semantic time gating.")
                backbone = getattr(self.config, "temporal_backbone", "distmult")
                if backbone == "chronor":
                    k = int(getattr(self.config, "chronor_k", 3))
                    self.model = RoMemChronoRModel(
                        num_entities=num_ent,
                        num_relations=num_rel,
                        dim=emb_dim,
                        rel_text_dim=self.relation_text_dim,
                        k=k,
                        gamma=gamma,
                        force_time_gate_one=bool(getattr(self.config, "force_time_gate_one", False)),
                    )
                    if self.verbose:
                        print(f"[TKGE] init backbone=chronor k={k} dim={emb_dim} total_dim={k*2*emb_dim} entities={num_ent} relations={num_rel}")
                else:
                    self.model = RoMemDistMultModel(
                        num_entities=num_ent,
                        num_relations=num_rel,
                        dim=emb_dim,
                        rel_text_dim=self.relation_text_dim,
                        gamma=gamma,
                        force_time_gate_one=bool(getattr(self.config, "force_time_gate_one", False)),
                    )
                    if self.verbose:
                        print(f"[TKGE] init backbone=distmult dim={emb_dim} entities={num_ent} relations={num_rel}")
            else:
                self.model = DistMultModel(num_entities=num_ent, num_relations=num_rel, dim=emb_dim, gamma=gamma)
                if self.verbose:
                    print(f"[TKGE] init backbone=distmult_static dim={emb_dim} entities={num_ent} relations={num_rel}")
            if self.config.use_lora:
                self._init_lora_layers(emb_dim)
            rebuild_optimizer = True
        else:
            # Expand embeddings if needed without reinitializing existing weights.
            # Use uniform init matching KGE-style range for new embeddings.
            if self.model.ent_emb.num_embeddings < num_ent:
                out_dim = self.model.ent_emb.embedding_dim
                emb_range = (gamma + 2.0) / out_dim
                new_emb = nn.Embedding(num_ent, out_dim)
                nn.init.uniform_(new_emb.weight, -emb_range, emb_range)
                new_emb.weight.data[: self.model.ent_emb.num_embeddings] = self.model.ent_emb.weight.data
                self.model.ent_emb = new_emb
                if self.config.use_lora:
                    self.model.ent_emb.weight.requires_grad = False
                rebuild_optimizer = True
            if self.model.rel_emb.num_embeddings < num_rel:
                out_dim = self.model.rel_emb.embedding_dim
                emb_range = (gamma + 2.0) / out_dim
                new_rel = nn.Embedding(num_rel, out_dim)
                nn.init.uniform_(new_rel.weight, -emb_range, emb_range)
                new_rel.weight.data[: self.model.rel_emb.num_embeddings] = self.model.rel_emb.weight.data
                self.model.rel_emb = new_rel
                if self.config.use_lora:
                    self.model.rel_emb.weight.requires_grad = False
                rebuild_optimizer = True
            # Expand inverse relation embeddings for ChronoR backbone.
            if hasattr(self.model, "rel_inv_emb") and self.model.rel_inv_emb.num_embeddings < num_rel:
                out_dim = self.model.rel_inv_emb.embedding_dim
                emb_range = (gamma + 2.0) / out_dim
                new_inv = nn.Embedding(num_rel, out_dim)
                nn.init.uniform_(new_inv.weight, -emb_range, emb_range)
                new_inv.weight.data[: self.model.rel_inv_emb.num_embeddings] = self.model.rel_inv_emb.weight.data
                self.model.rel_inv_emb = new_inv
                rebuild_optimizer = True
            if self.config.temporal_mode == "romem":
                self._ensure_relation_text_embeddings()
            if self.config.use_lora:
                expanded = self._expand_lora(num_ent, num_rel, emb_dim)
                if expanded:
                    rebuild_optimizer = True
        if self.optimizer is None:
            rebuild_optimizer = True
        if rebuild_optimizer:
            self.optimizer = self._create_optimizer()

    def _init_lora_layers(self, emb_dim: int) -> None:
        r = max(1, self.config.lora_rank)
        out_dim = self.model.ent_emb.embedding_dim
        self.lora_ent = loralib.Embedding(self.model.ent_emb.num_embeddings, out_dim, r)
        out_dim_rel = self.model.rel_emb.embedding_dim
        self.lora_rel = loralib.Embedding(self.model.rel_emb.num_embeddings, out_dim_rel, r)

    def _set_time_gate_stage(self, stage: str) -> None:
        if self.model is None or self.config.temporal_mode != "romem":
            return
        stage = str(stage).lower()
        if stage == self._time_gate_stage:
            return
        if stage == "omega":
            # Stage 1: train global omega, keep alpha fixed at 1.0.
            self.model.force_time_gate_one = True
            if hasattr(self.model, "speed_mlp"):
                for param in self.model.speed_mlp.parameters():
                    param.requires_grad = False
            if hasattr(self.model, "log_time_scale"):
                self.model.log_time_scale.requires_grad = True
            if hasattr(self.model, "log_inv_freq_base"):
                self.model.log_inv_freq_base.requires_grad = True
        elif stage == "pretrained":
            # Pretrained gate loaded from checkpoint: freeze gate, train omega + embeddings.
            self.model.force_time_gate_one = False
            if hasattr(self.model, "speed_mlp"):
                for param in self.model.speed_mlp.parameters():
                    param.requires_grad = False
            if hasattr(self.model, "log_time_scale"):
                self.model.log_time_scale.requires_grad = True
            if hasattr(self.model, "log_inv_freq_base"):
                self.model.log_inv_freq_base.requires_grad = True
        elif stage == "gate":
            # Stage 2: train alpha, optionally freeze omega.
            self.model.force_time_gate_one = False
            if hasattr(self.model, "speed_mlp"):
                for param in self.model.speed_mlp.parameters():
                    param.requires_grad = True
            if bool(getattr(self.config, "time_gate_freeze_omega_after", False)):
                if hasattr(self.model, "log_time_scale"):
                    self.model.log_time_scale.requires_grad = False
                if hasattr(self.model, "log_inv_freq_base"):
                    self.model.log_inv_freq_base.requires_grad = False
        self._time_gate_stage = stage
        self.optimizer = self._create_optimizer()
        if self.config.use_lora and self.lora_ent is not None and self.lora_rel is not None:
            # Start from identity: LoRA B=0 (A already zeroed) to avoid early instabilities.
            with torch.no_grad():
                if hasattr(self.lora_ent, "lora_B"):
                    self.lora_ent.lora_B.zero_()
                if hasattr(self.lora_rel, "lora_B"):
                    self.lora_rel.lora_B.zero_()
            # Freeze the underlying base embedding weights; train LoRA parameters only.
            self.lora_ent.weight.requires_grad = False
            self.lora_rel.weight.requires_grad = False
            self.model.ent_emb.weight.requires_grad = False
            self.model.rel_emb.weight.requires_grad = False

    def _expand_lora(self, num_ent: int, num_rel: int, emb_dim: int) -> bool:
        if self.lora_ent is None or self.lora_rel is None:
            self._init_lora_layers(emb_dim)
            return True
        r = max(1, self.config.lora_rank)
        expanded = False
        def _copy_overlap(old: nn.Module, new: nn.Module) -> None:
            old_params = dict(old.named_parameters())
            new_params = dict(new.named_parameters())
            with torch.no_grad():
                for name, old_tensor in old_params.items():
                    if name not in new_params:
                        continue
                    new_tensor = new_params[name]
                    slices = tuple(slice(0, min(a, b)) for a, b in zip(old_tensor.shape, new_tensor.shape))
                    new_tensor[slices].copy_(old_tensor[slices])

        if self.lora_ent.weight.shape[0] < num_ent:
            out_dim = self.model.ent_emb.embedding_dim
            new_ent = loralib.Embedding(num_ent, out_dim, r)
            _copy_overlap(self.lora_ent, new_ent)
            with torch.no_grad():
                if hasattr(new_ent, "lora_B"):
                    new_ent.lora_B.zero_()
            new_ent.weight.requires_grad = False
            self.lora_ent = new_ent
            expanded = True
        if self.lora_rel.weight.shape[0] < num_rel:
            out_dim = self.model.rel_emb.embedding_dim
            new_rel = loralib.Embedding(num_rel, out_dim, r)
            _copy_overlap(self.lora_rel, new_rel)
            with torch.no_grad():
                if hasattr(new_rel, "lora_B"):
                    new_rel.lora_B.zero_()
            new_rel.weight.requires_grad = False
            self.lora_rel = new_rel
            expanded = True
        return expanded

    def _trainable_parameters(self):
        if self.config.use_lora and self.lora_ent is not None and self.lora_rel is not None:
            params = []
            for module in (self.lora_ent, self.lora_rel):
                for name, param in module.named_parameters():
                    if "lora_" in name:
                        params.append(param)
            # Include any non-embedding trainable parameters in the TemporalRot model
            # (e.g., learnable time scale, frequency base, temperature, gate logits).
            if self.model is not None:
                for _, param in self.model.named_parameters():
                    if param.requires_grad:
                        params.append(param)
            return params
        return self.model.parameters()

    @staticmethod
    def _is_competing_slot(
        h_id: int,
        r_id: int,
        t_id: int,
        hr_to_tails: dict[tuple[int, int], List[int]],
        rt_to_heads: dict[tuple[int, int], List[int]],
    ) -> bool:
        tails = hr_to_tails.get((h_id, r_id), [])
        if len(tails) > 1:
            return True
        heads = rt_to_heads.get((r_id, t_id), [])
        if len(heads) > 1:
            return True
        return False

    def _score_temporal(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
        t_scalar: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Score triples using the pretrained gate alpha_r (detached, no gradients to gate)."""
        rel_text = self._relation_text_for_ids(r_idx, device)
        alpha_override = self.model.time_gate_alpha(rel_text).detach()

        return self.model.score_triples(
            h_idx,
            r_idx,
            t_idx,
            t_scalar,
            rel_text_emb=rel_text,
            alpha_override=alpha_override,
            ent_lora=self.lora_ent if self.config.use_lora else None,
            rel_lora=self.lora_rel if self.config.use_lora else None,
        )

    def _temporal_probe(
        self,
        triples_tensor: torch.Tensor,
        times_tensor: torch.Tensor,
        happen_mask: torch.Tensor,
        device: torch.device,
        n_samples: int = 5,
    ) -> None:
        """Score sampled temporal triples at their true time and at offset times."""
        import datetime

        self.model.eval()
        # Select triples that have timestamps
        temporal_idx = torch.where(happen_mask)[0]
        if temporal_idx.numel() == 0:
            return
        n = min(n_samples, temporal_idx.numel())
        perm = temporal_idx[torch.randperm(temporal_idx.numel())[:n]]

        # Collect all unique timestamps for offset probing
        unique_times = torch.unique(times_tensor[happen_mask]).sort().values
        # Pick a spread of probe times: min, 25%, 50%, 75%, max
        if unique_times.numel() >= 5:
            qt_idx = [0, unique_times.numel() // 4, unique_times.numel() // 2,
                       3 * unique_times.numel() // 4, unique_times.numel() - 1]
            probe_times = unique_times[qt_idx]
        else:
            probe_times = unique_times

        print(f"[TKGE] temporal_probe ({n} samples, {probe_times.numel()} probe times):")
        with torch.no_grad():
            for i in range(n):
                idx = perm[i].item()
                h_id, r_id, t_id = triples_tensor[idx, :3].tolist()
                true_time = float(times_tensor[idx].item())

                h_name = self.kg.id2entity.get(h_id, str(h_id))
                r_name = self.kg.id2relation.get(r_id, str(r_id))
                t_name = self.kg.id2entity.get(t_id, str(t_id))

                # Score at true time
                h_t = torch.tensor([h_id], device=device)
                r_t = torch.tensor([r_id], device=device)
                t_t = torch.tensor([t_id], device=device)
                true_t = torch.tensor([true_time], device=device)
                true_score = float(self._score_temporal(h_t, r_t, t_t, true_t, device).item())

                # Score at each probe time
                probe_scores = []
                for pt in probe_times:
                    pt_t = torch.tensor([float(pt.item())], device=device)
                    s = float(self._score_temporal(h_t, r_t, t_t, pt_t, device).item())
                    probe_scores.append((pt.item(), s))

                true_date = datetime.datetime.utcfromtimestamp(int(true_time)).strftime("%Y-%m-%d")
                print(f"  ({h_name}, {r_name}, {t_name}) @ {true_date} -> score={true_score:.3f}")
                for pt_val, pt_score in probe_scores:
                    pt_date = datetime.datetime.utcfromtimestamp(int(pt_val)).strftime("%Y-%m-%d")
                    marker = " <-- true" if abs(pt_val - true_time) < 86400 else ""
                    delta_days = (pt_val - true_time) / 86400.0
                    print(f"    @{pt_date} (Δ{delta_days:+.0f}d): score={pt_score:.3f}{marker}")
        self.model.train()

    def update_facts(self, facts: List[Tuple[str, str, str]], *, train: bool = True) -> None:
        if not facts:
            return
        if self.config.temporal_mode == "romem":
            triples_with_time = [_to_triple_time(f, self.config.time_source) for f in facts]
            changed = self.kg.add_timed_triples(
                [(h, r, t, ts, hh) for (h, r, t), ts, hh in triples_with_time]
            )
        else:
            triples = [_to_triple(f) for f in facts]
            changed = self.kg.add_triples(triples)
        if self.config.temporal_mode == "romem":
            self._ensure_relation_text_embeddings()
        self._maybe_init_model()
        if changed and train:
            self._train()

    def remove_facts(self, facts: List[Tuple[str, str, str]]) -> None:
        if not facts:
            return
        triples = [_to_triple(f) for f in facts]
        self.kg.remove_triples(triples)
        self._maybe_init_model()
        self._train()

    def encode_facts(self, facts: List[Tuple[str, str, str]], query_time: QueryTime | None = None):
        if self.model is None:
            self._maybe_init_model()

        vectors = []
        t_scalar = torch.tensor([float(query_time.unix_seconds if query_time is not None else 0.0)])
        for item in facts:
            h, r, t = _to_triple(item)
            h_id = self.kg.entity2id.get(h)
            r_id = self.kg.relation2id.get(r)
            t_id = self.kg.entity2id.get(t)
            if h_id is None or r_id is None or t_id is None:
                out_dim = int(self.model.ent_emb.embedding_dim) * 2
                vectors.append(torch.zeros(out_dim).tolist())
                continue
            dev = t_scalar.device
            if self.config.temporal_mode == "romem":
                rel_text = self._relation_text_for_ids(torch.tensor([r_id], device=dev), dev)
                vec = self.model.triple_embedding(
                    torch.tensor([h_id], device=dev),
                    torch.tensor([r_id], device=dev),
                    torch.tensor([t_id], device=dev),
                    t_scalar,
                    rel_text_emb=rel_text,
                    ent_lora=self.lora_ent if self.config.use_lora else None,
                    rel_lora=self.lora_rel if self.config.use_lora else None,
                ).squeeze(0)
            else:
                vec = self.model.triple_embedding(
                    torch.tensor([h_id], device=dev),
                    torch.tensor([r_id], device=dev),
                    torch.tensor([t_id], device=dev),
                    ent_lora=self.lora_ent if self.config.use_lora else None,
                    rel_lora=self.lora_rel if self.config.use_lora else None,
                ).squeeze(0)
            vectors.append(vec.detach().tolist())
        return vectors

    def _train(self) -> None:
        if self.model is None:
            return
        if self.config.temporal_mode == "romem":
            triples = self.kg.get_id_triples_with_times()
        else:
            triples = self.kg.get_id_triples()
        if not triples:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Detect device change (e.g. after checkpoint load to CPU) and reset optimizer
        _prev_device = next(self.model.parameters()).device
        self.model.to(device)
        if _prev_device != device:
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        gamma = float(getattr(self.config, "triple_margin", 1.0))
        adv_temp = float(getattr(self.config, "adversarial_temperature", 1.0))
        reg_weight = float(getattr(self.config, "regularization_weight", 1e-5))
        backbone = getattr(self.config, "temporal_backbone", "distmult")
        is_chronor = (self.config.temporal_mode == "romem" and backbone == "chronor")
        if self.verbose and self.train_calls == 0:
            loss_type = "CE+N3" if is_chronor else ("self-adv+L3" if self.config.temporal_mode == "romem" else "self-adv+L3(static)")
            n_ent = self.kg.num_entities
            n_rel = self.kg.num_relations
            print(f"[TKGE] training loss_path={loss_type} backbone={backbone} "
                  f"entities={n_ent} relations={n_rel} triples={len(triples)} "
                  f"device={device} ln(n_ent)={math.log(n_ent):.4f}")
        epochs = int(getattr(self.config, "steps_per_update", 10))
        num_neg = getattr(self.config, "num_negatives", 5)
        batch_size = min(self.config.batch_size, len(triples))
        if epochs <= 0:
            self.train_calls += 1
            if self.verbose:
                mode = "lora" if self.config.use_lora else "full"
                print(
                    f"[TKGE] train_skip call={self.train_calls} mode={mode} "
                    f"triples={len(triples)} reason=steps_per_update<=0"
                )
            return
        if self.config.temporal_mode == "romem":
            triples_tensor = torch.tensor([t[:3] for t in triples], dtype=torch.long, device=device)
            times_tensor = torch.tensor([t[3] for t in triples], dtype=torch.float32, device=device)
            happen_mask_tensor = torch.tensor([bool(t[4]) for t in triples], dtype=torch.bool, device=device)
        else:
            triples_tensor = torch.tensor(triples, dtype=torch.long, device=device)
        num_conflict = int(getattr(self.config, "num_conflict_negatives", 1))
        hr_to_tails: dict[tuple[int, int], List[int]] = {}
        rt_to_heads: dict[tuple[int, int], List[int]] = {}
        hr_to_times: dict[tuple[int, int], List[float]] = {}
        is_competing_tensor = None
        global_time_tensor = None
        hr_to_tails_raw: dict[tuple[int, int], set[int]] = {}
        rt_to_heads_raw: dict[tuple[int, int], set[int]] = {}
        triples_list = triples_tensor.tolist()
        for h_id, r_id, t_id in triples_list:
            hr_to_tails_raw.setdefault((h_id, r_id), set()).add(t_id)
            rt_to_heads_raw.setdefault((r_id, t_id), set()).add(h_id)
        hr_to_tails = {k: list(v) for k, v in hr_to_tails_raw.items()}
        rt_to_heads = {k: list(v) for k, v in rt_to_heads_raw.items()}
        if self.config.temporal_mode == "romem":
            is_competing_tensor = torch.tensor(
                [
                    self._is_competing_slot(h_id, r_id, t_id, hr_to_tails, rt_to_heads)
                    for h_id, r_id, t_id in triples_list
                ],
                dtype=torch.bool,
                device=device,
            )
        if self.config.temporal_mode == "romem":
            if times_tensor.numel() > 0:
                global_time_tensor = times_tensor[happen_mask_tensor] if happen_mask_tensor.any() else times_tensor
            if happen_mask_tensor.any().item():
                for (h_id, r_id, t_id), ts, hh in zip(
                    triples_tensor.tolist(), times_tensor.tolist(), happen_mask_tensor.tolist()
                ):
                    if hh:
                        hr_to_times.setdefault((h_id, r_id), set()).add(float(ts))
                hr_to_times = {k: list(v) for k, v in hr_to_times.items()}
        # --- Gate checkpoint loading ---
        # If a pretrained alpha_r checkpoint is configured, load it once
        # (on the first training call) and use the "pretrained" stage for
        # the entire training run, bypassing the two-stage omega/gate schedule.
        gate_ckpt_path = str(getattr(self.config, "time_gate_checkpoint", "") or "")
        _using_pretrained_gate = False
        if gate_ckpt_path and self.config.temporal_mode == "romem":
            if not os.path.isfile(gate_ckpt_path):
                raise FileNotFoundError(
                    f"Pretrained gate checkpoint not found: {gate_ckpt_path}\n"
                    "Either run pretraining first (bash scripts/pretrain_relation_ar.sh) "
                    "or set tkge_time_gate_checkpoint to empty string to use two-stage training."
                )
            if self._time_gate_stage != "pretrained":
                try:
                    ckpt_info = self.load_time_gate_checkpoint(gate_ckpt_path, strict_dim=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load pretrained gate checkpoint: {gate_ckpt_path}\n{e}"
                    ) from e
                if self.verbose:
                    print(
                        f"[TKGE] loaded pretrained gate checkpoint: {gate_ckpt_path} "
                        f"(ckpt_dim={ckpt_info.get('checkpoint_relation_text_dim')}, "
                        f"cur_dim={ckpt_info.get('current_relation_text_dim')})"
                    )
            _using_pretrained_gate = True

        stage1_epochs = int(getattr(self.config, "time_gate_stage1_epochs", 0) or 0)
        if self.config.temporal_mode == "romem":
            if _using_pretrained_gate:
                initial_stage = "pretrained"
            else:
                initial_stage = "omega" if stage1_epochs > 0 else "gate"
            self._set_time_gate_stage(initial_stage)
        self.train_calls += 1
        if self.verbose:
            if self.config.temporal_mode == "romem":
                num_happen = int(happen_mask_tensor.sum().item()) if "happen_mask_tensor" in locals() else 0
                if num_happen > 0:
                    unique_happen = int(torch.unique(times_tensor[happen_mask_tensor]).numel())
                else:
                    unique_happen = 0
                nonzero_times = int((times_tensor != 0).sum().item()) if "times_tensor" in locals() else 0
                print(
                    f"[TKGE] temporal_summary triples={len(triples)} "
                    f"happen={num_happen} happen_unique={unique_happen} time_nonzero={nonzero_times}"
                )
            mode = "lora" if self.config.use_lora else "full"
            print(
                f"[TKGE] train_start call={self.train_calls} mode={mode} "
                f"triples={len(triples)} emb_dim={self.config.embedding_dim} "
                f"lr={self.config.learning_rate} batch_size={self.config.batch_size} "
                f"epochs={epochs} num_neg={num_neg}"
            )
            if hasattr(self.model, "log_time_scale"):
                init_scale = float(self.model.log_time_scale.exp().detach().cpu())
                print(f"[TKGE] init_time_scale={init_scale:.6e} rad/sec")

        losses = []
        best_state = None
        best_loss = None
        best_epoch = None
        # Clear stale window-best from previous training calls
        if hasattr(self, '_window_best_loss'):
            del self._window_best_loss
        if hasattr(self, '_window_best_epoch'):
            del self._window_best_epoch
        batches_per_epoch = max(1, int(math.ceil(triples_tensor.size(0) / batch_size)))
        for epoch_idx in range(epochs):
            if self.config.temporal_mode == "romem":
                if _using_pretrained_gate:
                    desired_stage = "pretrained"
                else:
                    desired_stage = "omega" if (stage1_epochs > 0 and epoch_idx < stage1_epochs) else "gate"
                self._set_time_gate_stage(desired_stage)
            temporal_warmup = int(getattr(self.config, "temporal_warmup_epochs", 0))
            temporal_enabled = epoch_idx >= temporal_warmup
            perm = torch.randperm(triples_tensor.size(0), device=device)
            epoch_loss_sum = 0.0
            epoch_triple_sum = 0.0
            epoch_time_sum = 0.0
            epoch_reg_sum = 0.0
            epoch_batches = 0
            for batch_idx in range(0, perm.numel(), batch_size):
                idx = perm[batch_idx : batch_idx + batch_size]
                batch = triples_tensor[idx]
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                t_scalar = None
                happen_mask = None
                if self.config.temporal_mode == "romem":
                    t_scalar = times_tensor[idx]
                    happen_mask = happen_mask_tensor[idx]
                bsz = t.size(0)
                reg_loss = torch.tensor(0.0, device=device)

                if self.config.temporal_mode == "romem" and is_chronor:
                    # ── ChronoR: 1-vs-all cross-entropy (head + tail) ──
                    batch_comp_mask = (
                        is_competing_tensor[idx]
                        if is_competing_tensor is not None
                        else torch.zeros_like(h, dtype=torch.bool, device=device)
                    )
                    ce_fn = nn.CrossEntropyLoss(reduction='mean')

                    # Alpha routing (detached for base triple loss).
                    rel_text = self._relation_text_for_ids(r, device)
                    alpha = self.model.time_gate_alpha(rel_text).detach()
                    theta_s = self.model._theta(t_scalar) * alpha

                    h_emb = self.model.ent_emb(h)
                    r_emb = self.model.rel_emb(r)
                    r_inv = self.model.rel_inv_emb(r)
                    t_emb = self.model.ent_emb(t)

                    if self.config.use_lora and self.lora_ent is not None:
                        h_emb = h_emb + self.lora_ent(h)
                        t_emb = t_emb + self.lora_ent(t)
                    if self.config.use_lora and self.lora_rel is not None:
                        r_emb = r_emb + self.lora_rel(r)

                    h_rot = _rotate(h_emb, theta_s)
                    t_rot = _rotate(t_emb, theta_s)
                    all_ent = self.model.ent_emb.weight  # [N, total_dim]

                    # Tail prediction: unrotate query to score against raw entity table
                    qt = _rotate(h_rot * r_emb * r_inv, -theta_s)
                    scores_t = qt @ all_ent.T  # [B, N]
                    loss_tail = ce_fn(scores_t, t)

                    # Head prediction: symmetric unrotation trick
                    qh = _rotate(r_emb * r_inv * t_rot, -theta_s)
                    scores_h = qh @ all_ent.T  # [B, N]
                    loss_head = ce_fn(scores_h, h)

                    triple_loss = (loss_tail + loss_head) / 2

                    # N3 (L4) per-batch regularization
                    if reg_weight > 0:
                        reg_loss = reg_weight * (
                            (h_emb.abs() ** 4).sum() +
                            (r_emb.abs() ** 4).sum() +
                            (r_inv.abs() ** 4).sum() +
                            (t_emb.abs() ** 4).sum()
                        ) / bsz

                elif self.config.temporal_mode == "romem":
                    # ── DistMult backbone: self-adversarial neg sampling ──
                    batch_comp_mask = (
                        is_competing_tensor[idx]
                        if is_competing_tensor is not None
                        else torch.zeros_like(h, dtype=torch.bool, device=device)
                    )

                    neg_ent = torch.randint(0, self.model.ent_emb.num_embeddings, (bsz, num_neg), device=device)
                    if num_conflict > 0 and hr_to_tails:
                        for i in range(bsz):
                            tails = hr_to_tails.get((int(h[i].item()), int(r[i].item())))
                            if not tails or len(tails) <= 1:
                                continue
                            candidates = [x for x in tails if x != int(t[i].item())]
                            if not candidates:
                                continue
                            for j in range(min(num_conflict, num_neg)):
                                neg_ent[i, j] = random.choice(candidates)

                    # Positive score (pretrained alpha, detached).
                    pos_score_base = self._score_temporal(
                        h, r, t, t_scalar, device=device)

                    # Alternating head/tail corruption.
                    if epoch_batches % 2 == 0:
                        # Tail corruption
                        neg_h_flat = h.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_r_flat = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_t_flat = neg_ent.reshape(-1)
                        neg_time = t_scalar.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                    else:
                        # Head corruption
                        neg_h_flat = neg_ent.reshape(-1)
                        neg_r_flat = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_t_flat = t.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_time = t_scalar.unsqueeze(1).expand(-1, num_neg).reshape(-1)

                    neg_score_base = self._score_temporal(
                        neg_h_flat, neg_r_flat, neg_t_flat, neg_time, device=device
                    ).view(bsz, num_neg)

                    # Self-adversarial triple loss.
                    neg_weights = F.softmax(neg_score_base * adv_temp, dim=1).detach()
                    neg_loss = -(neg_weights * F.logsigmoid(-neg_score_base)).sum(1).mean()
                    pos_loss = -F.logsigmoid(pos_score_base).mean()
                    triple_loss = (pos_loss + neg_loss) / 2

                    # L3 global regularization.
                    if reg_weight > 0:
                        reg_loss = reg_weight * (
                            self.model.ent_emb.weight.norm(p=3) ** 3 +
                            self.model.rel_emb.weight.norm(p=3) ** 3)

                else:
                    # ── Static DistMult: self-adversarial neg sampling ──
                    neg_ent = torch.randint(0, self.model.ent_emb.num_embeddings, (bsz, num_neg), device=device)
                    pos_score = self.model.score_triples(
                        h, r, t,
                        ent_lora=self.lora_ent if self.config.use_lora else None,
                        rel_lora=self.lora_rel if self.config.use_lora else None,
                    )
                    # Alternating head/tail corruption.
                    if epoch_batches % 2 == 0:
                        neg_h_flat = h.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_r_flat = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_t_flat = neg_ent.reshape(-1)
                    else:
                        neg_h_flat = neg_ent.reshape(-1)
                        neg_r_flat = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                        neg_t_flat = t.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                    neg_score = self.model.score_triples(
                        neg_h_flat, neg_r_flat, neg_t_flat,
                        ent_lora=self.lora_ent if self.config.use_lora else None,
                        rel_lora=self.lora_rel if self.config.use_lora else None,
                    ).view(bsz, num_neg)
                    # Self-adversarial loss.
                    neg_weights = F.softmax(neg_score * adv_temp, dim=1).detach()
                    neg_loss_s = -(neg_weights * F.logsigmoid(-neg_score)).sum(1).mean()
                    pos_loss_s = -F.logsigmoid(pos_score).mean()
                    triple_loss = (pos_loss_s + neg_loss_s) / 2
                    # L3 global regularization.
                    if reg_weight > 0:
                        reg_loss = reg_weight * (
                            self.model.ent_emb.weight.norm(p=3) ** 3 +
                            self.model.rel_emb.weight.norm(p=3) ** 3)

                # Temporal supervision: time-contrastive loss (relation-gated), only when RoMem is active.
                time_loss = torch.tensor(0.0, device=device)
                if (
                    self.config.temporal_mode == "romem"
                    and bool(getattr(self.config, "use_time_contrastive", False))
                    and t_scalar is not None
                    and times_tensor.numel() > 0
                    and happen_mask is not None
                    and happen_mask.any().item()
                    and temporal_enabled
                ):
                    # Apply time-contrastive supervision for facts with known happen times.
                    valid_idx = torch.nonzero(happen_mask, as_tuple=False).squeeze(-1)
                    # If no valid samples in this batch, skip.
                    if valid_idx.numel() == 0:
                        loss = triple_loss + reg_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm=1.0)
                        self.optimizer.step()
                        loss_val = float(loss.item())
                        losses.append(loss_val)
                        epoch_loss_sum += loss_val
                        epoch_batches += 1
                        if self.verbose:
                            print(
                                f"[TKGE] epoch={epoch_idx+1}/{epochs} "
                                f"batch={epoch_batches}/{batches_per_epoch} loss={loss_val:.4f}"
                            )
                        continue

                    h_v = h[valid_idx]
                    r_v = r[valid_idx]
                    t_v = t[valid_idx]
                    t_scalar_v = t_scalar[valid_idx]

                    j = int(getattr(self.config, "num_time_negatives", 4))
                    j = max(1, j)
                    # Sample negative times from the observed time pool (with optional jitter).
                    use_hr_time_neg = bool(getattr(self.config, "time_neg_source", "hr") == "hr")
                    global_time_list = []
                    if global_time_tensor is not None and global_time_tensor.numel() > 0:
                        global_time_list = [float(v) for v in global_time_tensor.tolist()]
                    # Curriculum schedule for minimum negative gap.
                    min_days_start = getattr(self.config, "time_neg_min_days_start", 0.0)
                    min_days_end = getattr(self.config, "time_neg_min_days_end", min_days_start)
                    min_days_decay = int(getattr(self.config, "time_neg_min_days_decay_epochs", 0) or 0)
                    if min_days_decay > 0:
                        frac = min((epoch_idx + 1) / float(min_days_decay), 1.0)
                        min_days = float(min_days_start) + (float(min_days_end) - float(min_days_start)) * frac
                    else:
                        min_days = float(min_days_start)
                    min_gap_seconds = max(0.0, min_days) * 86400.0
                    if use_hr_time_neg and hr_to_times:
                        neg_times_list = []
                        for i in range(t_v.size(0)):
                            key = (int(h_v[i].item()), int(r_v[i].item()))
                            candidates = hr_to_times.get(key)
                            if candidates:
                                if len(candidates) > 1:
                                    c = [x for x in candidates if x != float(t_scalar_v[i].item())]
                                    candidates = c or candidates
                                else:
                                    candidates = []
                            if not candidates:
                                if global_time_list:
                                    t_now = float(t_scalar_v[i].item())
                                    filtered = [x for x in global_time_list if x != t_now]
                                    candidates = filtered or global_time_list
                            if not candidates:
                                candidates = [float(t_scalar_v[i].item())]
                            if min_gap_seconds > 0 and candidates:
                                t_now = float(t_scalar_v[i].item())
                                filtered = [x for x in candidates if abs(x - t_now) >= min_gap_seconds]
                                candidates = filtered or candidates
                            neg_times_list.append([random.choice(candidates) for _ in range(j)])
                        neg_times = torch.tensor(neg_times_list, dtype=torch.float32, device=device)
                    else:
                        if global_time_tensor is None or global_time_tensor.numel() == 0:
                            neg_times = t_scalar_v.unsqueeze(1).repeat(1, j)
                        else:
                            neg_times_list = []
                            for i in range(t_v.size(0)):
                                t_now = float(t_scalar_v[i].item())
                                if min_gap_seconds > 0:
                                    mask = (global_time_tensor - t_now).abs() >= min_gap_seconds
                                    pool = global_time_tensor[mask] if mask.any() else global_time_tensor
                                else:
                                    pool = global_time_tensor
                                idx = torch.randint(0, pool.size(0), (j,), device=device)
                                neg_times_list.append(pool[idx])
                            neg_times = torch.stack(neg_times_list, dim=0)
                    far_days = int(getattr(self.config, "time_neg_far_days", 0))
                    if far_days > 0:
                        # Replace one negative per example with a far offset (alternating sign).
                        offsets = torch.full_like(neg_times[:, 0], float(far_days) * 86400.0)
                        signs = torch.where(
                            (torch.arange(t_v.size(0), device=device) % 2) == 0,
                            torch.tensor(1.0, device=device),
                            torch.tensor(-1.0, device=device),
                        )
                        neg_times[:, 0] = t_scalar_v + (offsets * signs)
                    jitter_years = float(getattr(self.config, "time_neg_jitter_years", 0.0))
                    if jitter_years > 0:
                        jitter = (torch.rand_like(neg_times) * 2.0 - 1.0) * (jitter_years * _SECONDS_PER_YEAR)
                        neg_times = neg_times + jitter

                    # Weighting that avoids over-penalizing close times (persistence-friendly):
                    # w(Δt) = 1 - exp(-Δt^2/(2σ^2)), where Δt is in years.
                    sigma_base = float(getattr(self.config, "time_sigma_years", 0.5))
                    sigma_start = getattr(self.config, "time_sigma_years_start", None)
                    sigma_end = getattr(self.config, "time_sigma_years_end", None)
                    sigma_decay = int(getattr(self.config, "time_sigma_decay_epochs", 0) or 0)
                    if sigma_start is None and sigma_end is None:
                        sigma_years = sigma_base
                    else:
                        sigma_start = float(sigma_start) if sigma_start is not None else sigma_base
                        sigma_end = float(sigma_end) if sigma_end is not None else sigma_start
                        if sigma_decay > 0:
                            frac = min((epoch_idx + 1) / float(sigma_decay), 1.0)
                            sigma_years = sigma_start + (sigma_end - sigma_start) * frac
                        else:
                            sigma_years = sigma_start
                    sigma_years = max(1e-6, float(sigma_years))
                    delta_years = (neg_times - t_scalar_v.unsqueeze(1)).abs() / _SECONDS_PER_YEAR

                    # Score the same triple at negative times.
                    h_rep = h_v.unsqueeze(1).expand(-1, j).reshape(-1)
                    r_rep = r_v.unsqueeze(1).expand(-1, j).reshape(-1)
                    t_rep = t_v.unsqueeze(1).expand(-1, j).reshape(-1)
                    neg_time_flat = neg_times.reshape(-1)
                    # Score with pretrained alpha_r (detached, no gradients to gate).
                    neg_time_score = self._score_temporal(
                        h_rep, r_rep, t_rep, neg_time_flat, device=device
                    ).view(t_v.size(0), j)

                    # Use the positive score for the same subset of samples.
                    pos_score_v = self._score_temporal(
                        h_v, r_v, t_v, t_scalar_v, device=device
                    )
                    time_loss_type = str(getattr(self.config, "time_loss_type", "pairwise")).lower()
                    if time_loss_type == "listwise":
                        # Listwise distribution matching: closer times should score higher.
                        # Target distribution uses a Gaussian kernel centered at the true time.
                        target_logits = torch.cat(
                            [
                                torch.zeros((t_v.size(0), 1), device=device),
                                -(delta_years**2) / (2.0 * (sigma_years**2)),
                            ],
                            dim=1,
                        )
                        target_probs = F.softmax(target_logits, dim=1)
                        score_mat = torch.cat([pos_score_v.unsqueeze(1), neg_time_score], dim=1)
                        log_probs = F.log_softmax(score_mat, dim=1)
                        per_sample_time_loss = -(target_probs * log_probs).sum(dim=1)
                        if self.verbose >= 2 and ((epoch_idx + 1) % 5 == 0):
                            idx = 0
                            pos_time_dbg = float(t_scalar_v[idx].item())
                            neg_times_dbg = [float(v) for v in neg_times[idx].detach().cpu().tolist()]
                            pos_time_iso = datetime.datetime.utcfromtimestamp(
                                int(pos_time_dbg)
                            ).strftime("%Y-%m-%d")
                            neg_times_iso = [
                                datetime.datetime.utcfromtimestamp(int(v)).strftime("%Y-%m-%d")
                                for v in neg_times_dbg
                            ]
                            pos_score_dbg = float(pos_score_v[idx].item())
                            neg_scores_dbg = [float(v) for v in neg_time_score[idx].detach().cpu().tolist()]
                            target_dbg = [float(v) for v in target_probs[idx].detach().cpu().tolist()]
                            far_days_dbg = int(getattr(self.config, "time_neg_far_days", 0))
                            print(
                                "[TKGE][time_dbg] "
                                f"epoch={epoch_idx+1}/{epochs} batch={epoch_batches+1}/{batches_per_epoch} "
                                f"pos_time={pos_time_dbg:.3f} ({pos_time_iso}) pos_score={pos_score_dbg:.4f} "
                                f"neg_times={neg_times_dbg} ({neg_times_iso}) neg_scores={neg_scores_dbg} "
                                f"target_probs={target_dbg} far_days={far_days_dbg} "
                                f"min_days={min_days:.1f} sigma_years={sigma_years:.3f}"
                            )
                    else:
                        # Pairwise log-sigmoid, weighted by temporal distance.
                        # Encourage pos_score_v - neg_time_score > time_margin.
                        time_margin = float(getattr(self.config, "time_contrastive_margin", 0.5))
                        w = 1.0 - torch.exp(-(delta_years**2) / (2.0 * (sigma_years**2)))
                        diff = (pos_score_v.unsqueeze(1) - neg_time_score) - time_margin
                        log_sig = -F.logsigmoid(diff)
                        per_sample_time_loss = (log_sig * w).mean(dim=1)  # [Bv]
                        if self.verbose >= 2 and ((epoch_idx + 1) % 5 == 0):
                            idx = 0
                            pos_time_dbg = float(t_scalar_v[idx].item())
                            neg_times_dbg = [float(v) for v in neg_times[idx].detach().cpu().tolist()]
                            pos_time_iso = datetime.datetime.utcfromtimestamp(
                                int(pos_time_dbg)
                            ).strftime("%Y-%m-%d")
                            neg_times_iso = [
                                datetime.datetime.utcfromtimestamp(int(v)).strftime("%Y-%m-%d")
                                for v in neg_times_dbg
                            ]
                            pos_score_dbg = float(pos_score_v[idx].item())
                            neg_scores_dbg = [float(v) for v in neg_time_score[idx].detach().cpu().tolist()]
                            w_dbg = [float(v) for v in w[idx].detach().cpu().tolist()]
                            far_days_dbg = int(getattr(self.config, "time_neg_far_days", 0))
                            print(
                                "[TKGE][time_dbg] "
                                f"epoch={epoch_idx+1}/{epochs} batch={epoch_batches+1}/{batches_per_epoch} "
                                f"pos_time={pos_time_dbg:.3f} ({pos_time_iso}) pos_score={pos_score_dbg:.4f} "
                                f"neg_times={neg_times_dbg} ({neg_times_iso}) neg_scores={neg_scores_dbg} "
                                f"weights={w_dbg} far_days={far_days_dbg} "
                                f"min_days={min_days:.1f} sigma_years={sigma_years:.3f}"
                            )

                    time_loss = per_sample_time_loss.mean()

                loss = (
                    triple_loss
                    + reg_loss
                    + float(getattr(self.config, "time_contrastive_weight", 0.0)) * time_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), max_norm=1.0)
                self.optimizer.step()
                loss_val = float(loss.item())
                losses.append(loss_val)
                epoch_loss_sum += loss_val
                epoch_triple_sum += float(triple_loss.item())
                epoch_time_sum += float(time_loss.item())
                epoch_reg_sum += float(reg_loss.item())
                epoch_batches += 1
            epoch_loss = epoch_loss_sum / max(epoch_batches, 1)
            # Compute checkpoint_start_epoch early (also used for best-loss gating below)
            checkpoint_start_epoch = int(getattr(self.config, "checkpoint_start_epoch", 0) or 0)
            if checkpoint_start_epoch <= 0:
                checkpoint_start_epoch = max(
                    1,
                    int(getattr(self.config, "time_neg_min_days_decay_epochs", 0) or 0),
                    int(getattr(self.config, "time_sigma_decay_epochs", 0) or 0),
                    int(getattr(self.config, "temporal_warmup_epochs", 0) or 0) + 1,
                )
            checkpoint_start_epoch = min(checkpoint_start_epoch, epochs)
            # Track window-best for periodic logging (only after checkpoint_start_epoch)
            _interval = max(self.verbose_epoch_interval, 1)
            if (epoch_idx + 1) >= checkpoint_start_epoch:
                if not hasattr(self, '_window_best_loss') or (epoch_idx + 1) % _interval == 1 or epoch_idx + 1 == checkpoint_start_epoch:
                    self._window_best_loss = epoch_loss
                    self._window_best_epoch = epoch_idx + 1
                elif epoch_loss < self._window_best_loss:
                    self._window_best_loss = epoch_loss
                    self._window_best_epoch = epoch_idx + 1
            if self.verbose and self.verbose < 3 and ((epoch_idx + 1) % _interval == 0 or epoch_idx == 0 or epoch_idx + 1 == epochs):
                parts = [f"[TKGE] epoch={epoch_idx+1}/{epochs}", f"loss={epoch_loss:.4f}"]
                nb = max(epoch_batches, 1)
                avg_triple = epoch_triple_sum / nb
                avg_time = epoch_time_sum / nb
                avg_reg = epoch_reg_sum / nb
                parts.append(f"triple={avg_triple:.4f}")
                if avg_time > 0:
                    parts.append(f"time={avg_time:.4f}")
                if avg_reg > 0:
                    parts.append(f"reg={avg_reg:.4f}")
                if hasattr(self, '_window_best_loss'):
                    parts.append(f"best{_interval}={self._window_best_loss:.4f}@{self._window_best_epoch}")
                else:
                    parts.append(f"best{_interval}=N/A (warmup)")
                print(" ".join(parts))
            if (
                self.verbose >= 2
                and self.config.temporal_mode == "romem"
                and hasattr(self.model, "speed_mlp")
                and (epoch_idx + 1) % 5 == 0
            ):
                try:
                    with torch.no_grad():
                        self._ensure_relation_text_embeddings()
                        if self.relation_text_embeddings is not None:
                            rel_text = self.relation_text_embeddings.to(device)
                            alpha = self.model.time_gate_alpha(rel_text).squeeze(-1).detach().cpu().tolist()
                            scale = float(self.model.log_time_scale.exp().detach().cpu())
                            rel_stats = []
                            for i, a in enumerate(alpha):
                                rel = self.kg.id2relation.get(i, str(i))
                                eff_scale = scale * max(float(a), 1e-12)
                                period_seconds = (2.0 * math.pi) / eff_scale
                                period_days = period_seconds / 86400.0
                                period_years = period_days / 365.25
                                rel_stats.append(
                                    {
                                        "rel": rel,
                                        "alpha": float(a),
                                        "period_days": period_days,
                                        "period_years": period_years,
                                    }
                                )
                            rel_stats_sorted = sorted(rel_stats, key=lambda x: x["alpha"])
                            slow = rel_stats_sorted[:5]
                            fast = rel_stats_sorted[-5:][::-1]
                            print(
                                f"[TKGE][gate_dbg] epoch={epoch_idx+1}/{epochs} "
                                f"slowest={slow} fastest={fast}"
                            )
                except Exception:
                    pass
            if self.config.checkpoint_strategy == "best_train_loss":
                if (epoch_idx + 1) >= checkpoint_start_epoch and (
                    best_loss is None or epoch_loss < best_loss
                ):
                    best_loss = epoch_loss
                    best_epoch = epoch_idx + 1
                    # snapshot trainable params only
                    if self.config.use_lora:
                        temporal_state = None
                        if self.config.temporal_mode == "romem":
                            temporal_state = {
                                "speed_mlp": {k: v.detach().clone() for k, v in self.model.speed_mlp.state_dict().items()}
                                if hasattr(self.model, "speed_mlp")
                                else None,
                                "log_time_scale": self.model.log_time_scale.detach().clone()
                                if hasattr(self.model, "log_time_scale")
                                else None,
                                "log_inv_freq_base": self.model.log_inv_freq_base.detach().clone()
                                if hasattr(self.model, "log_inv_freq_base")
                                else None,
                            }
                        best_state = {
                            "lora_ent": {k: v.detach().clone() for k, v in self.lora_ent.state_dict().items()},
                            "lora_rel": {k: v.detach().clone() for k, v in self.lora_rel.state_dict().items()},
                            "temporal": temporal_state,
                        }
                    else:
                        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        if self.verbose:
            print(
                f"[TKGE] train_end call={self.train_calls} "
                f"loss_mean={sum(losses)/len(losses):.4f} loss_last={losses[-1]:.4f} "
                f"ckpt={self.config.checkpoint_strategy} best_epoch={best_epoch} best_loss={best_loss} "
                f"ckpt_start={checkpoint_start_epoch}"
            )
        if self.config.checkpoint_strategy == "best_train_loss" and best_state is not None:
            if self.config.use_lora:
                self.lora_ent.load_state_dict(best_state["lora_ent"])
                self.lora_rel.load_state_dict(best_state["lora_rel"])
                temporal_state = best_state.get("temporal")
                if temporal_state:
                    speed = temporal_state.get("speed_mlp")
                    if speed is not None and hasattr(self.model, "speed_mlp"):
                        self.model.speed_mlp.load_state_dict(speed)
                    if temporal_state.get("log_time_scale") is not None and hasattr(self.model, "log_time_scale"):
                        self.model.log_time_scale.data.copy_(temporal_state["log_time_scale"])
                    if temporal_state.get("log_inv_freq_base") is not None and hasattr(self.model, "log_inv_freq_base"):
                        self.model.log_inv_freq_base.data.copy_(temporal_state["log_inv_freq_base"])
            else:
                self.model.load_state_dict(best_state)

        if self.verbose and hasattr(self.model, "log_time_scale"):
            try:
                scale = float(self.model.log_time_scale.exp().detach().cpu())
                base = (
                    float(self.model.log_inv_freq_base.exp().detach().cpu())
                    if hasattr(self.model, "log_inv_freq_base")
                    else None
                )
                period_seconds = (2.0 * math.pi) / max(scale, 1e-12)
                period_days = period_seconds / 86400.0
                period_years = period_days / 365.25
                msg = (
                    f"[TKGE] learned_time_scale={scale:.6e} rad/sec "
                    f"(period={period_days:.3f} days, {period_years:.3f} years)"
                )
                if base is not None:
                    msg += f" inv_freq_base={base:.3f}"
                print(msg)
                # Alpha distribution from pretrained gate
                if hasattr(self.model, "speed_mlp"):
                    self._ensure_relation_text_embeddings()
                    if self.relation_text_embeddings is not None:
                        rel_text = self.relation_text_embeddings.to(
                            self.model.log_time_scale.device
                        )
                        alpha = self.model.time_gate_alpha(rel_text).squeeze(-1).detach().cpu()
                        print(
                            f"[TKGE] alpha_r stats: "
                            f"mean={alpha.mean():.4f} std={alpha.std():.4f} "
                            f"min={alpha.min():.4f} max={alpha.max():.4f} "
                            f"n_static(α<0.1)={int((alpha < 0.1).sum())} "
                            f"n_temporal(α>0.5)={int((alpha > 0.5).sum())} "
                            f"n_relations={alpha.numel()}"
                        )
            except Exception:
                pass

        # ── Temporal probe: score sampled triples at various timestamps ──
        if (
            self.verbose
            and self.config.temporal_mode == "romem"
            and self.model is not None
            and "triples_tensor" in locals()
            and "times_tensor" in locals()
        ):
            try:
                self._temporal_probe(triples_tensor, times_tensor, happen_mask_tensor, device)
            except Exception:
                pass

        # Move model back to CPU after training so retrieval/scoring works
        if device.type != "cpu":
            self.model.to(torch.device("cpu"))
            self.optimizer = None  # optimizer references stale CUDA params
