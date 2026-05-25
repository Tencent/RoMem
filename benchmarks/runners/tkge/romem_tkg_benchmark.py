#!/usr/bin/env python3
"""
Standalone TKG benchmark for RoMem's KGE models on ICEWS05-15.

Tests the temporal knowledge graph embedding component in isolation
(no semantic retrieval, no PageRank, no OpenIE) so we can directly
compare link prediction quality against external TKG baselines.

All models use baseline-matched training (self-adversarial negative
sampling, L3 regularization, no normalization) so the ONLY difference
between distmult and romem is the temporal rotation mechanism.

Temporal losses (matching RoMem framework):
  - Time contrastive: listwise KL with Gaussian soft targets (alpha=1)
  - Gate competition: trainable gate for competing slots (gate stage)

Ablation via CLI flags:
  --time-contrastive-weight 0  disables time contrastive loss
  --gate-stage gate             enables trainable gate MLP

Usage:
    python romem_tkg_benchmark.py --model romem --epochs 500
    python romem_tkg_benchmark.py --model romem --time-contrastive-weight 0
    python romem_tkg_benchmark.py --model distmult --epochs 500
    python romem_tkg_benchmark.py --model romem_nogate --epochs 500
"""

import argparse
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import production models directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys as _sys
_sys.path.insert(0, str(_PROJECT_ROOT))

from romem.kge.model.distmult import DistMultModel
from romem.kge.model.romem_distmult import RoMemDistMultModel, _rotate
from romem.kge.model.romem_chronor import RoMemChronoRModel

_SECONDS_PER_YEAR = 365.25 * 86400.0


# ---------------------------------------------------------------------------
# Scoring helper with alpha routing
# ---------------------------------------------------------------------------

def score_temporal(model, h, r, t, ts, alpha_mode, alpha_r_fixed=None,
                   rel_text_embs=None):
    """Score temporal triples with different alpha routing modes.

    alpha_mode:
      "fixed"  - use precomputed alpha_r_fixed[r] (frozen gate)
      "detach" - compute alpha from speed_mlp but detach (no gate gradients)
      "full"   - compute alpha from speed_mlp with gradients
      "one"    - force alpha=1 (pure rotation, no gate)
    """
    if alpha_mode == "one":
        alpha = torch.ones((h.size(0), 1), device=h.device)
    elif alpha_mode == "fixed":
        alpha = alpha_r_fixed[r]
    elif alpha_mode == "detach":
        alpha = model.time_gate_alpha(rel_text_embs[r]).detach()
    elif alpha_mode == "full":
        alpha = model.time_gate_alpha(rel_text_embs[r])
    else:
        raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
    return model.score_triples(h, r, t, ts, alpha_override=alpha)


# ---------------------------------------------------------------------------
# Pretrained gate loading
# ---------------------------------------------------------------------------

def _load_env_file(path=".env"):
    """Load key=value pairs from .env file into os.environ (if not already set)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def load_pretrained_gate(model, gate_checkpoint_path, relation2id,
                         rel_emb_cache_path, freeze=True):
    """Load pretrained speed_mlp weights, compute alpha_r per relation.

    Args:
        model: RoMemDistMultModel or RoMemChronoRModel instance
        gate_checkpoint_path: Path to alpha_r_pretrained.pt
        relation2id: dict mapping relation name -> int id
        rel_emb_cache_path: Path to cached OpenAI relation embeddings
        freeze: If True, freeze gate parameters after loading

    Returns:
        alpha_r: [num_relations, 1] tensor of gate values
        rel_text_embs: [num_relations, dim] aligned relation text embeddings
    """
    # Load gate checkpoint
    obj = torch.load(gate_checkpoint_path, weights_only=False)
    state = obj["speed_mlp"]
    ckpt_dim = int(obj.get("relation_text_dim", 0))
    print(f"  Gate checkpoint: rel_text_dim={ckpt_dim}")

    # Rebuild speed_mlp with correct dimensions and load weights
    model.rel_text_dim = ckpt_dim
    hidden = max(32, min(128, ckpt_dim // 2))
    model.speed_mlp = nn.Sequential(
        nn.Linear(ckpt_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    model.speed_mlp.load_state_dict(state, strict=True)

    # Load cached relation text embeddings
    if not os.path.exists(rel_emb_cache_path):
        raise FileNotFoundError(
            f"Cached relation embeddings not found: {rel_emb_cache_path}\n"
            "Run: python cache_relation_embeddings.py --dataset-dir <path> first."
        )
    cache = torch.load(rel_emb_cache_path, weights_only=False)
    cached_rel2id = cache["relation2id"]
    cached_embs = cache["embeddings"]  # [cached_num_rel, dim]
    print(f"  Cached embeddings: {cached_embs.shape} from model={cache.get('model', '?')}")

    # Align cached embeddings to our relation2id ordering
    num_rel = len(relation2id)
    rel_text_embs = torch.zeros(num_rel, cached_embs.shape[1])
    matched = 0
    for rel_name, rid in relation2id.items():
        if rel_name in cached_rel2id:
            rel_text_embs[rid] = cached_embs[cached_rel2id[rel_name]]
            matched += 1
    print(f"  Matched {matched}/{num_rel} relations from cache")
    if matched < num_rel:
        print(f"  WARNING: {num_rel - matched} relations have zero embeddings!")

    # Verify dimension match
    if rel_text_embs.shape[1] != ckpt_dim:
        raise ValueError(
            f"Embedding dim mismatch: cached={rel_text_embs.shape[1]}, "
            f"gate checkpoint expects={ckpt_dim}. "
            f"Re-run cache_relation_embeddings.py with --dimensions {ckpt_dim}"
        )

    # Compute alpha_r for all relations
    model.eval()
    with torch.no_grad():
        alpha_r = model.time_gate_alpha(rel_text_embs)  # [num_rel, 1]

    print(f"  alpha_r: min={alpha_r.min():.4f}, max={alpha_r.max():.4f}, "
          f"mean={alpha_r.mean():.4f}")

    # Optionally freeze the gate MLP
    if freeze:
        for param in model.speed_mlp.parameters():
            param.requires_grad = False
        print("  Gate: FROZEN (pretrained stage)")
    else:
        print("  Gate: TRAINABLE (gate stage)")

    return alpha_r, rel_text_embs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_quads(path):
    """Load tab-separated quads: head  relation  tail  YYYY-MM-DD"""
    quads = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                quads.append((parts[0], parts[1], parts[2], parts[3]))
    return quads


def build_vocab(train, valid, test):
    """Build entity2id and relation2id from all splits (train first for stable IDs)."""
    entity2id = {}
    relation2id = {}
    for quads in [train, valid, test]:
        for h, r, t, _ in quads:
            if h not in entity2id:
                entity2id[h] = len(entity2id)
            if t not in entity2id:
                entity2id[t] = len(entity2id)
            if r not in relation2id:
                relation2id[r] = len(relation2id)
    return entity2id, relation2id


def date_to_timestamp(date_str):
    """Convert YYYY-MM-DD to unix timestamp (seconds)."""
    return datetime.fromisoformat(date_str).timestamp()


def quads_to_tensors(quads, entity2id, relation2id):
    """Convert string quads to tuple of tensors (h, r, t, timestamp)."""
    h_ids, r_ids, t_ids, timestamps = [], [], [], []
    for h, r, t, date in quads:
        h_ids.append(entity2id[h])
        r_ids.append(relation2id[r])
        t_ids.append(entity2id[t])
        timestamps.append(date_to_timestamp(date))
    return (
        torch.tensor(h_ids, dtype=torch.long),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(t_ids, dtype=torch.long),
        torch.tensor(timestamps, dtype=torch.float64),
    )


def build_filter_dicts(*splits):
    """Build filtering dicts from ALL splits for filtered evaluation.

    Returns:
        hr2t: {(h_id, r_id): set(t_ids)}
        rt2h: {(r_id, t_id): set(h_ids)}
    """
    hr2t = defaultdict(set)
    rt2h = defaultdict(set)
    for split in splits:
        h_all, r_all, t_all, _ts = split
        for i in range(len(h_all)):
            hi, ri, ti = h_all[i].item(), r_all[i].item(), t_all[i].item()
            hr2t[(hi, ri)].add(ti)
            rt2h[(ri, ti)].add(hi)
    return dict(hr2t), dict(rt2h)


def build_temporal_indices(h_all, r_all, t_all, ts_all):
    """Precompute temporal data structures from training data.

    Returns:
        hr_to_times: {(h, r): [timestamps]} for time negative sampling
        is_competing: [n] bool tensor per training triple
        global_times: unique training timestamps tensor
    """
    n = len(h_all)
    hr_to_tails_raw = defaultdict(set)
    hr_to_times_raw = defaultdict(set)

    for i in range(n):
        hi = h_all[i].item()
        ri = r_all[i].item()
        ti = t_all[i].item()
        tsi = ts_all[i].item()
        hr_to_tails_raw[(hi, ri)].add(ti)
        hr_to_times_raw[(hi, ri)].add(tsi)

    competing_hr_keys = {k for k, v in hr_to_tails_raw.items() if len(v) > 1}
    hr_to_times = {k: list(v) for k, v in hr_to_times_raw.items()}

    is_competing = torch.tensor([
        (h_all[i].item(), r_all[i].item()) in competing_hr_keys
        for i in range(n)
    ], dtype=torch.bool)

    global_times = ts_all.unique()

    n_competing = int(is_competing.sum().item())
    print(f"  Temporal indices: {len(competing_hr_keys)} competing (h,r) keys, "
          f"{n_competing}/{n} competing triples, "
          f"{global_times.numel()} unique timestamps")

    return hr_to_times, is_competing, global_times


# ---------------------------------------------------------------------------
# Training (baseline-matched triple loss + RoMem temporal losses)
# ---------------------------------------------------------------------------

def train_epoch(model, data, args, optimizer, alpha_r_fixed, rel_text_embs,
                temporal_indices, epoch, device, is_temporal, gate_stage,
                chronor_loss=False):
    """Train one epoch with triple loss + optional RoMem temporal losses.

    When chronor_loss=False (default, DistMult-matched):
      Triple loss: self-adversarial negative sampling + L3 global reg
    When chronor_loss=True (ChronoR-matched):
      Triple loss: 1-vs-all cross-entropy (head + tail) + N3 per-batch reg

    Shared temporal losses (controlled by --time-*-weight flags):
      Time contrastive: listwise KL with Gaussian soft targets (alpha=1)
      Gate competition: trainable gate for competing slots (gate stage only)
    """
    model.train()
    h_all, r_all, t_all, ts_all = [x.to(device) for x in data]
    n = len(h_all)
    num_ent = model.ent_emb.num_embeddings
    num_neg = args.num_negatives
    adv_temp = args.adversarial_temperature
    reg_weight = args.regularization
    tc_weight = args.time_contrastive_weight if is_temporal else 0.0
    sigma_years = max(1e-6, args.time_sigma_years)

    # Unpack temporal indices
    if temporal_indices is not None:
        hr_to_times, is_competing, global_times = temporal_indices
        is_competing = is_competing.to(device)
        global_times = global_times.to(device)
    else:
        hr_to_times = None
        is_competing = None
        global_times = None

    # Alpha mode for base triple loss
    if is_temporal:
        base_alpha_mode = "detach" if gate_stage == "gate" else "fixed"

    indices = torch.randperm(n, device=device)
    total_loss = 0.0
    total_tc_loss = 0.0
    total_gate_comp = 0.0
    num_batches = 0

    for start in range(0, n, args.batch_size):
        batch_idx = indices[start:start + args.batch_size]
        h = h_all[batch_idx]
        r = r_all[batch_idx]
        t = t_all[batch_idx]
        ts = ts_all[batch_idx]
        bsz = len(h)

        if chronor_loss:
            # ── 1-vs-all cross-entropy (ChronoR-consistent) ─────────
            # Efficient trick: score(t_j) = _rotate(query, -θ) · t_j
            # allows scoring ALL entities with a single matmul against
            # the unrotated entity table.
            ce_fn = nn.CrossEntropyLoss(reduction='mean')

            # Alpha for rotation
            if base_alpha_mode == "one":
                alpha = torch.ones((bsz, 1), device=device)
            elif base_alpha_mode == "fixed":
                alpha = alpha_r_fixed[r]
            elif base_alpha_mode == "detach":
                alpha = model.time_gate_alpha(rel_text_embs[r]).detach()
            else:
                alpha = alpha_r_fixed[r]

            theta_s = model._theta(ts) * alpha  # [B, k*d]

            h_emb = model.ent_emb.weight[h]
            r_emb = model.rel_emb.weight[r]
            r2_emb = model.rel_inv_emb.weight[r]
            t_emb = model.ent_emb.weight[t]

            h_rot = _rotate(h_emb, theta_s)
            t_rot = _rotate(t_emb, theta_s)

            all_ent = model.ent_emb.weight  # [N, dim]

            # Tail prediction: unrotate query to score against raw entities
            qt = _rotate(h_rot * r_emb * r2_emb, -theta_s)
            scores_t = qt @ all_ent.T  # [B, N]
            loss_tail = ce_fn(scores_t, t)

            # Head prediction: symmetric unrotation trick
            qh = _rotate(r_emb * r2_emb * t_rot, -theta_s)
            scores_h = qh @ all_ent.T  # [B, N]
            loss_head = ce_fn(scores_h, h)

            triple_loss = (loss_tail + loss_head) / 2

            # N3 (L4) per-batch regularization (matching ChronoR)
            reg_loss = torch.tensor(0.0, device=device)
            if reg_weight > 0:
                reg_loss = reg_weight * (
                    (h_emb.abs() ** 4).sum() +
                    (r_emb.abs() ** 4).sum() +
                    (r2_emb.abs() ** 4).sum() +
                    (t_emb.abs() ** 4).sum()
                ) / bsz

        else:
            # ── Self-adversarial negative sampling (DistMult-matched) ──
            # ---- Positive scores ----
            if is_temporal:
                pos_score = score_temporal(model, h, r, t, ts, base_alpha_mode,
                                          alpha_r_fixed, rel_text_embs)
            else:
                pos_score = model.score_triples(h, r, t)

            # ---- Negative sampling (alternating head/tail) ----
            neg_ent = torch.randint(0, num_ent, (bsz, num_neg), device=device)

            if num_batches % 2 == 0:
                # Tail corruption
                neg_h = h.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_r = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_t_flat = neg_ent.reshape(-1)
                if is_temporal:
                    neg_ts = ts.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                    neg_score = score_temporal(
                        model, neg_h, neg_r, neg_t_flat, neg_ts,
                        base_alpha_mode, alpha_r_fixed, rel_text_embs
                    ).view(bsz, num_neg)
                else:
                    neg_score = model.score_triples(
                        neg_h, neg_r, neg_t_flat).view(bsz, num_neg)
            else:
                # Head corruption
                neg_h_flat = neg_ent.reshape(-1)
                neg_r = r.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_t = t.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                if is_temporal:
                    neg_ts = ts.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                    neg_score = score_temporal(
                        model, neg_h_flat, neg_r, neg_t, neg_ts,
                        base_alpha_mode, alpha_r_fixed, rel_text_embs
                    ).view(bsz, num_neg)
                else:
                    neg_score = model.score_triples(
                        neg_h_flat, neg_r, neg_t).view(bsz, num_neg)

            # ---- Self-adversarial triple loss ----
            neg_weights = F.softmax(neg_score * adv_temp, dim=1).detach()
            neg_loss = -(neg_weights * F.logsigmoid(-neg_score)).sum(dim=1).mean()
            pos_loss = -F.logsigmoid(pos_score).mean()
            triple_loss = (pos_loss + neg_loss) / 2

            # ---- L3 global regularization ----
            reg_loss = torch.tensor(0.0, device=device)
            if reg_weight > 0:
                reg_loss = reg_weight * (
                    model.ent_emb.weight.norm(p=3) ** 3 +
                    model.rel_emb.weight.norm(p=3) ** 3
                )
                if hasattr(model, 'rel_inv_emb'):
                    reg_loss = reg_loss + reg_weight * (
                        model.rel_inv_emb.weight.norm(p=3) ** 3
                    )

        # ---- Gate competition loss (gate stage only, not for nogate) ----
        gate_comp_loss = torch.tensor(0.0, device=device)
        gate_reg = torch.tensor(0.0, device=device)
        if (is_temporal and gate_stage == "gate" and
                is_competing is not None and
                not model.force_time_gate_one):
            batch_comp_mask = is_competing[batch_idx]
            if batch_comp_mask.any().item():
                h_c = h[batch_comp_mask]
                r_c = r[batch_comp_mask]
                t_c = t[batch_comp_mask]
                ts_c = ts[batch_comp_mask]
                n_c = h_c.size(0)
                # Full alpha mode: gradients flow through gate
                pos_comp = score_temporal(model, h_c, r_c, t_c, ts_c,
                                         "full", alpha_r_fixed, rel_text_embs)
                # Negative sampling for competing slots
                neg_ent_c = torch.randint(0, num_ent, (n_c, num_neg), device=device)
                neg_h_c = h_c.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_r_c = r_c.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_t_c = neg_ent_c.reshape(-1)
                neg_ts_c = ts_c.unsqueeze(1).expand(-1, num_neg).reshape(-1)
                neg_comp = score_temporal(
                    model, neg_h_c, neg_r_c, neg_t_c, neg_ts_c,
                    "full", alpha_r_fixed, rel_text_embs
                ).view(n_c, num_neg)
                # Self-adversarial loss with full alpha
                neg_w_c = F.softmax(neg_comp * adv_temp, dim=1).detach()
                gate_comp_loss = 0.5 * (
                    -F.logsigmoid(pos_comp).mean() +
                    -(neg_w_c * F.logsigmoid(-neg_comp)).sum(1).mean()
                )

            # Gate regularization: push non-competing alpha toward 0
            non_comp_mask = ~is_competing[batch_idx]
            gate_reg_w = args.gate_reg_weight
            if non_comp_mask.any().item() and gate_reg_w > 0:
                r_nc = r[non_comp_mask]
                alpha_nc = model.time_gate_alpha(rel_text_embs[r_nc])
                gate_reg = gate_reg_w * alpha_nc.mean()

        # ---- Time contrastive loss (alpha=1, trains rotation dynamics) ----
        tc_loss = torch.tensor(0.0, device=device)
        if (is_temporal and tc_weight > 0 and
                global_times is not None and global_times.numel() > 0):
            j = args.num_time_negatives

            # Sample negative times from global pool
            pool_idx = torch.randint(
                0, global_times.size(0), (bsz, j), device=device)
            neg_times = global_times[pool_idx]

            # Delta in years for Gaussian soft targets
            delta_years = (neg_times - ts.unsqueeze(1)).abs() / _SECONDS_PER_YEAR

            # Score same triple at negative times (alpha=1, no gate gradient)
            h_rep = h.unsqueeze(1).expand(-1, j).reshape(-1)
            r_rep = r.unsqueeze(1).expand(-1, j).reshape(-1)
            t_rep = t.unsqueeze(1).expand(-1, j).reshape(-1)
            neg_time_flat = neg_times.reshape(-1)

            neg_time_score = score_temporal(
                model, h_rep, r_rep, t_rep, neg_time_flat,
                "one", alpha_r_fixed, rel_text_embs
            ).view(bsz, j)

            pos_time_score = score_temporal(
                model, h, r, t, ts,
                "one", alpha_r_fixed, rel_text_embs
            )

            # Listwise distribution matching (KL divergence)
            # Target: Gaussian kernel centered at true time
            target_logits = torch.cat([
                torch.zeros((bsz, 1), device=device),
                -(delta_years ** 2) / (2.0 * sigma_years ** 2),
            ], dim=1)
            target_probs = F.softmax(target_logits, dim=1)
            score_mat = torch.cat(
                [pos_time_score.unsqueeze(1), neg_time_score], dim=1)
            log_probs = F.log_softmax(score_mat, dim=1)
            tc_loss = -(target_probs * log_probs).sum(dim=1).mean()

        # ---- Total loss ----
        loss = (triple_loss + reg_loss
                + args.gate_comp_weight * gate_comp_loss + gate_reg
                + tc_weight * tc_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_tc_loss += float(tc_loss.item())
        total_gate_comp += float(gate_comp_loss.item())
        num_batches += 1

    nb = max(num_batches, 1)
    return {
        "loss": total_loss / nb,
        "tc_loss": total_tc_loss / nb,
        "gate_comp": total_gate_comp / nb,
    }


# ---------------------------------------------------------------------------
# Evaluation (filtered link prediction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, hr2t, rt2h, alpha_r, device, desc="Eval"):
    """Standard filtered link prediction: MRR, Hits@1/3/10."""
    model.eval()
    h_all, r_all, t_all, ts_all = [x.to(device) for x in data]
    n = len(h_all)
    num_ent = model.ent_emb.num_embeddings
    is_temporal = isinstance(model, (RoMemDistMultModel, RoMemChronoRModel))
    has_inv_rel = hasattr(model, 'rel_inv_emb')

    # Pre-compute all entity embeddings
    all_ent = model.ent_emb.weight  # [N, dim] raw

    ranks = []
    t_start = time.time()

    for i in range(n):
        hi, ri, ti, tsi = h_all[i], r_all[i], t_all[i], ts_all[i]
        h_id, r_id, t_id = hi.item(), ri.item(), ti.item()

        if is_temporal:
            alpha = alpha_r[ri].unsqueeze(0)
            theta = model._theta(tsi.unsqueeze(0)) * alpha

            # Rotate all entities for this timestamp
            all_ent_rot = _rotate(all_ent, theta.expand(num_ent, -1))

            h_emb = model.ent_emb.weight[hi].unsqueeze(0)
            r_emb = model.rel_emb.weight[ri].unsqueeze(0)
            t_emb = model.ent_emb.weight[ti].unsqueeze(0)

            h_rot = _rotate(h_emb, theta)
            t_rot = _rotate(t_emb, theta)

            # Tail/Head scoring: flat dot product
            if has_inv_rel:
                r2_emb = model.rel_inv_emb.weight[ri].unsqueeze(0)
                query_tail = (h_rot * r_emb * r2_emb).squeeze(0)
                query_head = (r_emb * r2_emb * t_rot).squeeze(0)
            else:
                query_tail = (h_rot * r_emb).squeeze(0)
                query_head = (r_emb * t_rot).squeeze(0)
            scores_tail = all_ent_rot @ query_tail
            scores_head = all_ent_rot @ query_head
        else:
            # Static DistMult: score = sum(h * r * t) = (h * r) . t
            h_emb = model.ent_emb.weight[hi].unsqueeze(0)
            r_emb = model.rel_emb.weight[ri].unsqueeze(0)

            query_tail = (h_emb * r_emb).squeeze(0)
            scores_tail = all_ent @ query_tail

            t_emb = model.ent_emb.weight[ti].unsqueeze(0)
            query_head = (r_emb * t_emb).squeeze(0)
            scores_head = all_ent @ query_head

        # --- Filtered tail prediction ---
        filter_t = hr2t.get((h_id, r_id), set())
        if len(filter_t) > 1:
            filt_idx = [e for e in filter_t if e != t_id]
            if filt_idx:
                scores_tail[filt_idx] = float('-inf')
        rank_t = 1 + (scores_tail > scores_tail[t_id]).sum().item()

        # --- Filtered head prediction ---
        filter_h = rt2h.get((r_id, t_id), set())
        if len(filter_h) > 1:
            filt_idx = [e for e in filter_h if e != h_id]
            if filt_idx:
                scores_head[filt_idx] = float('-inf')
        rank_h = 1 + (scores_head > scores_head[h_id]).sum().item()

        ranks.extend([rank_t, rank_h])

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            cur_mrr = np.mean(1.0 / np.array(ranks, dtype=np.float64))
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc}: {i+1}/{n}  running MRR={cur_mrr:.4f}  "
                  f"ETA={eta:.0f}s", flush=True)

    ranks = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks))
    hits1 = float(np.mean(ranks <= 1))
    hits3 = float(np.mean(ranks <= 3))
    hits10 = float(np.mean(ranks <= 10))
    return mrr, hits1, hits3, hits10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RoMem TKG Benchmark on ICEWS05-15")
    p.add_argument("--dataset-dir", default="../dataset/icews05-15",
                    help="Path to dataset directory with train/valid/test.txt")
    p.add_argument("--model", choices=[
                        "distmult", "romem", "romem_nogate",
                        "chronor_romem", "chronor_romem_nogate",
                        "chronor_romem_notc"],
                    default="romem")
    p.add_argument("--gate-checkpoint",
                    default=str(_PROJECT_ROOT / "outputs" / "pretrain_gate_data" / "alpha_r_pretrained.pt"),
                    help="Path to pretrained speed_mlp checkpoint")
    p.add_argument("--gate-stage", choices=["pretrained", "gate"],
                    default="pretrained",
                    help="Gate mode: pretrained=frozen gate, gate=trainable gate MLP")

    # Model hyperparameters (matching KGE baseline defaults)
    p.add_argument("--embedding-dim", type=int, default=500,
                    help="Embedding dim (default 500 to match baseline)")
    p.add_argument("--gamma", type=float, default=200.0,
                    help="Gamma for init range: embedding_range = (gamma+2)/dim")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num-negatives", type=int, default=256,
                    help="Number of negative samples per positive (baseline uses 256)")
    p.add_argument("--adversarial-temperature", type=float, default=1.0,
                    help="Self-adversarial sampling temperature")
    p.add_argument("--regularization", type=float, default=1e-5,
                    help="L3 regularization weight")
    p.add_argument("--k", type=int, default=3,
                    help="Number of components for ChronoR backbone (default: 3)")
    p.add_argument("--validate-every", type=int, default=20)
    p.add_argument("--optimizer", choices=["adam", "adagrad"], default=None,
                    help="Optimizer (default: adagrad for chronor models, adam otherwise)")
    p.add_argument("--seed", type=int, default=42)

    # Time contrastive loss
    p.add_argument("--time-contrastive-weight", type=float, default=0.5,
                    help="Weight for time contrastive loss (0 to disable)")
    p.add_argument("--num-time-negatives", type=int, default=4,
                    help="Number of negative time samples per positive")
    p.add_argument("--time-sigma-years", type=float, default=0.5,
                    help="Gaussian sigma (years) for soft time targets")

    # Gate learning
    p.add_argument("--gate-comp-weight", type=float, default=1.0,
                    help="Weight for gate competition loss (gate stage only)")
    p.add_argument("--gate-reg-weight", type=float, default=0.01,
                    help="Regularization pushing non-competing alpha toward 0")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Dataset: {args.dataset_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    dataset_dir = Path(args.dataset_dir)
    train_quads = load_quads(dataset_dir / "train.txt")
    valid_quads = load_quads(dataset_dir / "valid.txt")
    test_quads = load_quads(dataset_dir / "test.txt")
    print(f"Loaded: train={len(train_quads)}, valid={len(valid_quads)}, "
          f"test={len(test_quads)}")

    entity2id, relation2id = build_vocab(train_quads, valid_quads, test_quads)
    num_ent = len(entity2id)
    num_rel = len(relation2id)
    print(f"Vocab:  {num_ent} entities, {num_rel} relations")

    train_data = quads_to_tensors(train_quads, entity2id, relation2id)
    valid_data = quads_to_tensors(valid_quads, entity2id, relation2id)
    test_data = quads_to_tensors(test_quads, entity2id, relation2id)

    hr2t, rt2h = build_filter_dicts(train_data, valid_data, test_data)

    # ── Create model ──────────────────────────────────────────────────────
    is_temporal = args.model in (
        "romem", "romem_nogate", "chronor_romem", "chronor_romem_nogate",
        "chronor_romem_notc")
    alpha_r_fixed = None
    rel_text_embs = None
    gate_stage = (args.gate_stage
                  if args.model in ("romem", "chronor_romem",
                                    "chronor_romem_notc")
                  else "pretrained")

    if args.model == "distmult":
        model = DistMultModel(
            num_ent, num_rel, args.embedding_dim, gamma=args.gamma)
    elif args.model in ("chronor_romem", "chronor_romem_nogate",
                        "chronor_romem_notc"):
        force_gate_one = args.model == "chronor_romem_nogate"
        # k components of 2d each; d = embedding_dim // (2*k)
        d = args.embedding_dim // (2 * args.k)
        print(f"ChronoR backbone: k={args.k}, d={d}, "
              f"total_dim={args.k * 2 * d}")
        model = RoMemChronoRModel(
            num_ent, num_rel, d, 1536,
            k=args.k,
            gamma=args.gamma,
            force_time_gate_one=force_gate_one,
        )
    else:
        # romem or romem_nogate: temporal rotation on top of baseline DistMult
        force_gate_one = (args.model == "romem_nogate")
        model = RoMemDistMultModel(
            num_ent, num_rel, args.embedding_dim // 2, 1536,
            gamma=args.gamma,
            force_time_gate_one=force_gate_one,
        )

    model = model.to(device)

    # ── Load pretrained gate (romem only) ─────────────────────────────────
    _SCRIPT_DIR = Path(__file__).resolve().parent
    rel_emb_cache = str(_SCRIPT_DIR / "data" / "icews_relation_embeddings.pt")

    if args.model in ("romem", "chronor_romem",
                       "chronor_romem_notc"):
        print(f"\nLoading pretrained gate from {args.gate_checkpoint}")
        freeze_gate = (gate_stage != "gate")
        alpha_r_fixed, rel_text_embs = load_pretrained_gate(
            model, args.gate_checkpoint, relation2id, rel_emb_cache,
            freeze=freeze_gate)
        alpha_r_fixed = alpha_r_fixed.to(device)
        rel_text_embs = rel_text_embs.to(device)
        model = model.to(device)  # re-send after speed_mlp rebuild
    elif args.model in ("romem_nogate", "chronor_romem_nogate"):
        alpha_r_fixed = torch.ones(num_rel, 1, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Print temporal loss config
    if is_temporal:
        print(f"\nTemporal losses:")
        print(f"  Time contrastive: weight={args.time_contrastive_weight}, "
              f"negatives={args.num_time_negatives}, "
              f"sigma={args.time_sigma_years}yr")
        print(f"  Gate stage: {gate_stage}")
        if gate_stage == "gate":
            print(f"  Gate competition: weight={args.gate_comp_weight}, "
                  f"reg={args.gate_reg_weight}")

    # ── Precompute temporal indices ───────────────────────────────────────
    temporal_indices = None
    if is_temporal:
        print("\nBuilding temporal indices...")
        temporal_indices = build_temporal_indices(*train_data)

    # ── Optimizer (only trainable params) ─────────────────────────────────
    # Override weights for ablation variants
    if args.model == "chronor_romem_notc":
        args.time_contrastive_weight = 0.0

    chronor_loss = args.model.startswith("chronor_")
    opt_name = args.optimizer or ("adagrad" if chronor_loss else "adam")
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr)
    print(f"Optimizer: {opt_name}, lr={args.lr}")
    if chronor_loss:
        print(f"Loss: 1-vs-all CE (ChronoR-consistent), N3 (L4) per-batch reg")

    # ── Training loop ─────────────────────────────────────────────────────
    best_mrr = 0.0
    best_state = None

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        metrics = train_epoch(
            model, train_data, args, optimizer,
            alpha_r_fixed, rel_text_embs,
            temporal_indices, epoch, device, is_temporal, gate_stage,
            chronor_loss=chronor_loss)
        elapsed = time.time() - t0

        if (epoch + 1) % 10 == 0 or epoch == 0:
            parts = [f"loss={metrics['loss']:.4f}"]
            if is_temporal and args.time_contrastive_weight > 0:
                parts.append(f"tc={metrics['tc_loss']:.4f}")
            if is_temporal and gate_stage == "gate":
                parts.append(f"gate={metrics['gate_comp']:.4f}")
            print(f"Epoch {epoch+1:4d}/{args.epochs}  "
                  f"{'  '.join(parts)}  time={elapsed:.1f}s")

        # Periodic validation + training accuracy check
        if (epoch + 1) % args.validate_every == 0:
            # For gate stage, recompute alpha from current model state
            if gate_stage == "gate" and rel_text_embs is not None:
                model.eval()
                with torch.no_grad():
                    alpha_r_eval = model.time_gate_alpha(rel_text_embs)
                    print(f"  [gate] alpha_r: min={alpha_r_eval.min():.4f}, "
                          f"max={alpha_r_eval.max():.4f}, "
                          f"mean={alpha_r_eval.mean():.4f}")
            else:
                alpha_r_eval = alpha_r_fixed

            # Evaluate on a random subset of training data (same size as valid)
            n_train = len(train_data[0])
            n_sample = min(len(valid_data[0]), n_train)
            idx = torch.randperm(n_train)[:n_sample]
            train_sample = tuple(x[idx] for x in train_data)
            print(f"\n--- Epoch {epoch+1}: Train sample ({n_sample}) ---")
            t_mrr, t_h1, t_h3, t_h10 = evaluate(
                model, train_sample, hr2t, rt2h,
                alpha_r_eval, device, desc="Train")
            print(f"Train  MRR={t_mrr:.4f}  Hits@1={t_h1:.4f}  "
                  f"Hits@3={t_h3:.4f}  Hits@10={t_h10:.4f}")

            print(f"--- Epoch {epoch+1}: Validation ---")
            mrr, h1, h3, h10 = evaluate(
                model, valid_data, hr2t, rt2h,
                alpha_r_eval, device, desc="Valid")
            print(f"Valid  MRR={mrr:.4f}  Hits@1={h1:.4f}  "
                  f"Hits@3={h3:.4f}  Hits@10={h10:.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                best_state = {
                    'model': {k: v.cpu().clone()
                              for k, v in model.state_dict().items()},
                    'epoch': epoch + 1,
                    'mrr': mrr,
                }
                print(f"  ** New best valid MRR: {mrr:.4f}")
            print()

    # ── Load best model ───────────────────────────────────────────────────
    if best_state is not None:
        print(f"Loading best model from epoch {best_state['epoch']} "
              f"(valid MRR={best_state['mrr']:.4f})")
        model.load_state_dict(best_state['model'])

    # ── Recompute alpha for test evaluation ───────────────────────────────
    if gate_stage == "gate" and rel_text_embs is not None:
        model.eval()
        with torch.no_grad():
            alpha_r_test = model.time_gate_alpha(rel_text_embs)
    else:
        alpha_r_test = alpha_r_fixed

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\n--- Test Evaluation ---")
    mrr, h1, h3, h10 = evaluate(
        model, test_data, hr2t, rt2h,
        alpha_r_test, device, desc="Test")

    label = {
        "distmult": "RoMem-DistMult",
        "romem": "RoMem",
        "romem_nogate": "RoMem-NoGate",
        "chronor_romem": "ChronoR-RoMem",
        "chronor_romem_nogate": "ChronoR-RoMem-NG",
        "chronor_romem_notc": "ChronoR-RoMem-NoTC",
    }[args.model]

    print(f"\n{label} Fil setting:")
    print(f"MRR = {mrr:.4f}")
    print(f"Hit@1 = {h1:.4f}")
    print(f"Hit@3 = {h3:.4f}")
    print(f"Hit@10 = {h10:.4f}")


if __name__ == "__main__":
    main()
