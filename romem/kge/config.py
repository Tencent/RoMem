"""
Configuration utilities for the TKGE adapter.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TKGEConfig:
    """
    Lightweight configuration placeholder for the TKGE integration.

    Extendable for future loss/optimizer choices.
    """

    snapshot_cache_dir: str | None = None
    embedding_dim: int = 64
    learning_rate: float = 1e-3
    triple_margin: float = 0.5
    batch_size: int = 128
    # Training epochs per update (each epoch iterates over all triples).
    steps_per_update: int = 10
    num_negatives: int = 5
    device: str = "cpu"
    use_lora: bool = False
    lora_rank: int = 8
    checkpoint_strategy: str = "best_train_loss"

    # Temporal extension (RoMem)
    temporal_mode: str = "none"  # "none" | "romem"
    temporal_backbone: str = "distmult"  # "distmult" | "chronor"
    chronor_k: int = 3  # Number of components for ChronoR backbone
    time_source: str = "happen_else_obs"

    # Time-contrastive training (temporal supervision)
    use_time_contrastive: bool = False
    time_contrastive_weight: float = 0.5
    time_contrastive_margin: float = 0.5
    # Time-contrastive loss type: pairwise margin vs listwise distribution matching.
    time_loss_type: Literal["pairwise", "listwise"] = "listwise"
    num_time_negatives: int = 4
    # Persistence-safe weighting; sigma in years for weighting function w(Δt).
    time_sigma_years: float = 0.5
    # Optional sigma curriculum.
    time_sigma_years_start: float | None = None
    time_sigma_years_end: float | None = None
    time_sigma_decay_epochs: int = 0
    # Prefer negatives drawn from the same (head, relation) group when available.
    time_neg_source: Literal["hr", "global"] = "hr"
    # Jitter (in years) applied to sampled negative times to diversify negatives.
    time_neg_jitter_years: float = 0.0
    # Force one far negative per example by adding +/- offset in days (0 disables).
    time_neg_far_days: int = 0
    # Optional minimum-gap curriculum for negative times.
    time_neg_min_days_start: float = 0.0
    time_neg_min_days_end: float = 0.0
    time_neg_min_days_decay_epochs: int = 0
    # Warmup epochs before enabling temporal losses.
    temporal_warmup_epochs: int = 0
    # Relation-level temporal gating regularization.
    time_gate_reg_weight: float = 1e-3
    # Optional override: force alpha_r = 1.0 for all relations (useful for debugging/ablation).
    force_time_gate_one: bool = False
    # Two-stage gate training: stage-1 trains global time scale with alpha=1, stage-2 learns alpha.
    time_gate_stage1_epochs: int = 5
    # Freeze global omega parameters after stage-1 (so alpha learns on a fixed spectrum).
    time_gate_freeze_omega_after: bool = True
    # Start epoch for best-loss checkpointing (0 = auto).
    checkpoint_start_epoch: int = 0
    # Path to pretrained alpha_r checkpoint. When set, the gate MLP is loaded
    # and frozen at the start of training, bypassing the two-stage schedule.
    # Only s, omega, and entity/relation embeddings are trained online.
    time_gate_checkpoint: str = ""

    # Initialization and loss parameters (benchmark-matched).
    gamma: float = 200.0  # Init range: embedding_range = (gamma + 2) / dim
    adversarial_temperature: float = 1.0  # Self-adversarial neg sampling temperature
    regularization_weight: float = 1e-5  # L3 global (DistMult) or N3 per-batch (ChronoR)

    # Conflict-aware negatives: sample tails from the same (head, relation) group.
    num_conflict_negatives: int = 1
