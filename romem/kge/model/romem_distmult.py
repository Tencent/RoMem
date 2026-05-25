from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def _rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate complex-valued embedding represented as concatenated real/imag parts.
    Optimized implementation using native complex tensors.

    x: [..., 2d] where first d are real, last d are imag
    theta: [..., d] rotation angles
    """
    # Split into real and imaginary parts
    # We assume the layout is [real_1, ..., real_d, imag_1, ..., imag_d]
    d = x.shape[-1] // 2
    x_real, x_imag = x[..., :d], x[..., d:]

    # Construct complex tensor
    x_c = torch.complex(x_real, x_imag)

    # Create rotation phasor e^{i * theta}
    # polar(abs, angle) creates r * (cos(a) + i*sin(a))
    phasor = torch.polar(torch.ones_like(theta), theta)

    # Rotate via efficient complex multiplication
    y_c = x_c * phasor

    # Concatenate back to original layout
    return torch.cat([y_c.real, y_c.imag], dim=-1)


class RoMemDistMultModel(nn.Module):
    """
    RoMem: Time-conditioned Knowledge Graph Embedding.

    Mechanics:
    - Entities/Relations live in R^{2d} (interpreted as C^d).
    - Time is modeled as a rotation in the complex plane (U(1) group).
    - The frequency spectrum is learned to adapt to the dataset's time granularity.
    """

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            dim: int,
            rel_text_dim: int,
            gamma: float = 200.0,
            # Default Prior: 1 unit = 1 Day (Assuming input t_scalar is in seconds)
            # 1.0 / 86400.0 means the model starts by 'expecting' day-level changes.
            init_time_scale: float = 1.0 / 86400.0,
            force_time_gate_one: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.force_time_gate_one = bool(force_time_gate_one)

        # -------------------------------------------------------
        # 1. Embeddings (Structural Stream)
        # -------------------------------------------------------
        epsilon = 2.0
        embedding_range = (gamma + epsilon) / (2 * dim)

        self.ent_emb = nn.Embedding(num_entities, 2 * dim)
        self.rel_emb = nn.Embedding(num_relations, 2 * dim)
        nn.init.uniform_(self.ent_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.rel_emb.weight, -embedding_range, embedding_range)

        # -------------------------------------------------------
        # 2. Learnable Time Dynamics
        # -------------------------------------------------------
        # Learnable Time Scale (Log-space for stability)
        # Allows model to auto-tune from "Day-level" to "Year-level" sensitivity.
        self.log_time_scale = nn.Parameter(torch.tensor(np.log(init_time_scale)))

        # Learnable Frequency Base (Log-space)
        # Controls the "bandwidth" of the RoPE spectrum.
        self.log_inv_freq_base = nn.Parameter(torch.tensor(np.log(10000.0)))

        # Buffer for RoPE indices (0, 1, ..., d-1)
        self.register_buffer("freq_indices", torch.arange(0, dim, dtype=torch.float32) / dim, persistent=False)

        # Semantic Temporal Gate (Learnable MLP)
        # Controls HOW MUCH a relation rotates using its text embedding.
        # (e.g., 'born in' -> alpha≈0, 'president of' -> alpha≈1)
        self.rel_text_dim = int(rel_text_dim)
        hidden = max(32, min(128, self.rel_text_dim // 2))
        self.speed_mlp = nn.Sequential(
            nn.Linear(self.rel_text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def time_gate_alpha(self, r_text_emb: torch.Tensor) -> torch.Tensor:
        """
        Computes relation-specific rotation strength alpha_r in (0, 1).
        """
        if self.force_time_gate_one:
            return torch.ones((r_text_emb.size(0), 1), dtype=torch.float32, device=r_text_emb.device)

        if r_text_emb is None:
            raise ValueError("rel_text_emb is required for semantic time gating.")

        logits = self.speed_mlp(r_text_emb)
        return torch.sigmoid(logits)

    def _theta(self, t_scalar: torch.Tensor) -> torch.Tensor:
        """
        Computes dynamic rotation angles based on learned scale and frequencies.
        t_scalar: [Batch] raw time values (e.g., unix seconds)
        """
        # 1. Get Learnable Scale (Safe Clamp to prevent collapse)
        # We ensure the scale never drops below 1e-9 (turning off time entirely)
        scale = self.log_time_scale.exp().clamp(min=1e-9)

        # 2. Get Learnable Base
        base = self.log_inv_freq_base.exp().clamp(min=2.0)

        # 3. Compute Inverse Frequencies
        # inv_freq = 1 / (base ^ (i / d))
        inv_freq = 1.0 / (base ** self.freq_indices)

        # 4. Compute Theta
        # t: [Batch, 1]
        t = t_scalar.float().unsqueeze(-1) * scale

        # result: [Batch, Dim]
        return t * inv_freq.unsqueeze(0)

    def score_triples(
            self,
            h_idx: torch.Tensor,
            r_idx: torch.Tensor,
            t_idx: torch.Tensor,
            t_scalar: torch.Tensor,
            rel_text_emb: Optional[torch.Tensor] = None,
            alpha_override: Optional[torch.Tensor] = None,
            ent_lora: Optional[nn.Module] = None,
            rel_lora: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Forward pass for Training/Scoring.
        """
        # A. Fetch Embeddings
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        tail = self.ent_emb(t_idx)

        # B. Apply LoRA (if present)
        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            tail = tail + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)

        # C. Compute Dynamic Rotation
        theta = self._theta(t_scalar)  # [B, d] Base rotation for time t
        if alpha_override is None:
            alpha = self.time_gate_alpha(rel_text_emb)  # [B, 1] Relation sensitivity
        else:
            alpha = alpha_override
        theta = theta * alpha  # Gated rotation

        # D. Rotate Head and Tail
        h_rot = _rotate(h, theta)
        tail_rot = _rotate(tail, theta)

        # E. Score (Real-valued dot product in 2d space)
        return torch.sum(h_rot * r * tail_rot, dim=-1)

    def triple_embedding(
            self,
            h_idx: torch.Tensor,
            r_idx: torch.Tensor,
            t_idx: torch.Tensor,
            t_scalar: torch.Tensor,
            rel_text_emb: Optional[torch.Tensor] = None,
            alpha_override: Optional[torch.Tensor] = None,
            ent_lora: Optional[nn.Module] = None,
            rel_lora: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Generates the dense embedding for downstream retrieval/clustering.
        """
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        tail = self.ent_emb(t_idx)

        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            tail = tail + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)

        # Apply Rotation
        theta = self._theta(t_scalar)
        if alpha_override is None:
            alpha = self.time_gate_alpha(rel_text_emb)
        else:
            alpha = alpha_override
        theta = theta * alpha

        h_rot = _rotate(h, theta)
        tail_rot = _rotate(tail, theta)

        return torch.cat([h_rot * r, tail_rot], dim=-1)
