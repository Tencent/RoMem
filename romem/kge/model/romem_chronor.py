from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .romem_distmult import _rotate


class RoMemChronoRModel(nn.Module):
    """
    RoMem with ChronoR backbone: k-component bilinear model with
    functional temporal rotation.

    Key differences from RoMemDistMultModel:
    - k components per entity/relation (multi-component decomposition)
    - Separate forward and inverse relation embeddings (rel_inv_emb)
    - Scoring: sum(h_rot * r * r_inv * t_rot) across all k*2d dimensions

    The trace-diagonal sum in ChronoR's original k-component scoring
    reduces to a flat element-wise product + sum, so evaluation uses the
    same efficient dot-product approach as RoMemDistMultModel.

    Compatible with 1-vs-all CE training via the unrotation trick:
      score(t_j) = _rotate(h_rot * r * r_inv, -theta) . t_j
    """

    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            dim: int,
            rel_text_dim: int,
            k: int = 3,
            gamma: float = 200.0,
            init_time_scale: float = 1.0 / 86400.0,
            force_time_gate_one: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.k = k
        self.force_time_gate_one = bool(force_time_gate_one)

        total_dim = k * 2 * dim

        # -------------------------------------------------------
        # 1. Embeddings (k-component structure)
        # -------------------------------------------------------
        epsilon = 2.0
        embedding_range = (gamma + epsilon) / total_dim

        self.ent_emb = nn.Embedding(num_entities, total_dim)
        self.rel_emb = nn.Embedding(num_relations, total_dim)
        self.rel_inv_emb = nn.Embedding(num_relations, total_dim)
        nn.init.uniform_(self.ent_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.rel_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.rel_inv_emb.weight, -embedding_range, embedding_range)

        # -------------------------------------------------------
        # 2. Learnable Time Dynamics (k*d frequencies)
        # -------------------------------------------------------
        kd = k * dim
        self.log_time_scale = nn.Parameter(torch.tensor(np.log(init_time_scale)))
        self.log_inv_freq_base = nn.Parameter(torch.tensor(np.log(10000.0)))
        self.register_buffer(
            "freq_indices",
            torch.arange(0, kd, dtype=torch.float32) / kd,
            persistent=False)

        # -------------------------------------------------------
        # 3. Semantic Temporal Gate
        # -------------------------------------------------------
        self.rel_text_dim = int(rel_text_dim)
        hidden = max(32, min(128, self.rel_text_dim // 2))
        self.speed_mlp = nn.Sequential(
            nn.Linear(self.rel_text_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def time_gate_alpha(self, r_text_emb: torch.Tensor) -> torch.Tensor:
        if self.force_time_gate_one:
            return torch.ones(
                (r_text_emb.size(0), 1),
                dtype=torch.float32, device=r_text_emb.device)
        if r_text_emb is None:
            raise ValueError("rel_text_emb is required for semantic time gating.")
        return torch.sigmoid(self.speed_mlp(r_text_emb))

    def _theta(self, t_scalar: torch.Tensor) -> torch.Tensor:
        """
        Computes rotation angles for k*d complex dimensions.
        t_scalar: [Batch] raw time values (e.g., unix seconds)
        Returns: [Batch, k*d]
        """
        scale = self.log_time_scale.exp().clamp(min=1e-9)
        base = self.log_inv_freq_base.exp().clamp(min=2.0)
        inv_freq = 1.0 / (base ** self.freq_indices)
        t = t_scalar.float().unsqueeze(-1) * scale
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
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        r_inv = self.rel_inv_emb(r_idx)
        tail = self.ent_emb(t_idx)

        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            tail = tail + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)

        theta = self._theta(t_scalar)
        if alpha_override is None:
            alpha = self.time_gate_alpha(rel_text_emb)
        else:
            alpha = alpha_override
        theta = theta * alpha

        h_rot = _rotate(h, theta)
        tail_rot = _rotate(tail, theta)

        return torch.sum(h_rot * r * r_inv * tail_rot, dim=-1)

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
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        r_inv = self.rel_inv_emb(r_idx)
        tail = self.ent_emb(t_idx)

        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            tail = tail + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)

        theta = self._theta(t_scalar)
        if alpha_override is None:
            alpha = self.time_gate_alpha(rel_text_emb)
        else:
            alpha = alpha_override
        theta = theta * alpha

        h_rot = _rotate(h, theta)
        tail_rot = _rotate(tail, theta)

        return torch.cat([h_rot * r * r_inv, tail_rot], dim=-1)
