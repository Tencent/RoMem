from __future__ import annotations

import torch
import torch.nn as nn


class DistMultModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int,
                 gamma: float = 200.0):
        super().__init__()
        epsilon = 2.0
        embedding_range = (gamma + epsilon) / dim

        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.ent_emb.weight, -embedding_range, embedding_range)
        nn.init.uniform_(self.rel_emb.weight, -embedding_range, embedding_range)

    def score_triples(self, h_idx, r_idx, t_idx, ent_lora=None, rel_lora=None):
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.ent_emb(t_idx)
        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            t = t + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)
        return torch.sum(h * r * t, dim=-1)

    def triple_embedding(self, h_idx, r_idx, t_idx, ent_lora=None, rel_lora=None):
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.ent_emb(t_idx)
        if ent_lora is not None:
            h = h + ent_lora(h_idx)
            t = t + ent_lora(t_idx)
        if rel_lora is not None:
            r = r + rel_lora(r_idx)
        return torch.cat([h * r, t], dim=-1)
