from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch

from .config import TKGEConfig
from .encoder import TKGEEncoder, _to_triple
from .time_utils import QueryTime


@dataclass
class TKGERetriever:
    """
    Structural tunnel built on top of TKGEEncoder.

    This module maintains an incrementally-updated KGE model and can score observed triples.
    """

    config: TKGEConfig
    verbose: int = 0
    verbose_epoch_interval: int = 5
    encoder: TKGEEncoder | None = None
    relation_embedder: callable | None = None

    def ensure_initialized(self) -> None:
        if self.encoder is None:
            self.encoder = TKGEEncoder.from_facts(
                [],
                config=self.config,
                verbose=self.verbose,
                verbose_epoch_interval=self.verbose_epoch_interval,
                relation_embedder=self.relation_embedder,
            )

    def update(self, triples: Iterable[object], *, train: bool = True) -> None:
        triples = list(triples)
        if not triples:
            return
        self.ensure_initialized()
        if self.verbose and self.verbose < 3 and self.encoder is not None:
            before = self.encoder.train_calls
            print(f"[TKGE] update triples={len(triples)} train_calls_before={before}")
        self.encoder.update_facts(list(triples), train=train)
        if self.verbose and self.verbose < 3 and self.encoder is not None:
            after = self.encoder.train_calls
            print(f"[TKGE] update_done train_calls_after={after}")

    def pretrain_time_gate(self, pretrain_data_dir: str, **kwargs) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.pretrain_time_gate_from_artifacts(pretrain_data_dir=pretrain_data_dir, **kwargs)

    def inspect_time_gate(
        self,
        relations: List[str] | None = None,
        top_k: int = 10,
        print_report: bool = True,
    ) -> list[dict[str, Any]]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.inspect_time_gate(relations=relations, top_k=top_k, print_report=print_report)

    def save_full_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.save_full_checkpoint(checkpoint_path)

    def load_full_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.load_full_checkpoint(checkpoint_path)

    def save_time_gate_checkpoint(self, checkpoint_path: str) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.save_time_gate_checkpoint(checkpoint_path=checkpoint_path)

    def load_time_gate_checkpoint(self, checkpoint_path: str, strict_dim: bool = True) -> dict[str, Any]:
        self.ensure_initialized()
        assert self.encoder is not None
        return self.encoder.load_time_gate_checkpoint(checkpoint_path=checkpoint_path, strict_dim=strict_dim)

    def score(self, triples: List[object], query_time: QueryTime | None = None) -> np.ndarray:
        if not triples:
            return np.array([], dtype=np.float32)
        self.ensure_initialized()
        assert self.encoder is not None

        if query_time is None:
            query_time = QueryTime.now()

        device = next(self.encoder.model.parameters()).device
        scores = []
        with torch.no_grad():
            for t in triples:
                h, r, tail = _to_triple(t)
                hi = self.encoder.kg.entity2id.get(h)
                ri = self.encoder.kg.relation2id.get(r)
                ti = self.encoder.kg.entity2id.get(tail)
                if hi is None or ri is None or ti is None:
                    scores.append(float("-inf"))
                    continue
                if self.encoder.config.temporal_mode == "romem":
                    rel_text = self.encoder._relation_text_for_ids(torch.tensor([ri], device=device), device)
                    s = self.encoder.model.score_triples(
                        torch.tensor([hi], device=device),
                        torch.tensor([ri], device=device),
                        torch.tensor([ti], device=device),
                        torch.tensor([float(query_time.unix_seconds)], device=device),
                        rel_text_emb=rel_text,
                        ent_lora=self.encoder.lora_ent if self.encoder.config.use_lora else None,
                        rel_lora=self.encoder.lora_rel if self.encoder.config.use_lora else None,
                    )
                else:
                    s = self.encoder.model.score_triples(
                        torch.tensor([hi], device=device),
                        torch.tensor([ri], device=device),
                        torch.tensor([ti], device=device),
                        ent_lora=self.encoder.lora_ent if self.encoder.config.use_lora else None,
                        rel_lora=self.encoder.lora_rel if self.encoder.config.use_lora else None,
                    )
                scores.append(s.item())
        return np.asarray(scores, dtype=np.float32)
