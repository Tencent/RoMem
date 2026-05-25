from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from romem.ingestion.embedding_store import EmbeddingStore
from romem.utils.misc_utils import compute_mdhash_id


@dataclass
class GraphBuilder:
    """
    Manages chunk/entity/fact embedding stores for graph construction.
    Provides a small API surface so callers don't manipulate stores directly.
    """

    chunk_store: EmbeddingStore
    entity_store: EmbeddingStore
    fact_store: EmbeddingStore
    fact_encoder: Optional[Any] = None

    @classmethod
    def from_paths(
        cls,
        embedding_model: Any,
        working_dir: str,
        batch_size: int,
        fact_encoder: Optional[Any] = None,
        use_tkge_facts: bool = False,
    ) -> "GraphBuilder":
        os.makedirs(working_dir, exist_ok=True)
        chunk_store = EmbeddingStore(
            embedding_model,
            os.path.join(working_dir, "chunk_embeddings"),
            batch_size,
            "chunk",
        )
        entity_store = EmbeddingStore(
            embedding_model,
            os.path.join(working_dir, "entity_embeddings"),
            batch_size,
            "entity",
        )
        fact_store = EmbeddingStore(
            embedding_model,
            os.path.join(working_dir, "fact_embeddings"),
            batch_size,
            "fact",
        )
        if fact_encoder is None and use_tkge_facts:
            from ..kge.encoder import TKGEEncoder
            fact_encoder = TKGEEncoder.from_facts([])
        return cls(
            chunk_store=chunk_store,
            entity_store=entity_store,
            fact_store=fact_store,
            fact_encoder=fact_encoder,
        )

    # Chunk helpers
    def add_chunks(self, docs: List[str]) -> None:
        self.chunk_store.insert_strings(docs)

    def missing_chunks(self, docs: List[str]) -> Dict[str, Dict[str, str]]:
        return self.chunk_store.get_missing_string_hash_ids(docs)

    def delete_chunks_by_texts(self, docs: List[str]) -> None:
        ids = [self.chunk_store.get_hash_id(d) for d in docs if d in self.chunk_store.text_to_hash_id]
        self.chunk_store.delete(ids)

    # Entity helpers
    def add_entities(self, entities: List[str]) -> None:
        self.entity_store.insert_strings(entities)

    def delete_entities(self, entities: List[str]) -> None:
        ids = [self.entity_store.get_hash_id(e) for e in entities if e in self.entity_store.text_to_hash_id]
        self.entity_store.delete(ids)

    # Fact helpers
    def _fact_to_text(self, fact: Any) -> str:
        if hasattr(fact, "triple"):
            return str(getattr(fact, "triple"))
        return str(fact)

    def add_facts(self, facts: List[Any]) -> None:
        if self.fact_encoder:
            self.fact_encoder.update_facts(facts)
            vectors = self.fact_encoder.encode_facts(facts)
            nodes_dict = {}
            fact_texts = [self._fact_to_text(f) for f in facts]
            for text, vec in zip(fact_texts, vectors):
                key = compute_mdhash_id(text, prefix="fact-")
                nodes_dict[key] = {"content": text, "embedding": vec}
            self.fact_store._upsert(
                list(nodes_dict.keys()),
                [v["content"] for v in nodes_dict.values()],
                [v["embedding"] for v in nodes_dict.values()],
            )
        else:
            self.fact_store.insert_strings([self._fact_to_text(f) for f in facts])

    def delete_facts(self, facts: List[Any]) -> None:
        fact_texts = [self._fact_to_text(f) for f in facts]
        ids = [self.fact_store.get_hash_id(f) for f in fact_texts if f in self.fact_store.text_to_hash_id]
        self.fact_store.delete(ids)
        if self.fact_encoder:
            self.fact_encoder.remove_facts(facts)

    # Fetchers
    def get_chunk_rows(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return self.chunk_store.get_rows(ids)

    def get_entity_rows(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return self.entity_store.get_rows(ids)

    def get_fact_rows(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return self.fact_store.get_rows(ids)

    def get_chunk_embeddings(self, ids: List[str]):
        return self.chunk_store.get_embeddings(ids)

    def get_entity_embeddings(self, ids: List[str]):
        return self.entity_store.get_embeddings(ids)

    def get_fact_embeddings(self, ids: List[str]):
        return self.fact_store.get_embeddings(ids)

    def all_chunk_rows(self) -> Dict[str, Dict[str, Any]]:
        return self.chunk_store.get_all_id_to_rows()

    def all_entity_rows(self) -> Dict[str, Dict[str, Any]]:
        return self.entity_store.get_all_id_to_rows()

    def all_fact_rows(self) -> Dict[str, Dict[str, Any]]:
        return self.fact_store.get_all_id_to_rows()

    # Introspection
    def all_chunk_ids(self) -> List[str]:
        return self.chunk_store.get_all_ids()

    def all_entity_ids(self) -> List[str]:
        return self.entity_store.get_all_ids()

    def all_fact_ids(self) -> List[str]:
        return self.fact_store.get_all_ids()

    # Text/id helpers
    def chunk_texts(self) -> set:
        return self.chunk_store.get_all_texts()

    def chunk_id_for_text(self, text: str) -> Optional[str]:
        return self.chunk_store.get_hash_id(text) if text in self.chunk_store.text_to_hash_id else None

    def fact_id_for_text(self, text: str) -> Optional[str]:
        return self.fact_store.get_hash_id(text) if text in self.fact_store.text_to_hash_id else None

    def entity_id_for_text(self, text: str) -> Optional[str]:
        return self.entity_store.get_hash_id(text) if text in self.entity_store.text_to_hash_id else None
