"""Graphiti backend for MultiTQ temporal KGQA evaluation."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Iterable

from pydantic import BaseModel
from tqdm import tqdm

from baselines.graphiti.graphiti_core.nodes import EpisodeType, EntityNode
from baselines.graphiti.graphiti_core.edges import EntityEdge
from baselines.graphiti.graphiti_core.utils.datetime_utils import utc_now
from baselines.graphiti.graphiti_core.models.nodes.node_db_queries import get_entity_node_save_bulk_query
from baselines.graphiti.graphiti_core.models.edges.edge_db_queries import get_entity_edge_save_bulk_query
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.runners.base_backend import GraphitiBackendBase
from benchmarks.types import TemporalTriple, MultiTQQuestion

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class MultiTQGraphitiBackend(GraphitiBackendBase):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
        search_reranker: str | None = None,
        group_id: str | None = None,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
    ):
        super().__init__(
            uri, user, password,
            database=database,
            usage_interval_env="MULTITQ_USAGE_INTERVAL",
            search_reranker=search_reranker,
        )
        self.search_limit = search_limit or int(os.getenv("MULTITQ_SEARCH_LIMIT", "50"))
        self.group_id = group_id or "multitq"
        self._indices_built = False

        ctx_values: list[int] = []
        ctx_env = os.getenv("MULTITQ_CONTEXT_K")
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        elif ctx_env:
            ctx_values = [int(p) for p in ctx_env.split(",") if p.strip().isdigit()]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({k for k in ctx_values if k > 0})

        self.answer_llm_client = self._initialize_answer_llm(
            answer_llm_model or os.getenv("MULTITQ_ANSWER_LLM_MODEL"),
            answer_llm_api_key or os.getenv("MULTITQ_ANSWER_LLM_API_KEY"),
            answer_llm_base_url or os.getenv("MULTITQ_ANSWER_LLM_BASE_URL"),
        )

    # ── Ingestion (from ICEWS Graphiti backend) ────────────────────

    async def ingest_all_triples(self, triples: Iterable[TemporalTriple]) -> None:
        """Ingest all KG triples using batch embeddings + bulk Neo4j UNWIND inserts."""
        all_triples = list(triples)
        if not all_triples:
            return
        if not self._indices_built:
            await self.graphiti.build_indices_and_constraints()
            self._indices_built = True

        embedder = self.graphiti.embedder
        driver = self.graphiti.driver

        # 1. Deduplicate entity nodes by name → stable UUID per entity
        entity_map: dict[str, EntityNode] = {}  # name -> node
        edges: list[EntityEdge] = []

        for triple in all_triples:
            created = triple.timestamp or utc_now()
            time_label = triple.timestamp.strftime("%Y-%m-%d") if triple.timestamp else f"time_{triple.time_id}"

            for name in (triple.head, triple.tail):
                if name not in entity_map:
                    entity_map[name] = EntityNode(
                        name=name, group_id=self.group_id, created_at=created,
                        summary="", attributes={},
                    )

            head_node = entity_map[triple.head]
            tail_node = entity_map[triple.tail]
            fact_text = self._triple_to_sentence(triple)
            edge = EntityEdge(
                name=triple.relation, fact=fact_text, group_id=self.group_id,
                source_node_uuid=head_node.uuid, target_node_uuid=tail_node.uuid,
                created_at=created, valid_at=created,
                attributes={"time_id": triple.time_id, "time_label": time_label},
            )
            edges.append(edge)

        unique_nodes = list(entity_map.values())
        logger.info("Bulk ingest: %d triples → %d unique entities, %d edges",
                     len(all_triples), len(unique_nodes), len(edges))

        # 2. Batch embeddings
        EMBED_BATCH = int(os.getenv("MULTITQ_EMBED_BATCH", "512"))
        node_names = [n.name for n in unique_nodes]
        edge_facts = [e.fact for e in edges]

        logger.info("Computing node name embeddings (%d entities, batch=%d)...", len(node_names), EMBED_BATCH)
        node_embeddings: list[list[float]] = []
        for i in range(0, len(node_names), EMBED_BATCH):
            batch = node_names[i:i + EMBED_BATCH]
            node_embeddings.extend(await embedder.create_batch(batch))
        for node, emb in zip(unique_nodes, node_embeddings):
            node.name_embedding = emb

        logger.info("Computing edge fact embeddings (%d edges, batch=%d)...", len(edge_facts), EMBED_BATCH)
        edge_embeddings: list[list[float]] = []
        for i in range(0, len(edge_facts), EMBED_BATCH):
            batch = edge_facts[i:i + EMBED_BATCH]
            edge_embeddings.extend(await embedder.create_batch(batch))
        for edge, emb in zip(edges, edge_embeddings):
            edge.fact_embedding = emb

        # 3. Bulk Neo4j inserts via UNWIND
        NEO4J_BATCH = int(os.getenv("MULTITQ_NEO4J_BATCH", "1000"))

        # Serialize nodes
        node_dicts = []
        for node in unique_nodes:
            d: dict = {
                "uuid": node.uuid,
                "name": node.name,
                "group_id": node.group_id,
                "summary": node.summary,
                "created_at": node.created_at.isoformat() if node.created_at else None,
                "name_embedding": node.name_embedding,
                "labels": list(set(node.labels + ["Entity"])),
            }
            d.update(node.attributes or {})
            node_dicts.append(d)

        node_query = get_entity_node_save_bulk_query(driver.provider, node_dicts)
        logger.info("Inserting %d entity nodes (batch=%d)...", len(node_dicts), NEO4J_BATCH)
        for i in tqdm(range(0, len(node_dicts), NEO4J_BATCH), desc="Bulk node insert", leave=False):
            batch = node_dicts[i:i + NEO4J_BATCH]
            session = driver.session()
            try:
                # Re-generate query for each batch (FalkorDB returns per-node queries)
                if isinstance(node_query, list):
                    for q, params in node_query:
                        await session.run(q, **params)
                else:
                    await session.run(node_query, nodes=batch)
            finally:
                await session.close()

        # Serialize edges
        edge_dicts = []
        for edge in edges:
            d = {
                "uuid": edge.uuid,
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "name": edge.name,
                "fact": edge.fact,
                "group_id": edge.group_id,
                "episodes": edge.episodes or [],
                "created_at": edge.created_at.isoformat() if edge.created_at else None,
                "expired_at": edge.expired_at.isoformat() if edge.expired_at else None,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                "fact_embedding": edge.fact_embedding,
            }
            d.update(edge.attributes or {})
            edge_dicts.append(d)

        edge_query = get_entity_edge_save_bulk_query(driver.provider)
        logger.info("Inserting %d entity edges (batch=%d)...", len(edge_dicts), NEO4J_BATCH)
        for i in tqdm(range(0, len(edge_dicts), NEO4J_BATCH), desc="Bulk edge insert", leave=False):
            batch = edge_dicts[i:i + NEO4J_BATCH]
            session = driver.session()
            try:
                await session.run(edge_query, entity_edges=batch)
            finally:
                await session.close()

        self.increment_episode_count(len(all_triples))
        logger.info("Bulk ingest complete: %d triples ingested", len(all_triples))

    # ── QA evaluation ──────────────────────────────────────────────

    async def evaluate_question(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        retrieval_metrics, context, details = await self._retrieve(question)
        llm_metrics, answers = await self._answer(question, context)
        accuracy_metrics, match_strategies = self._verify_answers(question, answers)
        context_strs = [getattr(e, "fact", "") or "" for e in context]
        return (
            retrieval_metrics,
            {**llm_metrics, **accuracy_metrics},
            {**details, "llm_answers": answers, "context_docs": context_strs,
             "match_strategies": match_strategies},
        )

    async def _retrieve(self, question: MultiTQQuestion) -> tuple[dict, list, dict]:
        try:
            results = await self._search_edges(question.question, self.search_limit)
        except Exception as exc:
            logger.warning("Search failed for quid=%s: %s", question.quid, exc)
            results = []

        docs = [getattr(e, "fact", "") or "" for e in results]
        metrics: dict[str, float] = {}
        matching_fact = ""
        for k in self.answer_context_sizes:
            subset = docs[:k]
            found = self._check_answer_in_context(question.answers, subset)
            metrics[f"answer_in_context@{k}"] = 1.0 if found else 0.0
            if found and not matching_fact:
                matching_fact = self._find_matching_fact(question.answers, subset)

        # Hits@k and MRR — find rank of first doc containing a gold answer
        match_rank: int | None = None
        for idx, doc in enumerate(docs, start=1):
            if not doc:
                continue
            doc_lower = doc.lower().replace("_", " ")
            for answer in question.answers:
                if answer.lower().replace("_", " ") in doc_lower:
                    match_rank = idx
                    break
            if match_rank is not None:
                break
        metrics["hits@1"] = 1.0 if match_rank == 1 else 0.0
        metrics["hits@3"] = 1.0 if match_rank is not None and match_rank <= 3 else 0.0
        metrics["hits@10"] = 1.0 if match_rank is not None and match_rank <= 10 else 0.0
        metrics["mrr"] = 1.0 / match_rank if match_rank else 0.0

        return metrics, results, {"matching_fact": matching_fact}

    async def _answer(self, question: MultiTQQuestion, context_edges) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client or self.graphiti.llm_client
        for k in self.answer_context_sizes:
            subset = list(context_edges)[:k]
            if not subset:
                metrics[f"llm@{k}_context_size"] = 0.0
                continue
            facts = []
            for edge in subset:
                fact = getattr(edge, "fact", "") or ""
                if fact:
                    facts.append(f"- {fact}")
            if not facts:
                metrics[f"llm@{k}_context_size"] = 0.0
                continue
            prompt = (
                f"Question: {question.question}\n\n"
                f"Supporting facts:\n{os.linesep.join(facts)}\n\n"
                'Provide a concise answer grounded in the facts. '
                'Respond with JSON {{"answer": "<text>"}}.'
            )
            messages = [
                Message(role="system", content="You answer questions using only provided facts."),
                Message(role="user", content=prompt),
            ]
            prediction = ""
            try:
                response = await client.generate_response(messages, response_model=QAResponse)
                prediction = response.get("answer", "")
            except Exception as exc:
                logger.warning("LLM answer failed for quid=%s (top-%s): %s", question.quid, k, exc)
            answers[k] = prediction
            metrics[f"llm@{k}_context_size"] = float(len(facts))
        return metrics, answers

    def _verify_answers(
        self, question: MultiTQQuestion, answers: dict[int, str]
    ) -> tuple[dict[str, float], dict[int, str]]:
        metrics: dict[str, float] = {}
        strategies: dict[int, str] = {}
        for k, prediction in answers.items():
            is_correct, strategy = verify_answer(
                prediction, question.answers, question.answer_type,
            )
            metrics[f"llm@{k}_accuracy"] = 1.0 if is_correct else 0.0
            strategies[k] = strategy
        return metrics, strategies

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _triple_to_sentence(triple: TemporalTriple) -> str:
        date_str = triple.timestamp.strftime("%Y-%m-%d")
        return f"On {date_str}, {triple.head} {triple.relation} {triple.tail}."

    @staticmethod
    def _check_answer_in_context(answers: list[str], docs: list[str]) -> bool:
        context = " ".join(docs).lower().replace("_", " ")
        for answer in answers:
            if answer.lower().replace("_", " ") in context:
                return True
        return False

    @staticmethod
    def _find_matching_fact(answers: list[str], docs: list[str]) -> str:
        for doc in docs:
            doc_lower = doc.lower().replace("_", " ")
            for answer in answers:
                if answer.lower().replace("_", " ") in doc_lower:
                    return doc
        return ""

    def _initialize_answer_llm(self, model: str | None, api_key: str | None, base_url: str | None):
        if not any([model, api_key, base_url]):
            return None
        config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        name = (config.model or "").lower()
        if not name or name.startswith("gpt"):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)
