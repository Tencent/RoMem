"""Mem0 backend for MultiTQ temporal KGQA evaluation (local Mem0: FAISS + Neo4j)."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
import os
import sys
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.evaluators.multitq_verifier import verify_answer
from benchmarks.runners.base_backend import Mem0BackendBase
from benchmarks.types import TemporalTriple, MultiTQQuestion

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class MultiTQMem0Backend(Mem0BackendBase):
    def __init__(
        self,
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        neo4j_database: str | None = None,
    ):
        self.search_limit = search_limit or int(os.getenv("MULTITQ_SEARCH_LIMIT", "50"))
        self.answer_llm_model = answer_llm_model or os.getenv("MULTITQ_ANSWER_LLM_MODEL") or os.getenv("OPENAI_MODEL")
        self.answer_llm_api_key = answer_llm_api_key or os.getenv("MULTITQ_ANSWER_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.answer_llm_base_url = answer_llm_base_url or os.getenv("MULTITQ_ANSWER_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

        ctx_values: list[int] = []
        ctx_env = os.getenv("MULTITQ_CONTEXT_K")
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        elif ctx_env:
            ctx_values = [int(p) for p in ctx_env.split(",") if p.strip().isdigit()]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({k for k in ctx_values if k > 0})

        # Ensure mem0 is importable
        mem0_root = Path(__file__).resolve().parents[3] / "baselines" / "mem0"
        mem0_path = str(mem0_root)
        if mem0_path not in sys.path:
            sys.path.insert(0, mem0_path)
        _orig_version = importlib_metadata.version

        def _safe_version(name: str):
            if name == "mem0ai":
                return "0.0.0"
            return _orig_version(name)

        importlib_metadata.version = _safe_version  # type: ignore
        try:
            from mem0.configs.base import MemoryConfig
            from mem0.graphs.configs import GraphStoreConfig, Neo4jConfig
            from mem0.embeddings.configs import EmbedderConfig
            from mem0.configs.vector_stores.faiss import FAISSConfig
        finally:
            importlib_metadata.version = _orig_version  # type: ignore

        mem_user = os.getenv("MEM0_USER_ID", "multitq")
        mem0_dir = os.getenv("MEM0_DIR", str(Path("outputs/mem0_multitq").resolve()))
        os.environ["MEM0_DIR"] = mem0_dir
        self._neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self._neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self._neo4j_database = neo4j_database or os.getenv("MULTITQ_NEO4J_DATABASE") or "neo4j"

        neo4j_cfg = Neo4jConfig(
            url=self._neo4j_uri, username=self._neo4j_user,
            password=self._neo4j_password, database=self._neo4j_database, base_label=True,
        )
        graph_store = GraphStoreConfig(provider="neo4j", config=neo4j_cfg)
        mem0_embed_model = os.getenv("MEM0_EMBED_MODEL") or "text-embedding-3-small"
        mem0_embed_dims = int(os.getenv("MEM0_EMBED_DIMS", "1536"))
        embed_provider = os.getenv("MEM0_EMBED_PROVIDER", "openai")
        embed_config = {
            "model": mem0_embed_model,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "embedding_dims": mem0_embed_dims,
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
        }
        if embed_provider.lower() in {"huggingface", "hf", "sentence_transformers"}:
            embed_config.setdefault("model_kwargs", {})["trust_remote_code"] = True
        config = MemoryConfig(
            graph_store=graph_store,
            embedder=EmbedderConfig(provider=embed_provider, config=embed_config),
        )
        config.vector_store.provider = "faiss"
        config.vector_store.config = FAISSConfig(
            path=os.getenv("MEM0_VECTOR_PATH", str(Path(mem0_dir) / "faiss_run")),
            collection_name=mem_user,
            embedding_model_dims=mem0_embed_dims,
            distance_strategy="cosine",
            normalize_L2=True,
        )
        mem0_llm_model = os.getenv("OPENAI_MODEL")
        mem0_llm_api_key = os.getenv("OPENAI_API_KEY")
        mem0_llm_base_url = os.getenv("OPENAI_BASE_URL")
        config.llm.provider = "openai"
        config.llm.config = {
            "model": mem0_llm_model,
            "api_key": mem0_llm_api_key,
            "openai_base_url": mem0_llm_base_url,
        }
        super().__init__(
            mem0_config=config, mem0_user_id=mem_user,
            usage_interval_env="MULTITQ_USAGE_INTERVAL", mem0_dir=mem0_dir,
        )
        self._maybe_wipe_all_stores()

    # ── Ingestion ──────────────────────────────────────────────────

    async def ingest_all_triples(self, triples: Iterable[TemporalTriple]) -> None:
        """Ingest all KG triples using batch embeddings + direct FAISS insert.

        Bypasses memory.add() to avoid per-item embedding, duplicate search,
        and per-item disk saves.  Instead: batch embed → bulk FAISS insert.
        """
        import asyncio
        import hashlib
        import uuid
        from datetime import datetime, timezone
        from tqdm import tqdm

        all_triples = list(triples)
        if not all_triples:
            return

        # Build sentences
        sentences: list[str] = []
        time_labels: list[str] = []
        for t in all_triples:
            sentences.append(self._triple_to_sentence(t))
            time_labels.append(
                t.timestamp.strftime("%Y-%m-%d") if t.timestamp else f"time_{t.time_id}"
            )

        logger.info("Bulk ingest: %d triples, embedding...", len(sentences))

        # Batch embed
        EMBED_BATCH = int(os.getenv("MULTITQ_EMBED_BATCH", "512"))
        embedder = self.memory.embedding_model
        all_embeddings: list[list[float]] = []

        # Try batch-native path first (BGE-M3, HuggingFace)
        if self._embedder_supports_batch(embedder):
            for i in tqdm(range(0, len(sentences), EMBED_BATCH), desc="Batch embed", leave=False):
                batch = sentences[i:i + EMBED_BATCH]
                embs = await asyncio.to_thread(embedder.embed, batch)
                all_embeddings.extend(embs)
        elif hasattr(embedder, "client") and hasattr(embedder.client, "embeddings"):
            # OpenAI-compatible: call client.embeddings.create with batch input
            model = embedder.config.model
            dims = embedder.config.embedding_dims
            for i in tqdm(range(0, len(sentences), EMBED_BATCH), desc="Batch embed", leave=False):
                batch = [s.replace("\n", " ") for s in sentences[i:i + EMBED_BATCH]]
                resp = await asyncio.to_thread(
                    embedder.client.embeddings.create,
                    input=batch, model=model, dimensions=dims,
                )
                all_embeddings.extend([d.embedding for d in resp.data])
        else:
            # Fallback: one-by-one via thread pool
            logger.warning("No batch embedding support; falling back to sequential embed")
            for i in tqdm(range(len(sentences)), desc="Embed", leave=False):
                emb = await asyncio.to_thread(embedder.embed, sentences[i])
                all_embeddings.append(emb)

        # Build payloads + IDs matching Mem0's expected format
        now_iso = datetime.now(timezone.utc).isoformat()
        ids: list[str] = []
        payloads: list[dict] = []
        for i, sentence in enumerate(sentences):
            ids.append(str(uuid.uuid4()))
            payloads.append({
                "data": sentence,
                "hash": hashlib.md5(sentence.encode()).hexdigest(),
                "created_at": now_iso,
                "user_id": self.mem_user_id,
                "time_label": time_labels[i],
            })

        # Bulk FAISS insert
        FAISS_BATCH = int(os.getenv("MULTITQ_FAISS_BATCH", "5000"))
        vs = self.memory.vector_store
        logger.info("Inserting %d vectors into FAISS (batch=%d)...", len(all_embeddings), FAISS_BATCH)
        for i in tqdm(range(0, len(all_embeddings), FAISS_BATCH), desc="FAISS insert", leave=False):
            end = min(i + FAISS_BATCH, len(all_embeddings))
            vs.insert(
                vectors=all_embeddings[i:end],
                payloads=payloads[i:end],
                ids=ids[i:end],
            )

        self.increment_episode_count(len(all_triples))
        logger.info("Bulk ingest complete: %d triples ingested", len(all_triples))

    @staticmethod
    def _embedder_supports_batch(embedder) -> bool:
        """Check if embedder.embed() natively accepts a list of strings."""
        import inspect
        try:
            sig = inspect.signature(embedder.embed)
            first_param = list(sig.parameters.values())[0]
            hints = str(first_param.annotation)
            return "Sequence" in hints or "list" in hints.lower()
        except Exception:
            return False

    # ── QA evaluation ──────────────────────────────────────────────

    async def evaluate_question(
        self, question: MultiTQQuestion
    ) -> tuple[dict[str, float], dict[str, float], dict]:
        retrieval_metrics, context_docs, details = await self._retrieve(question)
        llm_metrics, answers = await self._answer(question, context_docs)
        accuracy_metrics, match_strategies = self._verify_answers(question, answers)
        return (
            retrieval_metrics,
            {**llm_metrics, **accuracy_metrics},
            {**details, "llm_answers": answers, "context_docs": context_docs,
             "match_strategies": match_strategies},
        )

    async def _retrieve(self, question: MultiTQQuestion) -> tuple[dict, list, dict]:
        try:
            retrieval = self.memory.search(
                question.question, user_id=self.mem_user_id, limit=self.search_limit,
            )
        except Exception as exc:
            logger.warning("Mem0 search failed for quid=%s: %s", question.quid, exc)
            retrieval = []

        docs: list[str] = []
        if isinstance(retrieval, dict) and "results" in retrieval:
            for res in retrieval.get("results", []):
                if res:
                    docs.append(res.get("memory") or res.get("text") or "")
        elif isinstance(retrieval, list):
            for res in retrieval:
                mem = res.get("memory", "") if isinstance(res, dict) else str(res)
                docs.append(mem)

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

        return metrics, docs, {"matching_fact": matching_fact}

    async def _answer(self, question: MultiTQQuestion, context_docs: list[str]) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self._get_answer_llm_client()
        if client is None:
            return metrics, answers
        for k in self.answer_context_sizes:
            subset = [d for d in context_docs[:k] if d]
            if not subset:
                metrics[f"llm@{k}_context_size"] = 0.0
                continue
            facts = [f"- {doc}" for doc in subset]
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

    def _get_answer_llm_client(self):
        if not self.answer_llm_api_key:
            return None
        cfg = LLMConfig(model=self.answer_llm_model, api_key=self.answer_llm_api_key, base_url=self.answer_llm_base_url)
        if (cfg.model or "").lower().startswith("gpt"):
            return OpenAIClient(config=cfg)
        return OpenAIGenericClient(config=cfg)

    def _maybe_wipe_all_stores(self) -> None:
        if os.getenv("MEM0_CLEAR_ALL", "0") not in {"1", "true", "True"}:
            return
        self._wipe_all_stores()

    def _wipe_all_stores(self) -> None:
        try:
            if getattr(self.memory, "vector_store", None):
                self.memory.vector_store.reset()
            self._clear_default_database()
            logger.info("Mem0 stores wiped (vector + graph)")
        except Exception as exc:
            logger.warning("Mem0 store wipe failed: %s", exc)

    def _clear_default_database(self) -> None:
        try:
            from neo4j import GraphDatabase
        except Exception as exc:
            logger.warning("Neo4j driver import failed: %s", exc)
            return
        driver = GraphDatabase.driver(self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password))
        try:
            with driver.session(database=self._neo4j_database) as session:
                total_deleted = 0
                while True:
                    result = session.run(
                        """
                        CALL {
                            MATCH (n)
                            WITH n LIMIT $batch_size
                            DETACH DELETE n
                            RETURN COUNT(*) AS deleted
                        }
                        RETURN deleted
                        """,
                        batch_size=1000,
                    ).single()
                    deleted = result["deleted"] if result else 0
                    total_deleted += deleted
                    if deleted == 0:
                        break
                logger.info("Cleared %s nodes from Neo4j (db=%s)", total_deleted, self._neo4j_database)
        finally:
            driver.close()

