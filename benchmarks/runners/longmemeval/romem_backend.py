"""RoMem backend for LongMemEval benchmarking."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from baselines.graphiti.graphiti_core.prompts.models import Message

from benchmarks.runners.shared_utils import (
    initialize_answer_llm,
    parse_bool,
    parse_optional_bool,
)
from benchmarks.types import LongmemevalExample
from benchmarks.runners.romem_utils import apply_romem_config
from dataset.longmemeval.code.retrieval.eval_utils import (
    evaluate_retrieval,
    evaluate_retrieval_turn2session,
)

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class JudgeResponse(BaseModel):
    label: str


def _tokenize(value: str) -> list[str]:
    return [tok for tok in (value or '').lower().split() if tok]


def _safe_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "example")


class LongmemevalRoMemBackend:
    def __init__(
        self,
        data_path: str | None = None,
        reset_mode: str = 'delete',
        search_limit: int | None = None,
        answer_context_sizes: list[int] | None = None,
        answer_llm_model: str | None = None,
        answer_llm_api_key: str | None = None,
        answer_llm_base_url: str | None = None,
        judge_llm_model: str | None = None,
        judge_llm_api_key: str | None = None,
        judge_llm_base_url: str | None = None,
        disable_judge: bool = False,
        romem_config: dict | None = None,
        romem_save_dir: str | None = None,
        romem_llm_model: str | None = None,
        romem_embedding_model: str | None = None,
        romem_openie_mode: str | None = None,
        romem_temporal_awareness: str | None = None,
        romem_enable_tkge_tunnel: str | None = None,
        romem_tkge_verbose: int | None = None,
    ):
        self.reset_mode = reset_mode.lower()
        self.search_limit = search_limit or int(os.getenv('LONGMEM_SEARCH_LIMIT', '50'))
        ctx_values: list[int] = []
        ctx_env = os.getenv('LONGMEM_CONTEXT_K')
        if answer_context_sizes:
            ctx_values = answer_context_sizes
        elif ctx_env:
            ctx_values = [
                int(part)
                for part in ctx_env.split(',')
                if part.strip().isdigit()
            ]
        if not ctx_values:
            ctx_values = [5, 10]
        self.answer_context_sizes = sorted({k for k in ctx_values if k > 0})
        self.answer_llm_client = initialize_answer_llm(
            answer_llm_model or os.getenv('LONGMEM_ANSWER_LLM_MODEL'),
            answer_llm_api_key or os.getenv('LONGMEM_ANSWER_LLM_API_KEY'),
            answer_llm_base_url or os.getenv('LONGMEM_ANSWER_LLM_BASE_URL'),
        )
        self.disable_judge = disable_judge
        self.judge_llm_client = None
        if not disable_judge:
            self.judge_llm_client = initialize_answer_llm(
                judge_llm_model or os.getenv('LONGMEM_JUDGE_LLM_MODEL'),
                judge_llm_api_key or os.getenv('LONGMEM_JUDGE_LLM_API_KEY'),
                judge_llm_base_url or os.getenv('LONGMEM_JUDGE_LLM_BASE_URL'),
            )

        self._romem_root = Path(
            romem_save_dir
            or os.getenv('TEMPUS_SAVE_DIR')
            or 'outputs/romem_longmem'
        )
        self._romem_llm_model = romem_llm_model or os.getenv('TEMPUS_LLM_MODEL')
        self._romem_embedding_model = romem_embedding_model or os.getenv('TEMPUS_EMBED_MODEL')
        self._romem_openie_mode = romem_openie_mode or os.getenv('TEMPUS_OPENIE_MODE')
        self._romem_config = romem_config or {}
        self._romem_temporal_awareness = parse_bool(
            romem_temporal_awareness or os.getenv('TEMPUS_TEMPORAL_AWARENESS'),
            default=True,
        )
        enable_tkge_value = romem_enable_tkge_tunnel
        if enable_tkge_value is None:
            enable_tkge_value = os.getenv('TEMPUS_ENABLE_TKGE_TUNNEL')
        self._romem_enable_tkge_tunnel = parse_optional_bool(enable_tkge_value)
        self._romem_tkge_verbose = romem_tkge_verbose

        self.romem = None
        self._answer_session_ids: set[str] = set()
        self._answer_turn_ids: set[str] = set()
        self._doc_to_session: dict[str, list[str]] = {}
        self._doc_to_turn: dict[str, list[str]] = {}

    def _init_romem(self, question_id: str):
        from romem import RoMem
        from romem.utils.config_utils import BaseConfig

        run_dir = self._romem_root / _safe_dir_name(question_id)
        config = BaseConfig()
        apply_romem_config(config, self._romem_config)
        config.save_dir = str(run_dir)
        if self._romem_llm_model:
            config.llm_name = self._romem_llm_model
        if self._romem_embedding_model:
            config.embedding_model_name = self._romem_embedding_model
        if self._romem_openie_mode:
            config.openie_mode = self._romem_openie_mode
        config.temporal_awareness = self._romem_temporal_awareness
        if self._romem_enable_tkge_tunnel is not None:
            config.enable_tkge_tunnel = self._romem_enable_tkge_tunnel
        if self._romem_tkge_verbose is not None:
            config.tkge_verbose = int(self._romem_tkge_verbose)
        config.retrieval_top_k = self.search_limit
        return RoMem(
            global_config=config,
            save_dir=config.save_dir,
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
        )

    async def clear_state(self, question_id: str) -> None:
        self.romem = self._init_romem(question_id)
        self._answer_session_ids = set()
        self._answer_turn_ids = set()
        self._doc_to_session = {}
        self._doc_to_turn = {}

    async def ingest_example(self, example: LongmemevalExample) -> None:
        await self.clear_state(example.question_id)
        docs: list[str] = []
        turn_counts: dict[str, int] = {}
        for episode in example.episodes:
            docs.append(episode.content)
            session_id = episode.metadata.get('session_id')
            if session_id:
                self._doc_to_session.setdefault(episode.content, []).append(session_id)
                turn_idx = turn_counts.get(session_id, 0) + 1
                turn_counts[session_id] = turn_idx
                turn_id = f'{session_id}_{turn_idx}'
                self._doc_to_turn.setdefault(episode.content, []).append(turn_id)
                if episode.metadata.get('has_answer'):
                    self._answer_turn_ids.add(turn_id)
            if episode.metadata.get('is_answer_session') or episode.metadata.get('has_answer'):
                if session_id:
                    self._answer_session_ids.add(session_id)
        if docs:
            self.romem.index(docs=docs)

    async def evaluate_example(self, example: LongmemevalExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, details = await self._retrieve(example)
        llm_metrics, answers = await self._answer(example, context)
        judge_metrics = await self._judge_answers(example, answers)
        return (
            retrieval_metrics,
            {**llm_metrics, **judge_metrics},
            {**details, 'llm_answers': answers, 'context_docs': context},
        )

    async def _retrieve(self, example: LongmemevalExample) -> tuple[dict, list, dict]:
        try:
            retrieval = self.romem.retrieve(
                queries=[example.question],
                num_to_retrieve=self.search_limit,
            )
        except Exception as exc:
            logger.warning('RoMem search failed for %s: %s', example.question_id, exc)
            retrieval = []
        docs = []
        if retrieval:
            docs = retrieval[0].docs
        ranked_sessions: list[str] = []
        ranked_turns: list[str] = []
        seen_sessions: set[str] = set()
        snippet = ''
        session_cursor: dict[str, int] = {}
        turn_cursor: dict[str, int] = {}
        for doc in docs:
            session_ids = self._doc_to_session.get(doc) or []
            turn_ids = self._doc_to_turn.get(doc) or []
            if session_ids:
                cursor = session_cursor.get(doc, 0)
                session_id = session_ids[min(cursor, len(session_ids) - 1)]
                session_cursor[doc] = cursor + 1
            else:
                session_id = None
            if turn_ids:
                cursor = turn_cursor.get(doc, 0)
                turn_id = turn_ids[min(cursor, len(turn_ids) - 1)]
                turn_cursor[doc] = cursor + 1
            else:
                turn_id = None

            if turn_id:
                ranked_turns.append(turn_id)
                if turn_id in self._answer_turn_ids and not snippet:
                    snippet = doc
            if session_id and session_id not in seen_sessions:
                seen_sessions.add(session_id)
                ranked_sessions.append(session_id)
                if session_id in self._answer_session_ids and not snippet:
                    snippet = doc
        metrics: dict[str, float] = {}
        if not ranked_turns or not self._answer_turn_ids:
            for k in (5, 10, 15):
                metrics[f'turn_recall_all@{k}'] = 0.0
                metrics[f'turn_ndcg_any@{k}'] = 0.0
            return metrics, docs, {'matching_fact': snippet}
        turn_corpus_ids = ranked_turns
        turn_rankings = list(range(len(turn_corpus_ids)))
        turn_correct_docs = list(self._answer_turn_ids)
        for k in (5, 10, 15):
            _recall_any, recall_all, ndcg_any = evaluate_retrieval(
                turn_rankings, turn_correct_docs, turn_corpus_ids, k=k
            )
            metrics[f'turn_recall_all@{k}'] = float(recall_all)
            metrics[f'turn_ndcg_any@{k}'] = float(ndcg_any)
        return metrics, docs, {'matching_fact': snippet}

    async def _answer(self, example: LongmemevalExample, context_docs: Iterable[str]) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client
        if client is None:
            return metrics, answers
        docs_list = list(context_docs)
        for k in self.answer_context_sizes:
            subset = docs_list[:k]
            if not subset:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            facts = [f'- {doc}' for doc in subset if doc]
            if not facts:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            prompt = f"""Question: {example.question}

Supporting facts:
{os.linesep.join(facts)}

Provide a concise answer grounded in the facts. Respond with JSON {{"answer": "<text>"}}."""
            messages = [
                Message(role='system', content='You answer questions using only provided facts.'),
                Message(role='user', content=prompt),
            ]
            prediction = ''
            try:
                response = await client.generate_response(messages, response_model=QAResponse)
                prediction = response.get('answer', '')
            except Exception as exc:
                logger.warning('LLM answer failed for %s (top-%s): %s', example.question_id, k, exc)
            answers[k] = prediction
            metrics[f'llm@{k}_context_size'] = float(len(facts))
        return metrics, answers

    async def _judge_answers(self, example: LongmemevalExample, answers: dict[int, str]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for k, prediction in answers.items():
            verdict = await self._judge_single_answer(example, prediction)
            metrics[f'llm@{k}_accuracy'] = 1.0 if verdict else 0.0
        return metrics

    async def _judge_single_answer(self, example: LongmemevalExample, prediction: str) -> bool:
        if not prediction:
            return False
        if self.disable_judge or self.judge_llm_client is None:
            return self._f1(prediction, example.answer) >= 0.7
        prompt = self._build_judge_prompt(example, prediction)
        if not prompt:
            return self._f1(prediction, example.answer) >= 0.7
        messages = [
            Message(
                role='system',
                content='You are a strict evaluator. Reply only with JSON {"label": "yes"} or {"label": "no"}.',
            ),
            Message(role='user', content=prompt),
        ]
        try:
            response = await self.judge_llm_client.generate_response(
                messages,
                response_model=JudgeResponse,
                max_tokens=32,
            )
            verdict = response.get('label', '').strip().lower()
            return verdict.startswith('y')
        except Exception as exc:
            logger.warning('Judge evaluation failed for %s: %s', example.question_id, exc)
            return self._f1(prediction, example.answer) >= 0.7

    def _f1(self, prediction: str, answer: str) -> float:
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(answer)
        if not pred_tokens or not ref_tokens:
            return 0.0
        ref_counts = {}
        for tok in ref_tokens:
            ref_counts[tok] = ref_counts.get(tok, 0) + 1
        matching = 0
        for tok in pred_tokens:
            if ref_counts.get(tok, 0) > 0:
                matching += 1
                ref_counts[tok] -= 1
        precision = matching / len(pred_tokens)
        recall = matching / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _build_judge_prompt(self, example: LongmemevalExample, prediction: str) -> str:
        question = example.question
        answer = example.answer
        qtype = example.question_type.replace('-', '_').lower()
        abstention = '_abs' in example.question_id.lower()
        if abstention:
            template = (
                "I will give you an unanswerable question, an explanation, and a response from a model. "
                "Please answer yes if the model correctly states that the question cannot be answered. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
                "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
            )
            return template.format(question, answer, prediction)

        if 'temporal' in qtype:
            intro = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer. Ignore off-by-one errors in dates or counts."
            )
        elif 'knowledge_update' in qtype:
            intro = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response includes the updated correct answer even if previous information is mentioned."
            )
        elif 'preference' in qtype:
            intro = (
                "I will give you a question, a rubric describing the desired personalized response, and a response from a model. "
                "Answer yes if the response satisfies the rubric and uses the user's personal information correctly."
            )
        elif 'multi' in qtype or 'two_hop' in qtype:
            intro = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer or clearly provides the reasoning steps that lead to it."
            )
        else:
            intro = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Answer yes if the response contains the correct answer. Otherwise, answer no."
            )

        body = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        if 'preference' in qtype:
            body = "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        return intro + body.format(question, answer, prediction)
