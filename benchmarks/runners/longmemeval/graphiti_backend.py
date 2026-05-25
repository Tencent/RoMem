"""Backend for LongMemEval benchmarking (Graphiti)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel
from tqdm import tqdm

from baselines.graphiti.graphiti_core.graphiti import AddEpisodeResults
from baselines.graphiti.graphiti_core.nodes import EpisodeType
from baselines.graphiti.graphiti_core.prompts import prompt_library
from baselines.graphiti.graphiti_core.prompts.eval import EvalAddEpisodeResults
from baselines.graphiti.graphiti_core.prompts.models import Message
from baselines.graphiti.graphiti_core.llm_client.config import LLMConfig
from baselines.graphiti.graphiti_core.llm_client.openai_client import OpenAIClient
from baselines.graphiti.graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from benchmarks.runners.base_backend import GraphitiBackendBase
from benchmarks.types import LongmemevalExample
from dataset.longmemeval.code.retrieval.eval_utils import evaluate_retrieval

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    answer: str


class JudgeResponse(BaseModel):
    label: str


def _tokenize(value: str) -> list[str]:
    return [tok for tok in (value or '').lower().split() if tok]


class LongmemevalGraphitiBackend(GraphitiBackendBase):
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str | None = None,
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
        enable_official_comparison: bool = False,
        official_baseline_path: str | Path | None = None,
        search_reranker: str | None = None,
    ):
        super().__init__(
            uri,
            user,
            password,
            database=database,
            usage_interval_env='LONGMEM_USAGE_INTERVAL',
            search_reranker=search_reranker,
        )
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
        self.answer_llm_client = self._initialize_answer_llm(
            answer_llm_model or os.getenv('LONGMEM_ANSWER_LLM_MODEL'),
            answer_llm_api_key or os.getenv('LONGMEM_ANSWER_LLM_API_KEY'),
            answer_llm_base_url or os.getenv('LONGMEM_ANSWER_LLM_BASE_URL'),
        )
        self.disable_judge = disable_judge
        self.judge_llm_client = None
        if not disable_judge:
            self.judge_llm_client = self._initialize_answer_llm(
                judge_llm_model or os.getenv('LONGMEM_JUDGE_LLM_MODEL'),
                judge_llm_api_key or os.getenv('LONGMEM_JUDGE_LLM_API_KEY'),
                judge_llm_base_url or os.getenv('LONGMEM_JUDGE_LLM_BASE_URL'),
            )
        self.enable_official_comparison = enable_official_comparison
        self.official_baseline_path = Path(official_baseline_path) if official_baseline_path else None
        self._official_baseline = self._load_official_baseline() if self.enable_official_comparison else {}
        self._current_add_results: list[AddEpisodeResults] = []
        self._current_contexts: list[tuple[str, list[str]]] = []
        self._answer_episode_ids: set[str] = set()
        self._answer_turn_ids: set[str] = set()
        self._episode_to_session: dict[str, str] = {}
        self._episode_to_turn: dict[str, str] = {}

    async def ingest_example(self, example: LongmemevalExample) -> None:
        self._answer_episode_ids = set()
        self._answer_turn_ids = set()
        self._episode_to_session = {}
        self._episode_to_turn = {}
        if self.enable_official_comparison:
            self._current_add_results = []
            self._current_contexts = []
            history: list[str] = []

        turn_counts: dict[str, int] = {}
        progress = tqdm(example.episodes, desc=f'Ingesting sessions ({example.question_id})', leave=False)
        for idx, episode in enumerate(progress):
            result = await self.graphiti.add_episode(
                name=f'{example.question_id}-episode-{idx}',
                episode_body=episode.content,
                source_description=episode.metadata.get('session_id', 'longmemeval'),
                reference_time=episode.reference_time,
                source=EpisodeType.message,
            )
            if result and result.episode and result.episode.uuid:
                session_id = episode.metadata.get('session_id')
                if session_id:
                    self._episode_to_session[result.episode.uuid] = session_id
                    turn_idx = turn_counts.get(session_id, 0) + 1
                    turn_counts[session_id] = turn_idx
                    turn_id = f'{session_id}_{turn_idx}'
                    self._episode_to_turn[result.episode.uuid] = turn_id
                    if episode.metadata.get('has_answer'):
                        self._answer_turn_ids.add(turn_id)
            if episode.metadata.get('is_answer_session') or episode.metadata.get('has_answer'):
                self._answer_episode_ids.add(result.episode.uuid)
            if self.enable_official_comparison:
                self._current_add_results.append(result)
                self._current_contexts.append((episode.content, history.copy()))
                history.append(episode.content)
            self.increment_episode_count()
        progress.close()

    async def clear_graph(self) -> None:
        if self.reset_mode == 'delete':
            await self.graphiti.driver.execute_query('MATCH (n) DETACH DELETE n', params={})
        elif self.reset_mode == 'recreate':
            await self.graphiti.driver.close()
            self._init_graphiti()
        else:
            raise ValueError(f'Unknown reset_mode: {self.reset_mode}')

    async def evaluate_example(self, example: LongmemevalExample) -> tuple[dict, dict, dict]:
        await self.ingest_example(example)
        retrieval_metrics, context, details = await self._retrieve(example)
        llm_metrics, answers = await self._answer(example, context)
        judge_metrics = await self._judge_answers(example, answers)
        official_metrics = await self._maybe_official_compare(example)
        await self.clear_graph()
        return (
            retrieval_metrics,
            {**llm_metrics, **judge_metrics, **(official_metrics or {})},
            {**details, 'llm_answers': answers, 'context_docs': context},
        )

    async def _retrieve(self, example: LongmemevalExample) -> tuple[dict, list, dict]:
        try:
            results = await self._search_edges(example.question, self.search_limit)
        except Exception as exc:
            logger.warning('Search failed for %s: %s', example.question_id, exc)
            results = []
        ranked_turns: list[str] = []
        seen_turns: set[str] = set()
        snippet = ''
        for edge in results:
            fact = (getattr(edge, 'fact', '') or '').strip()
            for ep_uuid in getattr(edge, 'episodes', []) or []:
                turn_id = self._episode_to_turn.get(ep_uuid)
                if turn_id and turn_id not in seen_turns:
                    seen_turns.add(turn_id)
                    ranked_turns.append(turn_id)
                    if turn_id in self._answer_turn_ids and not snippet:
                        snippet = fact

        metrics: dict[str, float] = {}
        if not ranked_turns or not self._answer_turn_ids:
            for k in (5, 10, 15):
                metrics[f'turn_recall_all@{k}'] = 0.0
                metrics[f'turn_ndcg_any@{k}'] = 0.0
            return metrics, results, {'matching_fact': snippet}

        turn_corpus_ids = ranked_turns
        turn_rankings = list(range(len(turn_corpus_ids)))
        turn_correct_docs = list(self._answer_turn_ids)
        for k in (5, 10, 15):
            _recall_any, recall_all, ndcg_any = evaluate_retrieval(
                turn_rankings, turn_correct_docs, turn_corpus_ids, k=k
            )
            metrics[f'turn_recall_all@{k}'] = float(recall_all)
            metrics[f'turn_ndcg_any@{k}'] = float(ndcg_any)
        return metrics, results, {'matching_fact': snippet}

    async def _answer(self, example: LongmemevalExample, context_edges: Iterable) -> tuple[dict, dict[int, str]]:
        metrics: dict[str, float] = {}
        answers: dict[int, str] = {}
        client = self.answer_llm_client or self.graphiti.llm_client
        for k in self.answer_context_sizes:
            subset = list(context_edges)[:k]
            if not subset:
                metrics[f'llm@{k}_context_size'] = 0.0
                continue
            facts = []
            for edge in subset:
                fact = getattr(edge, 'fact', '') or ''
                if fact:
                    facts.append(f'- {fact}')
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

    async def _maybe_official_compare(self, example: LongmemevalExample) -> dict[str, float] | None:
        if not self.enable_official_comparison or not self.official_baseline_path:
            return None
        baseline = self._official_baseline.get(example.question_id)
        if not baseline:
            logger.warning('No official baseline entry for %s', example.question_id)
            return None
        if not self._current_add_results:
            return None
        score_sum = 0.0
        comparisons = 0
        for baseline_result, candidate_result, context in zip(
            baseline,
            self._current_add_results,
            self._current_contexts,
            strict=False,
        ):
            comparisons += 1
            message, previous = context
            payload = {
                'baseline': baseline_result,
                'candidate': candidate_result,
                'message': message,
                'previous_messages': previous,
            }
            response = await self.graphiti.llm_client.generate_response(
                prompt_library.eval.eval_add_episode_results(payload),
                response_model=EvalAddEpisodeResults,
            )
            candidate_is_worse = response.get('candidate_is_worse', False)
            score_sum += 0.0 if candidate_is_worse else 1.0
        if not comparisons:
            return None
        return {'official_agreement': score_sum / comparisons}

    def _initialize_answer_llm(
        self,
        model: str | None,
        api_key: str | None,
        base_url: str | None,
    ):
        if not any([model, api_key, base_url]):
            return None
        config = LLMConfig(model=model, api_key=api_key, base_url=base_url)
        name = (config.model or '').lower()
        if not name or name.startswith('gpt'):
            return OpenAIClient(config=config)
        return OpenAIGenericClient(config=config)

    def _load_official_baseline(self) -> dict[str, list[AddEpisodeResults]]:
        if not self.official_baseline_path or not self.official_baseline_path.exists():
            logger.warning('Official baseline path %s missing; comparison disabled.', self.official_baseline_path)
            self.enable_official_comparison = False
            return {}
        try:
            raw = json.loads(self.official_baseline_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning('Failed to parse official baseline JSON: %s', exc)
            self.enable_official_comparison = False
            return {}
        baseline: dict[str, list[AddEpisodeResults]] = {}
        for key, items in raw.items():
            try:
                baseline[key] = [AddEpisodeResults(**entry) for entry in items]
            except Exception as exc:
                logger.warning('Could not parse baseline entry %s: %s', key, exc)
        return baseline

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
                "Please answer yes if the model correctly states that the question cannot be answered.\n\n"
                "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
                "Does the model correctly identify the question as unanswerable? Answer yes or no only."
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
