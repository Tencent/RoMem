"""Async LLM judge for MemGPT-style answer accuracy scoring.

Implements a two-stage evaluation:
  1. Substring match (fast, deterministic)
  2. LLM judge fallback via OpenAI-compatible API
"""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-5.2"

JUDGE_PROMPT = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label"."""


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').lower()).strip()


_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            api_key=os.getenv('LLM_JUDGE_API_KEY') or os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('LLM_JUDGE_BASE_URL') or 'https://api.openai.com/v1',
        )
    return _client


async def score_answer(
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> tuple[bool, str]:
    """Score a generated answer against a gold answer.

    Returns (is_correct, method) where method is 'substring' or 'llm_judge'.
    """
    norm_gold = _normalize(gold_answer)
    norm_gen = _normalize(generated_answer)

    if not norm_gen:
        return False, 'substring'

    # Stage 1: substring match
    if norm_gold and norm_gold in norm_gen:
        return True, 'substring'

    # Stage 2: LLM judge
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=os.getenv('LLM_JUDGE_MODEL', JUDGE_MODEL),
            messages=[
                {
                    'role': 'user',
                    'content': JUDGE_PROMPT.format(
                        question=question,
                        gold_answer=gold_answer,
                        generated_answer=generated_answer,
                    ),
                }
            ],
            response_format={'type': 'json_object'},
            temperature=0.0,
        )
        content = response.choices[0].message.content or '{}'
        label = json.loads(content).get('label', 'WRONG')
        return label == 'CORRECT', 'llm_judge'
    except Exception as exc:
        logger.warning('LLM judge call failed: %s', exc)
        return False, 'llm_judge'
