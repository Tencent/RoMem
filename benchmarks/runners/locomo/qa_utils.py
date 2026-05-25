"""
Shared utilities for LoCoMo QA prompting and category handling.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.types import LocomoQA

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

MC_PROMPT = """
Based on the above context, choose the best answer from the options below. Respond with the exact choice text.

Question: {}
Choices:
{}
Answer:
"""


@dataclass
class Cat5AnswerKey:
    a: str
    b: str


def format_question(qa: LocomoQA) -> tuple[str, Cat5AnswerKey | None]:
    question = qa.question
    answer_key = None
    if qa.category == 2:
        question = f'{question} Use DATE of CONVERSATION to answer with an approximate date.'
    if qa.category == 5:
        answer_key = Cat5AnswerKey(
            a='Not mentioned in the conversation',
            b=qa.answer,
        )
        question = (
            f'{qa.question} Select the correct answer: (a) {answer_key.a} (b) {answer_key.b}.'
        )
    return question, answer_key


def build_prompt(context: str, qa: LocomoQA) -> tuple[str, Cat5AnswerKey | None]:
    question, answer_key = format_question(qa)
    prompt_template = QA_PROMPT_CAT_5 if qa.category == 5 else QA_PROMPT
    prompt = f'{context}\n\n{prompt_template.format(question)}'
    return prompt, answer_key


def normalize_cat5_answer(model_prediction: str, answer_key: Cat5AnswerKey | None) -> str:
    if answer_key is None:
        return model_prediction
    prediction = (model_prediction or '').strip().lower()
    if len(prediction) == 1:
        return answer_key.a if 'a' in prediction else answer_key.b
    if len(prediction) == 3:
        return answer_key.a if '(a)' in prediction else answer_key.b
    return model_prediction


def build_mc_prompt(context: str, question: str, choices: list[str]) -> str:
    formatted_choices = '\n'.join([f'{idx}. {choice}' for idx, choice in enumerate(choices)])
    return f'{context}\n\n{MC_PROMPT.format(question, formatted_choices)}'


def _normalize_choice(text: str) -> str:
    return ''.join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def match_choice_index(answer_text: str, choices: list[str]) -> int:
    if not answer_text:
        return -1
    cleaned = answer_text.strip()
    if cleaned.isdigit():
        idx = int(cleaned)
        if 0 <= idx < len(choices):
            return idx
        if 1 <= idx <= len(choices):
            return idx - 1
    if len(cleaned) == 1 and cleaned.lower() in 'abcdefghij':
        idx = ord(cleaned.lower()) - ord('a')
        if 0 <= idx < len(choices):
            return idx
    normalized_answer = _normalize_choice(cleaned)
    for idx, choice in enumerate(choices):
        if normalized_answer == _normalize_choice(choice):
            return idx
    for idx, choice in enumerate(choices):
        normalized_choice = _normalize_choice(choice)
        if normalized_answer and (normalized_answer in normalized_choice or normalized_choice in normalized_answer):
            return idx
    return -1
