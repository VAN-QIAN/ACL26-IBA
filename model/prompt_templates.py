"""Utilities for constructing dataset-aware answer prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_INFOSSEEK_SYSTEM_PROMPT = (
    "You always answer the question the user asks. Do not answer anything else."
)

_INFOSSEEK_ONE_SHOT = (
    "Context: The sounthern side of the Alps is next to Lake Como.\n\n"
    "Question: Which body of water is this mountain located in or next to? "
    "Just answer the questions , no explanations needed. Short answer is: Lake Como"
)

_INFOSSEEK_USER_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question} Just answer the questions , no explanations needed. "
    "Short answer is:"
)

_DEFAULT_INSTRUCTION_PREFIX = (
    "You are a knowledgeable assistant. Examine the image carefully before using any contextual evidence."
)

_DEFAULT_CONTEXT_GUIDANCE = (
    "Use ONLY the evidence text if it is helpful."
)

_DEFAULT_NO_CONTEXT_GUIDANCE = (
    "If the image alone is insufficient to answer, reply 'Not sure'."
)

_DEFAULT_REASONING_GUIDANCE = (
    "Briefly explain your reasoning before giving the final answer."
)


@dataclass(frozen=True)
class PromptParts:
    """Structured chat prompt components."""

    system: Optional[str]
    user: str


def _normalize_context_text(context: Optional[str]) -> str:
    text = (context or "").strip()
    return text if text else "None"


def build_prompt_parts(
    *,
    dataset_name: Optional[str],
    question: str,
    context_text: Optional[str],
    require_reasoning: bool,
) -> PromptParts:
    """Return dataset-specific prompt pieces for answer generation."""
    dataset = (dataset_name or "").strip().lower()
    normalized_context = _normalize_context_text(context_text)
    stripped_question = question.strip()

    if dataset == "infoseek":
        user_prompt = "\n\n".join(
            (
                _INFOSSEEK_ONE_SHOT,
                _INFOSSEEK_USER_TEMPLATE.format(
                    context=normalized_context,
                    question=stripped_question,
                ),
            )
        )
        return PromptParts(system=_INFOSSEEK_SYSTEM_PROMPT, user=user_prompt)

    instructions = [_DEFAULT_INSTRUCTION_PREFIX]
    if context_text:
        instructions.append(_DEFAULT_CONTEXT_GUIDANCE)
    else:
        instructions.append(_DEFAULT_NO_CONTEXT_GUIDANCE)
    if require_reasoning:
        instructions.append(_DEFAULT_REASONING_GUIDANCE)

    instruction_block = "\n".join(instructions)
    if context_text:
        user_prompt = (
            f"{instruction_block}\n"
            "---- Evidence ----\n"
            f"{context_text}\n"
            "------------------\n"
            f"Question: {stripped_question}\n"
            "Answer:"
        )
    else:
        user_prompt = (
            f"{instruction_block}\n"
            f"Question: {stripped_question}\n"
            "Answer:"
        )
    return PromptParts(system=None, user=user_prompt)


__all__ = ["PromptParts", "build_prompt_parts"]
