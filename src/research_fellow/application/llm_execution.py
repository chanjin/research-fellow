"""Common execution helper for one logical LLM-backed application stage.

This module deliberately owns execution mechanics only: automatic retry, optional
manual override validation, parsing and attempt metadata. Domain-specific prompt
construction and parsers remain in their application modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from research_fellow.application.llm_retry import call_with_retry

T = TypeVar("T")
Draft = Callable[[str], str | None]
Parser = Callable[[str], T]
Predicate = Callable[[T], bool]


@dataclass(frozen=True)
class LLMStageResult(Generic[T]):
    raw: str
    value: T
    attempts: int
    source: str  # "llm" | "manual"


def execute_llm_stage(
    draft: Draft,
    prompt: str,
    *,
    stage: str,
    parser: Parser[T],
    accept: Predicate[T] | None = None,
    manual_response: str | None = None,
    invalid_manual_message: str = "수동 복구 응답을 파싱하지 못했습니다.",
    max_attempts: int = 3,
) -> LLMStageResult[T]:
    """Execute one LLM stage while keeping domain parsing outside the runtime.

    The same parser/acceptance rule is applied to both automatic and manually
    supplied responses. This removes repeated retry/parse/manual-override glue
    from individual workflows without turning domain-specific judgments into a
    generic function.
    """

    def parse_and_accept(text: str) -> bool:
        value = parser(text)
        return bool(accept(value)) if accept is not None else True

    if manual_response is not None and str(manual_response).strip():
        raw = str(manual_response)
        try:
            value = parser(raw)
            valid = bool(accept(value)) if accept is not None else True
        except (ValueError, TypeError, KeyError):
            valid = False
            value = None  # type: ignore[assignment]
        if not valid:
            raise ValueError(invalid_manual_message)
        return LLMStageResult(raw=raw, value=value, attempts=1, source="manual")

    raw, attempts = call_with_retry(
        draft,
        prompt,
        stage=stage,
        validator=parse_and_accept,
        max_attempts=max_attempts,
    )
    return LLMStageResult(raw=raw, value=parser(raw), attempts=attempts, source="llm")
