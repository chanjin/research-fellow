"""Common execution helper for one logical LLM-backed application stage.

Execution mode is stage policy.  Complex research stages may deliberately stop
before any local-model call and request an external manual LLM result.  The same
parser/acceptance rule validates both local and pasted results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from research_fellow.application.llm_execution_policy import EXTERNAL_MANUAL, llm_execution_mode
from research_fellow.application.llm_retry import LLMRetryExhausted, call_with_retry

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
    item_key: str = "",
    invalid_manual_message: str = "수동 외부 LLM 응답을 파싱하지 못했습니다.",
    max_attempts: int = 3,
) -> LLMStageResult[T]:
    """Execute one LLM stage according to the stage-level execution policy."""

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

    if llm_execution_mode(stage) == EXTERNAL_MANUAL:
        raise LLMRetryExhausted(
            stage=stage,
            error_type="external_llm_required",
            message="이 단계는 현재 운영 정책상 로컬 LLM을 자동 호출하지 않습니다.",
            attempts=0,
            recommended_action="아래 프롬프트를 외부 LLM에서 실행하고 전체 응답을 붙여 넣으세요.",
            prompt=prompt,
            item_key=item_key,
        )

    raw, attempts = call_with_retry(
        draft,
        prompt,
        stage=stage,
        validator=parse_and_accept,
        max_attempts=max_attempts,
    )
    return LLMStageResult(raw=raw, value=parser(raw), attempts=attempts, source="llm")
