"""Application services for human request-input interactions."""
from __future__ import annotations
from typing import Any, Mapping


def create_researcher_question_thread(ledger: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    context = str(payload.get("context") or "").strip()
    if not question:
        raise ValueError("researcher question is required")
    return ledger.create_research_question_thread(
        question=question,
        source_type="researcher",
        rationale="연구자가 직접 제기한 연구질문입니다.",
        research_context=context,
        source_payload={"researcher_comment": context},
        status="interested",
    )


def create_external_advisory_thread(
    ledger: Any,
    payload: Mapping[str, Any],
    *,
    interpreted_question: str,
    interpretation: str = "",
) -> dict[str, Any]:
    request = str(payload.get("request") or "").strip()
    requester = str(payload.get("requester") or "").strip()
    context = str(payload.get("context") or "").strip()
    question = str(interpreted_question or "").strip()
    if not request:
        raise ValueError("external advisory request is required")
    if not question:
        raise ValueError("interpreted advisory question is required")
    return ledger.create_research_question_thread(
        question=question,
        source_type="external_advisory",
        rationale="외부 자문 요청을 M2 전문성에 비추어 해석한 질문입니다.",
        research_context=context,
        source_payload={
            "requester": requester,
            "original_request": request,
            "request_context": context,
            "interpretation": str(interpretation or ""),
        },
        status="interested",
    )
