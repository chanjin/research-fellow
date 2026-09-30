"""Application services for the two user-facing Research work intake paths.

The module deliberately creates no intake store of its own. Research questions
are committed to the existing durable Research Question thread state. External
advisory input remains an uncommitted UI draft until the existing interpretation
review Interaction is accepted, at which point it is committed through the same
Research Question thread state used by the legacy path.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from research_fellow.application.external_advisory_interaction import (
    apply_external_advisory_interpretation,
)
from research_fellow.application.input_interaction import create_researcher_question_thread
from research_fellow.infrastructure.prompt_renderer import render_prompt

DEFAULT_ADVISORY_EXPERTISE = "에이전트 공학과 도메인 전문 연구위원 설계"


def submit_research_question(ledger: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Commit one researcher-originated question through the existing service."""
    return create_researcher_question_thread(ledger, payload)


def prepare_external_advisory_interpretation(
    payload: Mapping[str, Any],
    *,
    draft_fn: Callable[[str], str | None],
    expertise: str = DEFAULT_ADVISORY_EXPERTISE,
) -> dict[str, Any]:
    """Prepare the existing advisory-interpretation review without persisting a new state.

    The original request is preserved verbatim in the returned Interaction input.
    Persistence only occurs after ``review_external_advisory_interpretation`` is
    submitted, keeping the Research Question thread as the single durable source.
    """
    request = str(payload.get("request") or "").strip()
    if not request:
        raise ValueError("external advisory request is required")
    requester = str(payload.get("requester") or "").strip()
    context = str(payload.get("context") or "").strip()
    prompt = render_prompt(
        "m2_external_interpretation.j2",
        requester=requester or "external requester",
        expertise=str(expertise or DEFAULT_ADVISORY_EXPERTISE),
        question=request,
        context=context,
    )
    interpretation = str(draft_fn(prompt) or request).strip()
    return {
        "external_advisory_request": {
            "requester": requester,
            "request": request,
            "context": context,
        },
        "proposed_interpretation": {
            "question": interpretation,
            "interpretation": interpretation,
        },
    }


def submit_external_advisory_interpretation(
    ledger: Any,
    request: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit an approved advisory interpretation through the existing durable path."""
    return apply_external_advisory_interpretation(ledger, request, review)
