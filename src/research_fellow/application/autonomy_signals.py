"""Deterministic domain signals used by Autonomy Policy evaluation.

These helpers do not decide whether a human is required.  They only translate
current domain state into explicit signals consumed by ``autonomy/policies``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def investigation_question_signals(reading_questions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    questions = [dict(item) for item in reading_questions]
    ids = [str(item.get("question_id") or "").strip() for item in questions]
    ids = [value for value in ids if value]
    if not questions:
        case = "none"
    elif len(questions) == 1 and len(ids) == 1:
        case = "single"
    else:
        case = "multiple"
    return {
        "selection_case": case,
        "selected_question_ids": ids if case == "single" else [],
    }


def revision_application_signals(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = [dict(item) for item in candidates]
    if len(items) != 1:
        return {}
    candidate = items[0]
    todo = dict(candidate.get("todo") or {})
    priority = str(todo.get("priority") or "P1")
    impact = {"P2": "low", "P1": "medium", "P0": "high"}.get(priority, "high")

    # Manuscript changes are versioned and therefore technically recoverable,
    # but only low-impact revisions are treated as easy to reverse in policy.
    reversibility = "easy" if impact == "low" else "moderate"

    payload = dict(candidate.get("proposal_payload") or {})
    proposal = dict(payload.get("proposal") or {})
    verdict = str(proposal.get("verdict") or "")
    evidence_ids = [
        *list(payload.get("selected_paper_ids") or []),
        *list(payload.get("selected_card_ids") or []),
    ]
    evidence_sufficiency = "sufficient" if verdict == "resolved" and bool(evidence_ids) else "partial"
    return {
        "impact": impact,
        "reversibility": reversibility,
        "evidence_sufficiency": evidence_sufficiency,
    }
