"""M2 research-state and knowledge-update read models."""
from __future__ import annotations
from typing import Any
from research_fellow.domain.research import ResearchState
from research_fellow.storage import Ledger

def latest_research_state(ledger: Ledger) -> ResearchState | None:
    """Read the latest researcher-visible state from the shared-phenomena ledger."""
    for item in ledger.phenomena(type_="research_update"):
        state = item["payload"].get("state")
        if state:
            try:
                return ResearchState.model_validate(state)
            except ValueError:
                continue
    return None


def recent_research_questions(ledger: Ledger, limit: int = 8) -> list[str]:
    """Researcher-owned question history, newest first and without duplicates."""
    questions: list[str] = []
    for item in ledger.phenomena(type_="research_update"):
        state = item["payload"].get("state", {})
        question = state.get("question") if isinstance(state, dict) else None
        if isinstance(question, str) and question.strip() and question not in questions:
            questions.append(question)
        if len(questions) >= limit:
            break
    return questions


def recent_knowledge_updates(ledger: Ledger, limit: int = 100) -> list[dict[str, Any]]:
    """Unreviewed M1 knowledge-card updates that have not yet been consumed by an M2 state review.

    Multiple pending events for the same card are collapsed into one inbox item while
    retaining every pending update id so a completed review can consume them together.
    """
    reviewed = ledger.reviewed_knowledge_update_ids()
    pending_by_card: dict[str, dict[str, Any]] = {}
    pending_ids_by_card: dict[str, list[str]] = {}
    for item in ledger.phenomena(recipient="m2", type_="knowledge_update"):
        if item.get("phenomenon_id") in reviewed:
            continue
        payload = item.get("payload") or {}
        card_id = str(payload.get("card_id", "")).strip()
        if not card_id or item.get("subject_type") != "knowledge_card" or payload.get("operation") == "deleted":
            continue
        pending_ids_by_card.setdefault(card_id, []).append(str(item.get("phenomenon_id", "")))
        if card_id not in pending_by_card:
            pending_by_card[card_id] = dict(item)
    result: list[dict[str, Any]] = []
    for card_id, item in pending_by_card.items():
        item["pending_update_ids"] = [value for value in pending_ids_by_card.get(card_id, []) if value]
        result.append(item)
    return result[: max(1, min(int(limit), 500))]
