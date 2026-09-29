"""Persistence of M2 research-question candidates and provenance."""
from __future__ import annotations
from typing import Any
from research_fellow.domain.research import ResearchQuestionCandidate
from research_fellow.storage import Ledger

def store_research_question_candidates(
    ledger: Ledger, candidates: list[ResearchQuestionCandidate], updates: list[dict[str, Any]], *, review_id: str | None = None,
) -> list[dict[str, Any]]:
    """Persist candidate questions with cumulative M1 provenance and optional review-batch links."""
    update_by_card: dict[str, str] = {}
    for update in updates:
        payload = update.get("payload", {}) if isinstance(update.get("payload"), dict) else {}
        card_id = str(payload.get("card_id", ""))
        if card_id:
            update_by_card[card_id] = str(update.get("phenomenon_id", ""))
    saved: list[dict[str, Any]] = []
    for candidate in candidates:
        existing = ledger.research_question_by_text(candidate.question)
        payload = candidate.model_dump(mode="json")
        payload["source_update_ids"] = [update_by_card[card_id] for card_id in candidate.source_card_ids if card_id in update_by_card]
        stored = ledger.upsert_research_question(payload)
        stored["_change_kind"] = "strengthened" if existing else "new"
        if review_id:
            ledger.link_research_question_review(review_id, str(stored["rq_id"]), stored["_change_kind"])
            for card_id in candidate.source_card_ids:
                ledger.add_research_question_source(
                    str(stored["rq_id"]), review_id, card_id, update_by_card.get(card_id, ""), candidate.rationale,
                )
            if stored["_change_kind"] == "new":
                summary = f"새 지식카드 {len(candidate.source_card_ids)}건을 근거로 새 연구질문이 생성되었습니다."
                change_type = "created"
            else:
                summary = f"새 지식카드 {len(candidate.source_card_ids)}건이 추가 근거로 연결되어 연구질문이 보강되었습니다."
                change_type = "evidence_added"
            ledger.add_research_question_change(str(stored["rq_id"]), change_type, summary, review_id=review_id)
        saved.append(stored)
    return saved
