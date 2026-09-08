"""Automatic M2→M1 research cycle driven by unreviewed knowledge cards."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from research_fellow.application.advising import (
    auto_rq_priority_prompt,
    dispatch_top_research_questions,
    parse_research_question_suggestions,
    parse_rq_priority_assessment,
    recent_knowledge_updates,
    recent_research_questions,
    store_research_question_candidates,
)
from research_fellow.application.auto_literature import execute_auto_literature_review
from research_fellow.application.prompt_tasks import research_question_suggestions_prompt
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]


def execute_auto_research_cycle(
    ledger: Ledger,
    cards: list[dict[str, Any]],
    cache_dir: Path,
    *,
    rq_drafter: Draft,
    priority_drafter: Draft,
    keyword_drafter: Draft,
    abstract_reviewer: Draft,
    fulltext_drafter: Draft,
    synthesis_drafter: Draft,
    max_suggestions: int = 8,
    max_select: int = 3,
) -> dict[str, Any]:
    """Consume unreviewed M1 cards, update RQs, select Top 3, and run M1 literature review.

    A review batch is completed only after RQ candidates have been parsed and linked to
    their source cards. If candidate generation fails, the cards remain unreviewed and
    can be retried later.
    """
    updates = recent_knowledge_updates(ledger, limit=500)
    if not updates:
        return {"status": "no_new_information", "source_card_count": 0, "review_id": "", "questions": [], "dispatched": [], "executions": []}

    valid_card_ids = {
        str((item.get("payload") or {}).get("card_id", "")) for item in updates
        if str((item.get("payload") or {}).get("card_id", ""))
    }
    prompt = research_question_suggestions_prompt(
        updates, recent_research_questions(ledger), cards, max_suggestions=max_suggestions,
    )
    raw = rq_drafter(prompt) or ""
    candidates = parse_research_question_suggestions(raw, valid_card_ids=valid_card_ids, limit=max_suggestions)
    if not candidates:
        return {
            "status": "rq_generation_failed", "source_card_count": len(valid_card_ids),
            "review_id": "", "questions": [], "dispatched": [], "executions": [],
        }

    review_id = ledger.create_research_state_review("auto", updates)
    try:
        saved = store_research_question_candidates(ledger, candidates, updates, review_id=review_id)
        actionable = [item for item in saved if item.get("status") not in {"hold", "rejected"}]
        ranked: list[dict[str, Any]] = []
        if actionable:
            priority_raw = priority_drafter(auto_rq_priority_prompt(actionable, max_select=max_select)) or ""
            ranked = parse_rq_priority_assessment(priority_raw, actionable, limit=max_select)
        dispatched = dispatch_top_research_questions(ledger, ranked, limit=max_select, review_id=review_id) if ranked else []

        executions: list[dict[str, Any]] = []
        for item in dispatched:
            if item.get("reused"):
                executions.append({"intent_id": item["intent"].intent_id, "status": "reused", "question": item["question"]})
                continue
            event = ledger.phenomenon(item["phenomenon_id"])
            if not event or event.get("status") != "ready":
                executions.append({"intent_id": item["intent"].intent_id, "status": "not_run", "question": item["question"]})
                continue
            result = execute_auto_literature_review(
                ledger, event, cache_dir,
                keyword_drafter=keyword_drafter,
                abstract_reviewer=abstract_reviewer,
                fulltext_drafter=fulltext_drafter,
                synthesis_drafter=synthesis_drafter,
            )
            executions.append({"intent_id": item["intent"].intent_id, "status": result.get("status", "failed"), "question": item["question"]})

        selected_count = len(dispatched)
        new_count = sum(1 for item in saved if item.get("_change_kind") == "new")
        strengthened_count = sum(1 for item in saved if item.get("_change_kind") == "strengthened")
        summary = f"새 지식카드 {len(valid_card_ids)}건에서 RQ 신규 {new_count}건, 보강 {strengthened_count}건을 도출하고 중요 RQ {selected_count}건을 선택했습니다."
        ledger.complete_research_state_review(
            review_id, generated_rq_count=len(saved), selected_rq_count=selected_count, summary=summary,
        )
        case_id = ledger.create_case("research", "M2 자동 연구 사이클")
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "auto_research_cycle",
            {
                "title": "M2 자동 연구 사이클 보고",
                "report": summary,
                "review_id": review_id,
                "source_card_count": len(valid_card_ids),
                "generated_rq_count": len(saved),
                "new_rq_count": new_count,
                "strengthened_rq_count": strengthened_count,
                "selected_rq_count": selected_count,
                "selected_rqs": [
                    {
                        "rq_id": item["rq_id"], "question": item["question"], "score": item["score"],
                        "reason": item["selection_reason"], "intent_id": item["intent"].intent_id,
                        "reused": bool(item.get("reused")),
                    }
                    for item in dispatched
                ],
                "executions": executions,
            },
            subject_id=review_id, status="completed",
        )
        return {
            "status": "completed", "review_id": review_id, "source_card_count": len(valid_card_ids),
            "questions": saved, "ranked": ranked, "dispatched": dispatched, "executions": executions,
            "new_rq_count": new_count, "strengthened_rq_count": strengthened_count,
        }
    except Exception as error:
        ledger.fail_research_state_review(review_id, str(error))
        raise
