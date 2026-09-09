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
from research_fellow.application.llm_retry import LLMRetryExhausted, call_with_retry, failure_payload
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
    resume_run_id: str = "",
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
    cycle_run_id = resume_run_id or ledger.create_auto_research_run(stage="rq_generation")
    try:
        manual_rq = ledger.manual_recovery_override(cycle_run_id, stage="rq_generation")
        if manual_rq:
            raw, attempts = manual_rq, 1
            if not parse_research_question_suggestions(raw, valid_card_ids=valid_card_ids, limit=max_suggestions):
                raise ValueError("수동 복구 연구질문 응답을 파싱하지 못했습니다.")
            ledger.clear_manual_recovery_override(cycle_run_id, stage="rq_generation")
        else:
            raw, attempts = call_with_retry(
                rq_drafter, prompt, stage="rq_generation",
                validator=lambda text: bool(parse_research_question_suggestions(text, valid_card_ids=valid_card_ids, limit=max_suggestions)),
            )
        ledger.update_auto_research_run(cycle_run_id, stage="rq_generation", retry_count=attempts - 1)
        candidates = parse_research_question_suggestions(raw, valid_card_ids=valid_card_ids, limit=max_suggestions)
    except LLMRetryExhausted as error:
        ledger.update_auto_research_run(
            cycle_run_id, status="needs_attention", stage=error.stage,
            error_type=error.error_type, error_message=error.message, retry_count=error.attempts,
        )
        ledger.record_auto_research_failure(
            failure_payload(error, context={"kind": "auto_cycle", "source_card_count": len(valid_card_ids), "source_card_ids": sorted(valid_card_ids)}),
            run_id=cycle_run_id,
        )
        return {
            "status": "rq_generation_failed", "source_card_count": len(valid_card_ids),
            "review_id": "", "questions": [], "dispatched": [], "executions": [],
            "retry_run_id": cycle_run_id, "retry_error": error.error_type,
        }

    review_id = ledger.create_research_state_review("auto", updates)
    with ledger.connect() as conn:
        conn.execute("UPDATE auto_research_runs SET review_id=?, updated_at=? WHERE run_id=?", (review_id, __import__("research_fellow.storage", fromlist=["now"]).now(), cycle_run_id))
    try:
        saved = store_research_question_candidates(ledger, candidates, updates, review_id=review_id)
        actionable = [item for item in saved if item.get("status") not in {"hold", "rejected"}]
        ranked: list[dict[str, Any]] = []
        if actionable:
            ledger.update_auto_research_run(cycle_run_id, stage="rq_prioritization")
            priority_prompt = auto_rq_priority_prompt(actionable, max_select=max_select)
            manual_priority = ledger.manual_recovery_override(cycle_run_id, stage="rq_prioritization")
            if manual_priority:
                priority_raw, attempts = manual_priority, 1
                if not parse_rq_priority_assessment(priority_raw, actionable, limit=max_select):
                    raise ValueError("수동 복구 중요도 평가 응답을 파싱하지 못했습니다.")
                ledger.clear_manual_recovery_override(cycle_run_id, stage="rq_prioritization")
            else:
                priority_raw, attempts = call_with_retry(
                    priority_drafter, priority_prompt, stage="rq_prioritization",
                    validator=lambda text: bool(parse_rq_priority_assessment(text, actionable, limit=max_select)),
                )
            ranked = parse_rq_priority_assessment(priority_raw, actionable, limit=max_select)
            ledger.update_auto_research_run(cycle_run_id, retry_count=attempts - 1)
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
                parent_run_id=cycle_run_id,
            )
            executions.append({"intent_id": item["intent"].intent_id, "status": result.get("status", "failed"), "question": item["question"]})

        selected_count = len(dispatched)
        new_count = sum(1 for item in saved if item.get("_change_kind") == "new")
        strengthened_count = sum(1 for item in saved if item.get("_change_kind") == "strengthened")
        summary = f"새 지식카드 {len(valid_card_ids)}건에서 RQ 신규 {new_count}건, 보강 {strengthened_count}건을 도출하고 중요 RQ {selected_count}건을 선택했습니다."
        attention_count = sum(1 for item in executions if item.get("status") in {"needs_attention", "failed"})
        if attention_count:
            summary += f" 후속 M1 문헌탐색 {attention_count}건은 중간 실패로 재시도가 필요합니다. M2의 RQ 생성·보강 결과는 보존됩니다."
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
            subject_id=review_id, status="failed" if attention_count else "completed",
        )
        cycle_status = "needs_attention" if attention_count else "completed"
        ledger.update_auto_research_run(
            cycle_run_id, status=cycle_status,
            stage="m1_followup_attention" if attention_count else "completed",
            error_type="m1_followup_failure" if attention_count else None,
            error_message=f"후속 M1 문헌탐색 {attention_count}건 재시도 필요" if attention_count else None,
        )
        return {
            "status": cycle_status, "review_id": review_id, "source_card_count": len(valid_card_ids),
            "questions": saved, "ranked": ranked, "dispatched": dispatched, "executions": executions,
            "new_rq_count": new_count, "strengthened_rq_count": strengthened_count,
        }
    except LLMRetryExhausted as error:
        ledger.fail_research_state_review(review_id, str(error))
        ledger.update_auto_research_run(
            cycle_run_id, status="needs_attention", stage=error.stage,
            error_type=error.error_type, error_message=error.message, retry_count=error.attempts,
        )
        ledger.record_auto_research_failure(
            failure_payload(error, context={"kind": "auto_cycle", "review_id": review_id, "actionable_questions": actionable if 'actionable' in locals() else []}),
            run_id=cycle_run_id, review_id=review_id,
        )
        return {
            "status": "needs_attention", "review_id": review_id, "source_card_count": len(valid_card_ids),
            "questions": [], "dispatched": [], "executions": [], "retry_run_id": cycle_run_id,
            "retry_error": error.error_type,
        }
    except Exception as error:
        ledger.fail_research_state_review(review_id, str(error))
        ledger.update_auto_research_run(cycle_run_id, status="needs_attention", stage="unexpected_error", error_type="unexpected_error", error_message=str(error))
        raise
