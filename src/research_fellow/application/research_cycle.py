"""Automatic M2→M1 research cycle executed from the AJD workflow DSL."""
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
from research_fellow.application.dsl import WorkflowDefinition, prepare_workflow_run, workflow_result
from research_fellow.application.llm_retry import LLMRetryExhausted
from research_fellow.application.llm_execution import execute_llm_stage
from research_fellow.application.prompt_tasks import research_question_suggestions_prompt
from research_fellow.application.research_state_update import execute_research_state_update
from research_fellow.application.run_tracking import ExecutionRunTracker
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]
WORKFLOW_PATH = "m2/research_cycle.yaml"


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
    source_updates: list[dict[str, Any]] | None = None,
    state_update_drafter: Draft | None = None,
) -> dict[str, Any]:
    """Execute the M2 research-cycle YAML workflow.

    Semantics:
      knowledge cards/updates → M2 Research Questions → priority selection →
      M1 Curation Intents → M1 literature review.

    The YAML owns orchestration and traceable AJD semantics. Python handlers retain
    deterministic ledger operations, parsing/validation, and external execution.
    """
    updates = list(source_updates) if source_updates is not None else recent_knowledge_updates(ledger, limit=500)
    if not updates:
        return {
            "status": "no_new_information", "source_card_count": 0, "review_id": "",
            "questions": [], "dispatched": [], "executions": [],
        }

    workflow = None
    valid_card_ids = {
        str((item.get("payload") or {}).get("card_id", "")) for item in updates
        if str((item.get("payload") or {}).get("card_id", ""))
    }
    run_tracker = ExecutionRunTracker.start(ledger, run_id=resume_run_id, stage="rq_generation")
    cycle_run_id = run_tracker.run_id
    workflow = prepare_workflow_run(WORKFLOW_PATH, {
        "ledger": ledger,
        "cards": cards,
        "cache_dir": cache_dir,
        "knowledge_updates": updates,
        "existing_research_questions": recent_research_questions(ledger),
        "research_question_backlog": ledger.research_question_backlog(statuses=["candidate", "interested", "exploring", "hold"], limit=100),
        "valid_card_ids": valid_card_ids,
        "cycle_run_id": cycle_run_id,
        "run_tracker": run_tracker,
        "rq_drafter": rq_drafter,
        "state_update_drafter": state_update_drafter or rq_drafter,
        "priority_drafter": priority_drafter,
        "keyword_drafter": keyword_drafter,
        "abstract_reviewer": abstract_reviewer,
        "fulltext_drafter": fulltext_drafter,
        "synthesis_drafter": synthesis_drafter,
        "max_suggestions": max_suggestions,
        "max_select": max_select,
    }, globals())
    context = workflow.context

    result = workflow.execute(recover=[
        (LLMRetryExhausted, _recover_retry_exhausted),
        (Exception, _recover_unexpected_error),
    ])
    # Preserve the existing public API while the DSL keeps domain-oriented names.
    result["questions"] = result.get("research_questions") or []
    result["ranked"] = result.get("ranked_questions") or []
    result["dispatched"] = result.get("dispatched_intents") or []
    return result


def _m2_research_state_update(context: dict[str, Any]) -> None:
    """Refresh existing RQs from new knowledge before generating or prioritizing follow-up work."""
    try:
        result = execute_research_state_update(
            context["ledger"],
            context["knowledge_updates"],
            context.get("research_question_backlog") or [],
            drafter=context["state_update_drafter"],
        )
    except LLMRetryExhausted:
        # State refresh is advisory to the broader cycle. Preserve the main RQ-generation path
        # when an older/manual drafter cannot satisfy the new structured assessment contract.
        result = {
            "status": "needs_attention", "state_transitions": [], "followup_questions": [],
            "resolved_count": 0, "reinforced_count": 0,
        }
    context["state_update_status"] = result.get("status", "")
    context["state_transitions"] = result.get("state_transitions") or []
    context["followup_questions"] = result.get("followup_questions") or []
    context["resolved_rq_count"] = int(result.get("resolved_count") or 0)
    context["reinforced_rq_count"] = int(result.get("reinforced_count") or 0)


def _formulate_research_questions(context: dict[str, Any]) -> None:
    """Generate, open the review for, and persist M2 research questions as one capability."""
    ledger: Ledger = context["ledger"]
    tracker: ExecutionRunTracker = context["run_tracker"]
    valid_card_ids = context["valid_card_ids"]
    prompt = research_question_suggestions_prompt(
        context["knowledge_updates"],
        context["existing_research_questions"],
        context["cards"],
        max_suggestions=context["max_suggestions"],
    )
    manual_rq = tracker.manual_override(stage="rq_generation")
    result = execute_llm_stage(
        context["rq_drafter"],
        prompt,
        stage="rq_generation",
        parser=lambda text: parse_research_question_suggestions(
            text, valid_card_ids=valid_card_ids, limit=context["max_suggestions"],
        ),
        accept=bool,
        manual_response=manual_rq,
        invalid_manual_message="수동 복구 연구질문 응답을 파싱하지 못했습니다.",
    )
    if result.source == "manual":
        tracker.clear_manual_override(stage="rq_generation")
    tracker.enter("rq_generation", retry_count=result.attempts - 1)

    review_id = ledger.create_research_state_review("auto", context["knowledge_updates"])
    tracker.bind_review(review_id)
    saved = store_research_question_candidates(
        ledger, result.value, context["knowledge_updates"], review_id=review_id,
    )
    context["review_id"] = review_id
    context["rq_candidates"] = result.value
    context["research_questions"] = saved
    generated_actionable = [item for item in saved if item.get("status") not in {"hold", "resolved", "rejected"}]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*(context.get("followup_questions") or []), *generated_actionable]:
        key = str(item.get("rq_id") or item.get("question") or "")
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    context["actionable_questions"] = merged


def _prioritize_research_questions(context: dict[str, Any]) -> None:
    actionable = context["actionable_questions"]
    tracker: ExecutionRunTracker = context["run_tracker"]
    tracker.enter("rq_prioritization")
    prompt = auto_rq_priority_prompt(actionable, max_select=context["max_select"])
    manual_priority = tracker.manual_override(stage="rq_prioritization")
    result = execute_llm_stage(
        context["priority_drafter"],
        prompt,
        stage="rq_prioritization",
        parser=lambda text: parse_rq_priority_assessment(text, actionable, limit=context["max_select"]),
        accept=bool,
        manual_response=manual_priority,
        invalid_manual_message="수동 복구 중요도 평가 응답을 파싱하지 못했습니다.",
    )
    if result.source == "manual":
        tracker.clear_manual_override(stage="rq_prioritization")
    context["ranked_questions"] = result.value
    tracker.note_attempts(result.attempts)


def _delegate_literature_review(context: dict[str, Any]) -> None:
    ranked = context["ranked_questions"]
    context["dispatched_intents"] = dispatch_top_research_questions(
        context["ledger"], ranked, limit=context["max_select"], review_id=context["review_id"],
    ) if ranked else []


def _m1_auto_literature_review(context: dict[str, Any]) -> None:
    """Execute one delegated M1 review; foreach orchestration belongs to the DSL runtime."""
    ledger: Ledger = context["ledger"]
    item = context["current_item"]
    executions: list[dict[str, Any]] = context.get("executions") or []
    context["executions"] = executions
    if item.get("reused"):
        executions.append({"intent_id": item["intent"].intent_id, "status": "reused", "question": item["question"]})
        return
    event = ledger.phenomenon(item["phenomenon_id"])
    if not event or event.get("status") != "ready":
        executions.append({"intent_id": item["intent"].intent_id, "status": "not_run", "question": item["question"]})
        return
    result = execute_auto_literature_review(
        ledger, event, context["cache_dir"],
        keyword_drafter=context["keyword_drafter"],
        abstract_reviewer=context["abstract_reviewer"],
        fulltext_drafter=context["fulltext_drafter"],
        synthesis_drafter=context["synthesis_drafter"],
        parent_run_id=context["cycle_run_id"],
    )
    executions.append({
        "intent_id": item["intent"].intent_id,
        "status": result.get("status", "failed"),
        "question": item["question"],
    })


def _complete_research_state_review(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    saved = context["research_questions"]
    dispatched = context["dispatched_intents"]
    executions = context["executions"]
    valid_card_ids = context["valid_card_ids"]
    review_id = context["review_id"]

    selected_count = len(dispatched)
    new_count = sum(1 for item in saved if item.get("_change_kind") == "new")
    strengthened_count = sum(1 for item in saved if item.get("_change_kind") == "strengthened")
    resolved_rq_count = int(context.get("resolved_rq_count") or 0)
    reinforced_rq_count = int(context.get("reinforced_rq_count") or 0)
    summary = (
        f"새 지식카드 {len(valid_card_ids)}건에서 기존 RQ 종결 {resolved_rq_count}건, 재보강 {reinforced_rq_count}건을 검토하고, "
        f"RQ 신규 {new_count}건, 보강 {strengthened_count}건을 도출하여 중요 RQ {selected_count}건을 선택했습니다."
    )
    attention_count = sum(1 for item in executions if item.get("status") in {"needs_attention", "failed"})
    if attention_count:
        summary += (
            f" 후속 M1 문헌탐색 {attention_count}건은 중간 실패로 재시도가 필요합니다. "
            "M2의 RQ 생성·보강 결과는 보존됩니다."
        )
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
    tracker: ExecutionRunTracker = context["run_tracker"]
    if attention_count:
        tracker.needs_attention(
            stage="m1_followup_attention",
            error_type="m1_followup_failure",
            error_message=f"후속 M1 문헌탐색 {attention_count}건 재시도 필요",
        )
    else:
        tracker.complete()
    context.update({
        "status": cycle_status,
        "source_card_count": len(valid_card_ids),
        "new_rq_count": new_count,
        "strengthened_rq_count": strengthened_count,
        "resolved_rq_count": resolved_rq_count,
        "reinforced_rq_count": reinforced_rq_count,
    })



def _recover_retry_exhausted(
    context: dict[str, Any], error: Exception, definition: WorkflowDefinition,
) -> dict[str, Any]:
    assert isinstance(error, LLMRetryExhausted)
    ledger: Ledger = context["ledger"]
    review_id = str(context.get("review_id") or "")
    tracker: ExecutionRunTracker = context["run_tracker"]
    run_id = tracker.run_id
    if review_id:
        ledger.fail_research_state_review(review_id, str(error))
    failure_context: dict[str, Any] = {"kind": "auto_cycle"}
    if review_id:
        failure_context.update({"review_id": review_id, "actionable_questions": context.get("actionable_questions", [])})
    else:
        failure_context.update({
            "source_card_count": len(context["valid_card_ids"]),
            "source_card_ids": sorted(context["valid_card_ids"]),
        })
    tracker.fail_from_llm(
        error, context=failure_context, review_id=review_id,
    )
    if not review_id:
        return workflow_result(
            definition, context, status="rq_generation_failed",
            values={
                "source_card_count": len(context["valid_card_ids"]),
                "review_id": "", "research_questions": [], "ranked_questions": [],
                "dispatched_intents": [], "executions": [],
                "retry_run_id": run_id, "retry_error": error.error_type,
            },
        )
    return workflow_result(
        definition, context, status="needs_attention",
        values={
            "review_id": review_id,
            "source_card_count": len(context["valid_card_ids"]),
            "research_questions": [], "ranked_questions": [],
            "dispatched_intents": [], "executions": [],
            "retry_run_id": run_id, "retry_error": error.error_type,
        },
    )


def _recover_unexpected_error(
    context: dict[str, Any], error: Exception, definition: WorkflowDefinition,
) -> dict[str, Any]:
    review_id = str(context.get("review_id") or "")
    ledger: Ledger = context["ledger"]
    if review_id:
        ledger.fail_research_state_review(review_id, str(error))
    context["run_tracker"].needs_attention(
        stage="unexpected_error", error_type="unexpected_error", error_message=str(error),
    )
    raise error
