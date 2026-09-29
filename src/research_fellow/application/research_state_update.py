"""M2 research-question state refresh driven by new M1 knowledge updates."""
from __future__ import annotations

from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.llm_execution import execute_llm_stage
from research_fellow.application.structured_output import extract_json_value
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]
WORKFLOW_PATH = "m2/research_state_update.yaml"
VALID_ACTIONS = {"reinforce", "continue", "resolve", "explore"}


def execute_research_state_update(
    ledger: Ledger,
    knowledge_updates: list[dict[str, Any]],
    research_question_backlog: list[dict[str, Any]],
    *,
    drafter: Draft,
) -> dict[str, Any]:
    """Refresh existing Research Questions before M2 generates or dispatches new work."""
    if not research_question_backlog:
        return {
            "status": "no_existing_questions",
            "state_transitions": [],
            "followup_questions": [],
            "resolved_count": 0,
            "reinforced_count": 0,
        }
    workflow = prepare_workflow_run(WORKFLOW_PATH, {
        "ledger": ledger,
        "knowledge_updates": knowledge_updates,
        "research_question_backlog": research_question_backlog,
        "drafter": drafter,
    }, globals())
    return workflow.execute()


def _assess_research_question_state(context: dict[str, Any]) -> None:
    prompt = render_prompt(
        "m2_research_state_update.j2",
        knowledge_updates=_compact_updates(context["knowledge_updates"]),
        research_questions=_compact_questions(context["research_question_backlog"]),
    )
    valid_rq_ids = {str(item.get("rq_id", "")) for item in context["research_question_backlog"]}
    valid_update_ids = {
        str(update_id)
        for item in context["knowledge_updates"]
        for update_id in ([item.get("phenomenon_id")] + list(item.get("pending_update_ids") or []))
        if str(update_id or "")
    }
    result = execute_llm_stage(
        context["drafter"],
        prompt,
        stage="research_state_update",
        parser=lambda text: parse_research_state_assessments(
            text, valid_rq_ids=valid_rq_ids, valid_update_ids=valid_update_ids,
        ),
        accept=lambda items: isinstance(items, list),
    )
    context["assessments"] = result.value


def _apply_research_question_state(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    backlog_by_id = {str(item.get("rq_id", "")): item for item in context["research_question_backlog"]}
    transitions: list[dict[str, Any]] = []
    followups: list[dict[str, Any]] = []
    reinforced_count = 0
    resolved_count = 0

    for assessment in context["assessments"]:
        rq_id = assessment["rq_id"]
        rq = backlog_by_id.get(rq_id)
        if not rq:
            continue
        action = assessment["action"]
        previous_status = str(rq.get("status", "candidate"))
        next_status = previous_status
        if action == "resolve":
            next_status = "resolved"
            resolved_count += 1
        elif action == "explore":
            next_status = "interested" if previous_status not in {"exploring"} else previous_status
            followups.append({**rq, "status": next_status, "_force_followup": True})
        elif action == "reinforce":
            reinforced_count += 1

        if next_status != previous_status:
            ledger.update_research_question_status(rq_id, next_status)
        if assessment.get("exploration_need"):
            ledger.update_research_question_exploration_need(rq_id, assessment["exploration_need"])
        for update_id in assessment.get("source_update_ids", []):
            ledger.add_research_question_update_source(rq_id, update_id, assessment.get("reason", ""))
        ledger.add_research_question_change(
            rq_id,
            f"state_{action}",
            assessment.get("reason") or f"새 지식 업데이트에 따라 연구질문을 {action} 상태로 재평가했습니다.",
        )
        transitions.append({
            "rq_id": rq_id,
            "question": rq.get("question", ""),
            "action": action,
            "previous_status": previous_status,
            "status": next_status,
            "reason": assessment.get("reason", ""),
            "source_update_ids": assessment.get("source_update_ids", []),
        })

    context.update({
        "status": "updated",
        "state_transitions": transitions,
        "followup_questions": followups,
        "resolved_count": resolved_count,
        "reinforced_count": reinforced_count,
    })


def parse_research_state_assessments(
    text: str,
    *,
    valid_rq_ids: set[str],
    valid_update_ids: set[str],
) -> list[dict[str, Any]]:
    value = extract_json_value(text)
    if isinstance(value, dict):
        value = value.get("assessments", [])
    if not isinstance(value, list):
        raise ValueError("연구상태 갱신 응답은 assessments 배열이어야 합니다.")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            continue
        rq_id = str(raw.get("rq_id", "")).strip()
        action = str(raw.get("action", "")).strip().lower()
        reason = str(raw.get("reason", "")).strip()
        if not rq_id or rq_id not in valid_rq_ids or rq_id in seen or action not in VALID_ACTIONS or not reason:
            continue
        ids = [
            str(item).strip() for item in (raw.get("source_update_ids") or [])
            if str(item).strip() in valid_update_ids
        ]
        result.append({
            "rq_id": rq_id,
            "action": action,
            "reason": reason,
            "exploration_need": str(raw.get("exploration_need", "")).strip(),
            "source_update_ids": list(dict.fromkeys(ids))[:12],
        })
        seen.add(rq_id)
    return result


def _compact_updates(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in updates:
        payload = item.get("payload") or {}
        result.append({
            "update_id": item.get("phenomenon_id", ""),
            "pending_update_ids": item.get("pending_update_ids", []),
            "card_id": payload.get("card_id", ""),
            "title": payload.get("title", ""),
            "operation": payload.get("operation", ""),
        })
    return result


def _compact_questions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("rq_id", "question", "rationale", "gap_or_tension", "research_context", "exploration_need", "status", "source_card_ids", "source_update_ids")
    return [{key: item.get(key) for key in keys} for item in items]
