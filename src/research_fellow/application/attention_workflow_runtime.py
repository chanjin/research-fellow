"""Runtime reinjection and completion hooks for Attention-resumed workflows.

The workflow DSL remains free of concrete runtime dependencies.  This module is
an application-layer binding from a workflow id to the non-durable services that
must be reattached when a durable checkpoint resumes.
"""
from __future__ import annotations

from typing import Any, Callable

from research_fellow.application import advisory_session
from research_fellow.application.attention_resolution import WorkflowRuntimeBinding
from research_fellow.application.episodic_memory import store_advisory_episode
from research_fellow.storage import Ledger


def attention_workflow_runtime_binding(
    workflow_id: str,
    *,
    ledger: Ledger,
    retriever: Any,
    draft_fn: Callable[[str], str | None],
    semantic: bool,
    embedding_model: str,
) -> WorkflowRuntimeBinding:
    """Return runtime-only dependencies for a persisted workflow checkpoint."""
    if workflow_id != "m2_advisory_session":
        return WorkflowRuntimeBinding({}, {})

    def completed(result: dict[str, Any], attention_payload: dict[str, Any]) -> None:
        metadata = attention_payload.get("resume_metadata") if isinstance(attention_payload, dict) else None
        metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
        case_id = str(metadata.get("case_id") or "")
        if not case_id:
            return
        question = str(metadata.get("question") or "")
        research_context = str(metadata.get("context") or "")
        confirmed = bool(result.get("advisory_plan_confirmed"))
        plan = result.get("plan")
        if not confirmed:
            ledger.record(
                case_id, "advice_report", "m2", ["researcher"], "research_question_response",
                {"title": "M2 · 자문 계획 보류", "question": question, "context": research_context,
                 "report": "연구자가 자문 계획을 확인하지 않아 근거 수집과 최종 답변 생성을 진행하지 않았습니다."},
                status="completed",
            )
            return
        clusters = result.get("evidence_clusters") or []
        answer = str(result.get("answer") or "")
        judgments = list(result.get("judgments") or [])
        evidence_ids = list(result.get("evidence_card_ids") or [])
        relation_ids = list(result.get("evidence_relation_ids") or [])
        unresolved_items = list(result.get("unresolved_items") or [])
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "research_question_response",
            {"title": "M2 · 계획형 연구자 질문 대응", "question": question, "context": research_context,
             "report": answer, "evidence_card_ids": evidence_ids,
             "evidence_relation_ids": relation_ids, "subquestion_judgments": judgments},
            status="completed",
        )
        if plan is not None:
            store_advisory_episode(
                ledger, case_id=case_id, episode_type="research_question",
                situation_summary=f"연구자 질문: {question}\n연구 맥락: {research_context}",
                decision_question=str(getattr(plan, "decision_question", question)),
                advisory_plan=[str(getattr(item, "question", item)) for item in getattr(plan, "subquestions", ())],
                answer=answer, evidence_card_ids=evidence_ids, evidence_relation_ids=relation_ids,
                unresolved_items=unresolved_items,
            )

    return WorkflowRuntimeBinding(
        runtime_values={
            "autonomy_audit_recorder": ledger.record_autonomy_decision,
            "retriever": retriever,
            "draft_fn": draft_fn,
            "semantic": semantic,
            "embedding_model": embedding_model,
        },
        namespace=vars(advisory_session),
        on_completed=completed,
    )
