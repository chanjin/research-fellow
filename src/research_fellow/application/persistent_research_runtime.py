"""Cross-workflow execution linkage for the persistent Research Fellow job.

The Workflow DSL already owns sequencing *inside* one workflow.  This module owns
application-level dispatch *between* top-level workflows when durable phenomena
say that the next job step is ready.  It creates no parallel job state: ready
Curation Intents, Knowledge Updates, Research Questions, execution runs, and
workflow checkpoints remain the sources of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from research_fellow.application.advising_state import recent_knowledge_updates
from research_fellow.application.auto_literature import execute_auto_literature_review
from research_fellow.application.research_cycle import execute_auto_research_cycle
from research_fellow.application.research_answer import ensure_answer_confirmation, execute_research_answer_draft
from research_fellow.memory import KnowledgeMemory
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]
LiteratureExecutor = Callable[..., dict[str, Any]]
ResearchCycleExecutor = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class PersistentResearchBindings:
    cache_dir: Path
    rq_drafter: Draft
    priority_drafter: Draft
    keyword_drafter: Draft
    abstract_reviewer: Draft
    fulltext_drafter: Draft
    synthesis_drafter: Draft


@dataclass(frozen=True)
class ExecutionLink:
    kind: str
    source_id: str
    status: str
    workflow_id: str
    summary: str
    target_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "status": self.status,
            "workflow_id": self.workflow_id,
            "summary": self.summary,
            "target_ids": list(self.target_ids),
        }


def _ready_curation_intents(ledger: Ledger, intent_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
    if intent_ids is None:
        return rows
    result: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") or {}
        intent_id = str(payload.get("intent_id") or row.get("subject_id") or "")
        if intent_id in intent_ids:
            result.append(row)
    return result



def _intent_has_waiting_checkpoint(checkpoint_dir: Path, intent_id: str) -> bool:
    if not intent_id or not checkpoint_dir.exists():
        return False
    import json
    for path in checkpoint_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "waiting_for_interaction":
            continue
        meta = payload.get("resume_metadata") or {}
        if isinstance(meta, dict) and str(meta.get("intent_id") or "") == intent_id:
            return True
    return False

def _linked_rq_ids(ledger: Ledger, intent_event: dict[str, Any]) -> list[str]:
    payload = intent_event.get("payload") or {}
    intent_id = str(payload.get("intent_id") or intent_event.get("subject_id") or "")
    return [
        str(item.get("rq_id") or "")
        for item in ledger.research_questions_for_intent(intent_id)
        if str(item.get("rq_id") or "")
    ]


def _resume_run_id_for_intent(ledger: Ledger, intent_id: str) -> str:
    for run in ledger.auto_research_runs(statuses=("needs_attention", "running"), limit=100):
        if str(run.get("intent_id") or "") == intent_id:
            return str(run.get("run_id") or "")
    return ""


def _resume_run_id_for_m2(ledger: Ledger) -> str:
    for run in ledger.auto_research_runs(statuses=("needs_attention", "running"), limit=100):
        if not str(run.get("intent_id") or "") and str(run.get("current_stage") or "") in {"rq_generation", "rq_prioritization", "research_state_update"}:
            return str(run.get("run_id") or "")
    return ""


def _record_m1_handoff_result(
    ledger: Ledger,
    intent_event: dict[str, Any],
    result: dict[str, Any],
) -> tuple[str, ...]:
    rq_ids = tuple(_linked_rq_ids(ledger, intent_event))
    status = str(result.get("status") or "failed")
    payload = intent_event.get("payload") or {}
    title = str(payload.get("title") or payload.get("question") or "M1 evidence acquisition")
    if status == "completed":
        summary = f"M1 문헌조사가 완료되었습니다: {title}. 결과를 후속 연구판단에 사용할 수 있습니다."
        change_type = "m1_evidence_acquired"
    elif status == "needs_attention":
        summary = f"M1 문헌조사는 현재 사람의 입력을 기다립니다: {title}. 외부 LLM 실행 또는 연구자 선택이 필요한 정상 Interaction boundary입니다."
        change_type = "m1_execution_attention"
    else:
        summary = f"M1 문헌조사가 완료되지 않았습니다: {title}. 실행 복구가 필요합니다."
        change_type = "m1_execution_failed"
    for rq_id in rq_ids:
        ledger.add_research_question_change(rq_id, change_type, summary)
    return rq_ids


def _execute_ready_m1(
    ledger: Ledger,
    bindings: PersistentResearchBindings,
    *,
    intent_ids: set[str] | None,
    max_items: int,
    literature_executor: LiteratureExecutor,
) -> list[ExecutionLink]:
    links: list[ExecutionLink] = []
    for event in _ready_curation_intents(ledger, intent_ids)[: max(0, int(max_items))]:
        payload = event.get("payload") or {}
        intent_id = str(payload.get("intent_id") or event.get("subject_id") or event.get("phenomenon_id") or "")
        checkpoint_dir = bindings.cache_dir / "workflow_checkpoints"
        # A ready Intent with a suspended workflow is already in progress.
        # Do not restart the literature round from step 1 on every Streamlit rerun;
        # the current human boundary must be resumed from its durable checkpoint.
        if _intent_has_waiting_checkpoint(checkpoint_dir, intent_id):
            rq_ids = tuple(_linked_rq_ids(ledger, event))
            links.append(ExecutionLink(
                kind="curation_intent_to_m1", source_id=intent_id, status="needs_attention",
                workflow_id="m1_auto_literature_review",
                summary="문헌조사 라운드가 현재 사람의 입력을 기다리고 있어 기존 checkpoint에서 계속합니다.",
                target_ids=rq_ids,
            ))
            continue
        result = literature_executor(
            ledger,
            event,
            bindings.cache_dir,
            keyword_drafter=bindings.keyword_drafter,
            abstract_reviewer=bindings.abstract_reviewer,
            fulltext_drafter=bindings.fulltext_drafter,
            synthesis_drafter=bindings.synthesis_drafter,
            resume_run_id=_resume_run_id_for_intent(ledger, intent_id),
            checkpoint_dir=checkpoint_dir,
        )
        rq_ids = _record_m1_handoff_result(ledger, event, result)
        status = str(result.get("status") or "failed")
        if status == "completed":
            summary = "M1 문헌조사를 실행하고 결과 보고 및 Intent 완료까지 연결했습니다."
        elif status == "needs_attention":
            summary = "M1 문헌조사를 시작했고 다음 사람 입력 단계로 연결했습니다. Attention > Inputs/Reviews에서 외부 LLM 응답 또는 논문 선택을 완료하면 Agent가 계속 진행합니다."
        else:
            summary = "M1 문헌조사 실행이 실패 상태로 종료되어 복구 대상으로 남겼습니다."
        links.append(ExecutionLink(
            kind="curation_intent_to_m1",
            source_id=intent_id,
            status=status,
            workflow_id="m1_auto_literature_review",
            summary=summary,
            target_ids=rq_ids,
        ))
    return links


def _consume_knowledge_updates(
    ledger: Ledger,
    memory: KnowledgeMemory,
    bindings: PersistentResearchBindings,
    *,
    research_cycle_executor: ResearchCycleExecutor,
) -> ExecutionLink | None:
    updates = recent_knowledge_updates(ledger, limit=500)
    if not updates:
        return None
    update_ids = tuple(
        str(item.get("phenomenon_id") or "")
        for item in updates
        if str(item.get("phenomenon_id") or "")
    )
    result = research_cycle_executor(
        ledger,
        memory.all(),
        bindings.cache_dir,
        rq_drafter=bindings.rq_drafter,
        priority_drafter=bindings.priority_drafter,
        keyword_drafter=bindings.keyword_drafter,
        abstract_reviewer=bindings.abstract_reviewer,
        fulltext_drafter=bindings.fulltext_drafter,
        synthesis_drafter=bindings.synthesis_drafter,
        resume_run_id=_resume_run_id_for_m2(ledger),
        source_updates=updates,
        state_update_drafter=bindings.rq_drafter,
    )
    status = str(result.get("status") or "completed")
    selected = len(result.get("dispatched") or [])
    summary = (
        f"새 승인 지식 {len(updates)}건을 M2 연구상태 재평가에 반영했고, "
        f"후속 M1 탐색 {selected}건을 연결했습니다."
    )
    return ExecutionLink(
        kind="knowledge_update_to_m2",
        source_id=update_ids[0] if update_ids else "knowledge_update",
        status=status,
        workflow_id="m2_research_cycle",
        summary=summary,
        target_ids=update_ids,
    )



def _completed_literature_intents_for_rq(ledger: Ledger, rq_id: str) -> set[str]:
    intent_ids = {str(x.get("intent_id") or "") for x in ledger.research_question_intents(rq_id) if str(x.get("intent_id") or "")}
    completed: set[str] = set()
    for row in ledger.phenomena(type_="advice_report"):
        if row.get("subject_type") != "auto_literature_report" or row.get("status") != "completed":
            continue
        intent_id = str((row.get("payload") or {}).get("intent_id") or "")
        if intent_id in intent_ids:
            completed.add(intent_id)
    return completed


def _pending_knowledge_requests_for_intents(ledger: Ledger, intent_ids: set[str]) -> bool:
    if not intent_ids:
        return False
    for row in ledger.phenomena(type_="decision_request", status="proposed"):
        if row.get("subject_type") not in {"knowledge_card", "knowledge_evidence"}:
            continue
        if str((row.get("payload") or {}).get("intent_id") or "") in intent_ids:
            return True
    return False


def _ensure_research_answer_confirmations(ledger: Ledger, memory: KnowledgeMemory) -> list[ExecutionLink]:
    links: list[ExecutionLink] = []
    for rq in ledger.research_question_backlog(statuses=["interested", "exploring", "hold"], limit=100):
        rq_id = str(rq.get("rq_id") or "")
        if not rq_id:
            continue
        completed_intents = _completed_literature_intents_for_rq(ledger, rq_id)
        if not completed_intents or _pending_knowledge_requests_for_intents(ledger, completed_intents):
            continue
        request_id = ensure_answer_confirmation(ledger, memory.all(), rq_id)
        if request_id:
            links.append(ExecutionLink(
                kind="knowledge_ready_to_answer_confirmation", source_id=rq_id, status="needs_attention",
                workflow_id="m2_research_question_answer", summary="현재 연구질문에 대한 지식이 업데이트되어 답변 초안 작성 여부를 연구자에게 확인합니다.",
                target_ids=(request_id,),
            ))
    return links


def _resume_run_id_for_answer(ledger: Ledger, rq_id: str) -> str:
    for run in ledger.auto_research_runs(statuses=("needs_attention", "running"), limit=200):
        if str(run.get("intent_id") or "") == rq_id and str(run.get("current_stage") or "") in {"research_answer_draft", "research_answer_first_draft", "research_answer_enrichment"}:
            return str(run.get("run_id") or "")
    return ""


def _execute_ready_answer_drafts(ledger: Ledger, memory: KnowledgeMemory, bindings: PersistentResearchBindings) -> list[ExecutionLink]:
    links: list[ExecutionLink] = []
    for event in [x for x in ledger.phenomena(recipient="m2", type_="advisory_exchange", status="ready") if x.get("subject_type") == "research_answer_draft_request"][:3]:
        rq_id = str((event.get("payload") or {}).get("rq_id") or event.get("subject_id") or "")
        unresolved = [f for f in ledger.auto_research_failures(status="needs_attention", limit=200)
                      if str(f.get("stage") or "") in {"research_answer_draft", "research_answer_first_draft", "research_answer_enrichment"} and str(f.get("intent_id") or "") == rq_id]
        if unresolved:
            links.append(ExecutionLink(kind="answer_confirmation_to_draft", source_id=rq_id, status="needs_attention", workflow_id="m2_research_question_answer", summary="답변 초안 작성에 외부 LLM 입력이 필요해 Attention > Inputs에서 대기합니다.", target_ids=(rq_id,)))
            continue
        result = execute_research_answer_draft(
            ledger, memory.all(), event, drafter=bindings.rq_drafter, resume_run_id=_resume_run_id_for_answer(ledger, rq_id),
        )
        status = str(result.get("status") or "failed")
        summary = (
            "답변 초안 작성에 외부 LLM 입력이 필요해 Attention > Inputs에서 대기합니다."
            if status == "needs_attention" else
            "현재 승인 지식과 근거 논문을 바탕으로 연구질문 답변 초안을 작성했습니다."
            if status == "completed" else "연구질문 답변 초안 작성이 완료되지 않았습니다."
        )
        links.append(ExecutionLink(kind="answer_confirmation_to_draft", source_id=rq_id, status=status, workflow_id="m2_research_question_answer", summary=summary, target_ids=(rq_id,)))
    return links

def advance_persistent_research(
    ledger: Ledger,
    memory: KnowledgeMemory,
    bindings: PersistentResearchBindings,
    *,
    intent_ids: Iterable[str] | None = None,
    max_m1_items: int = 3,
    consume_knowledge_updates: bool = False,
    literature_executor: LiteratureExecutor = execute_auto_literature_review,
    research_cycle_executor: ResearchCycleExecutor = execute_auto_research_cycle,
) -> dict[str, Any]:
    """Advance all currently runnable cross-workflow handoffs to a stable boundary.

    Human-required steps are *not* bypassed here.  Intra-workflow Interaction
    boundaries remain owned by the Workflow runtime/Attention checkpoint path.
    This dispatcher consumes already-ready durable work. Knowledge updates stay
    as M1→M2 inbox signals until an explicit researcher action chooses how to use
    them; they are not an automatic trigger for a new research cycle.
    """
    requested_ids = {
        str(value).strip() for value in (intent_ids or []) if str(value).strip()
    } or None
    links: list[ExecutionLink] = []

    # Knowledge approval itself never creates a new human-attention card.
    # Approved cards remain durable M1→M2 updates until the researcher explicitly
    # chooses the next job action from Research (draft answer or more literature).
    # Once that action is approved, its external-LLM boundary may create Attention.
    answer_links = _execute_ready_answer_drafts(ledger, memory, bindings)
    links.extend(answer_links)

    # Legacy automatic M2 state evolution remains available only to an explicit
    # caller. Normal operation keeps it off so one newly approved card cannot
    # immediately produce a weak search/RQ prompt.
    if consume_knowledge_updates:
        knowledge_link = _consume_knowledge_updates(
            ledger, memory, bindings, research_cycle_executor=research_cycle_executor,
        )
        if knowledge_link is not None:
            links.append(knowledge_link)

    # Run ready M1 work that came from direct RQ intake or a human-approved
    # curation intent and was not already consumed by the research cycle above.
    links.extend(_execute_ready_m1(
        ledger,
        bindings,
        intent_ids=requested_ids,
        max_items=max_m1_items,
        literature_executor=literature_executor,
    ))

    needs_attention = any(link.status in {"needs_attention", "failed"} for link in links)
    return {
        "status": "needs_attention" if needs_attention else ("advanced" if links else "idle"),
        "advanced_count": len(links),
        "links": [link.as_dict() for link in links],
        "summary": " ".join(link.summary for link in links),
    }
