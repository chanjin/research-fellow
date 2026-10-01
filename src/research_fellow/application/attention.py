"""Unified read model for human attention across agent workflows.

The queue is intentionally not a source of truth.  It projects existing durable
phenomena, pending interaction requests, and execution exceptions into a common
shape that a redesigned UI can use as its primary human-oversight entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from research_fellow.application.decision_interaction import pending_researcher_decision_requests
from research_fellow.application.dsl.autonomy_classification import load_autonomy_classification
from research_fellow.application.dsl.interaction import interaction_contract
from research_fellow.application.ontology_decision_interaction import pending_ontology_change_reviews
from research_fellow.storage import Ledger

ATTENTION_CATEGORIES = ("decisions", "reviews", "inputs", "exceptions")
_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class AttentionItem:
    attention_id: str
    category: str
    interaction_id: str
    title: str
    summary: str
    source_type: str
    source_id: str
    priority: str = "medium"
    created_at: str = ""
    payload: dict[str, Any] | None = None
    autonomy_level: str = "H1"
    target_autonomy_level: str = "H1"
    subject_title: str = ""
    round_id: str = ""
    round_label: str = ""
    phase_label: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "attention_id": self.attention_id,
            "category": self.category,
            "interaction_id": self.interaction_id,
            "title": self.title,
            "summary": self.summary,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "priority": self.priority,
            "created_at": self.created_at,
            "payload": dict(self.payload or {}),
            "autonomy_level": self.autonomy_level,
            "target_autonomy_level": self.target_autonomy_level,
            "subject_title": self.subject_title,
            "round_id": self.round_id,
            "round_label": self.round_label,
            "phase_label": self.phase_label,
        }


def _class_levels(interaction_id: str) -> tuple[str, str]:
    item = load_autonomy_classification().get(interaction_id)
    return (item.current_level, item.target_level) if item else ("H1", "H1")


def _attention_item(
    *, category: str, interaction_id: str, source_type: str, source_id: str,
    title: str, summary: str = "", priority: str = "medium", created_at: str = "",
    payload: Mapping[str, Any] | None = None,
    subject_title: str = "", round_id: str = "", round_label: str = "", phase_label: str = "",
) -> AttentionItem:
    if category not in ATTENTION_CATEGORIES:
        raise ValueError(f"Unsupported attention category: {category}")
    current, target = _class_levels(interaction_id)
    return AttentionItem(
        attention_id=f"{interaction_id}:{source_id}",
        category=category,
        interaction_id=interaction_id,
        title=title.strip() or interaction_id,
        summary=summary.strip(),
        source_type=source_type,
        source_id=source_id,
        priority=priority if priority in _PRIORITY_ORDER else "medium",
        created_at=created_at,
        payload=dict(payload or {}),
        autonomy_level=current,
        target_autonomy_level=target,
        subject_title=subject_title.strip(),
        round_id=round_id.strip(),
        round_label=round_label.strip(),
        phase_label=phase_label.strip(),
    )



def _rq_for_intent(ledger: Ledger, intent_id: str) -> tuple[str, str, int]:
    if not intent_id:
        return ("", "", 0)
    linked = ledger.research_questions_for_intent(intent_id)
    if not linked:
        return ("", "", 0)
    rq = dict(linked[0])
    rq_id = str(rq.get("rq_id") or "")
    title = str(rq.get("question") or rq.get("title") or "").strip()
    round_no = 0
    if rq_id:
        intents = list(reversed(ledger.research_question_intents(rq_id)))
        for idx, row in enumerate(intents, start=1):
            if str(row.get("intent_id") or "") == intent_id:
                round_no = idx
                break
    return (rq_id, title, round_no)


def _phase_label(interaction_id: str) -> str:
    return {
        "provide_external_llm_result": "외부 LLM 실행",
        "review_literature_candidates": "발견 문헌 검토",
        "resolve_pending_decision_requests": "지식카드 검토",
        "request_followup_literature_direction": "추가 문헌 조사 방향",
    }.get(interaction_id, interaction_id.replace("_", " "))

def _waiting_category(interaction_id: str) -> str:
    mode = interaction_contract(interaction_id).mode
    if mode in {"decide", "confirm"}:
        return "decisions"
    if mode in {"review", "select"}:
        return "reviews"
    return "inputs"


def attention_from_waiting_workflow(waiting_result: Mapping[str, Any], ledger: Ledger | None = None) -> AttentionItem | None:
    """Project a suspended WorkflowRun result into the Attention Queue."""
    if str(waiting_result.get("status") or "") != "waiting_for_interaction":
        return None
    request = waiting_result.get("interaction")
    if not isinstance(request, Mapping):
        return None
    interaction_id = str(request.get("interaction_id") or "").strip()
    if not interaction_id:
        return None
    workflow_id = str(waiting_result.get("workflow_id") or "workflow")
    checkpoint = waiting_result.get("checkpoint") or {}
    source_id = str(checkpoint.get("checkpoint_id") or f"{workflow_id}:{request.get('step_id') or interaction_id}")
    contract = interaction_contract(interaction_id)
    resume_metadata = dict(waiting_result.get("resume_metadata") or {}) if isinstance(waiting_result.get("resume_metadata"), Mapping) else {}
    intent_id = str(resume_metadata.get("intent_id") or "").strip()
    rq_title = str(resume_metadata.get("research_question") or "").strip()
    round_no = 0
    if ledger is not None and intent_id:
        _, linked_title, round_no = _rq_for_intent(ledger, intent_id)
        rq_title = linked_title or rq_title
    subject_title = rq_title
    phase = _phase_label(interaction_id)
    interaction_title = contract.raw.get("purpose", interaction_id).split("\n", 1)[0].strip()
    title = subject_title or interaction_title
    return _attention_item(
        category=_waiting_category(interaction_id),
        interaction_id=interaction_id,
        source_type="workflow_interaction",
        source_id=source_id,
        title=title,
        summary=f"{workflow_id} · {contract.mode}",
        priority="medium",
        payload={
            "workflow_id": workflow_id,
            "interaction": dict(request),
            "resume_metadata": dict(waiting_result.get("resume_metadata") or {}) if isinstance(waiting_result.get("resume_metadata"), Mapping) else {},
        },
        subject_title=subject_title,
        round_id=intent_id,
        round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
        phase_label=phase,
    )




def _synthetic_external_llm_attention(ledger: Ledger, existing: list[AttentionItem]) -> list[AttentionItem]:
    """Recover a missing external-LLM Attention item from durable run state.

    Older or partially migrated workspaces can contain an auto_research_run marked
    needs_attention even when the companion failure row/checkpoint was not persisted.
    The run is still durable execution state, so project a minimal actionable Input
    rather than leaving Research saying "exploring" with an empty Attention queue.
    """
    represented_runs: set[str] = set()
    waiting_intents: set[str] = set()
    for item in existing:
        payload = dict(item.payload or {})
        run_id = str(payload.get("run_id") or "").strip()
        if run_id:
            represented_runs.add(run_id)
        if item.source_type == "workflow_interaction" and item.round_id:
            waiting_intents.add(item.round_id)
    recovered: list[AttentionItem] = []
    for run in ledger.auto_research_runs(statuses=("needs_attention",), limit=200):
        run_id = str(run.get("run_id") or "").strip()
        intent_id = str(run.get("intent_id") or "").strip()
        stage = str(run.get("current_stage") or "").strip()
        error_type = str(run.get("last_error_type") or "").strip()
        if not run_id or run_id in represented_runs or intent_id in waiting_intents:
            continue
        if error_type != "external_llm_required":
            continue
        # search_strategy is reconstructible from the durable SearchProfile. Other
        # stages should normally have a failure row or workflow checkpoint and are
        # intentionally not guessed here.
        if stage != "search_strategy":
            continue
        _, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        recovered.append(_attention_item(
            category="inputs",
            interaction_id="provide_external_llm_result",
            source_type="auto_research_run_attention",
            source_id=run_id,
            title=rq_title or "External LLM execution required",
            summary="search_strategy · copy prompt → external LLM → paste response",
            priority="medium",
            created_at=str(run.get("updated_at") or run.get("created_at") or ""),
            payload=run,
            subject_title=rq_title,
            round_id=intent_id,
            round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
            phase_label="외부 LLM 실행",
        ))
    return recovered

def build_attention_queue(
    ledger: Ledger,
    *,
    waiting_workflows: Iterable[Mapping[str, Any]] = (),
    exceptions: Iterable[Mapping[str, Any]] = (),
) -> list[AttentionItem]:
    """Return the current human-attention read model.

    Ontology proposals are intentionally staged: proposals without researcher
    feedback appear as reviews; once feedback exists they move to final decisions.
    This avoids showing the same durable proposal twice in the queue.
    """
    items: list[AttentionItem] = []

    for request in pending_researcher_decision_requests(ledger):
        payload = dict(request.get("payload") or {})
        source_id = str(request.get("phenomenon_id") or request.get("request_id") or "")
        intent_id = str(payload.get("intent_id") or "").strip()
        _, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        items.append(_attention_item(
            category="decisions",
            interaction_id="resolve_pending_decision_requests",
            source_type="decision_request",
            source_id=source_id,
            title=str(payload.get("title") or request.get("subject_type") or "Decision request"),
            summary=str(payload.get("next_action") or payload.get("reason") or ""),
            priority="high" if str(request.get("subject_type") or "") in {"ontology", "research_question"} else "medium",
            created_at=str(request.get("created_at") or ""),
            payload=request,
            subject_title=rq_title,
            round_id=intent_id,
            round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
            phase_label=("지식카드 검토" if intent_id else ""),
        ))

    for review in pending_ontology_change_reviews(ledger):
        proposal = dict(review.get("proposal") or {})
        has_feedback = bool(str(review.get("researcher_comment") or "").strip())
        interaction_id = "resolve_ontology_change_reviews" if has_feedback else "review_ontology_change"
        category = "decisions" if has_feedback else "reviews"
        items.append(_attention_item(
            category=category,
            interaction_id=interaction_id,
            source_type="ontology_change_review",
            source_id=str(review.get("review_id") or ""),
            title=str(proposal.get("summary") or "Ontology change review"),
            summary="Final publish decision" if has_feedback else "Review proposed ontology change",
            priority="high",
            created_at=str(review.get("updated_at") or review.get("created_at") or ""),
            payload=review,
        ))

    for task in ledger.phenomena(type_="research_task", status="ready"):
        if str(task.get("subject_type") or "") != "paper_first_review":
            continue
        payload = dict(task.get("payload") or {})
        intent_id = str(payload.get("intent_id") or "").strip()
        _, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        paper = dict(payload.get("paper") or {})
        paper_title = str(paper.get("title") or payload.get("title") or "Paper").strip()
        items.append(_attention_item(
            category="inputs", interaction_id="provide_external_llm_result",
            source_type="research_task", source_id=str(task.get("phenomenon_id") or ""),
            title=rq_title or paper_title, summary=f"논문별 원문 리뷰 · {paper_title}",
            priority="medium", created_at=str(task.get("created_at") or ""), payload=task,
            subject_title=rq_title, round_id=intent_id,
            round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
            phase_label=f"논문 원문 리뷰 · {paper_title[:60]}",
        ))

    for result in waiting_workflows:
        item = attention_from_waiting_workflow(result, ledger)
        if item is not None:
            items.append(item)

    # A suspended workflow checkpoint is the authoritative current boundary for
    # a literature round. Once such a checkpoint exists, historical run/failure
    # projections from the same Intent must not keep showing an older Input.
    active_round_ids = {
        item.round_id for item in items
        if item.source_type == "workflow_interaction" and item.round_id
    }

    for index, exception in enumerate(exceptions):
        intent_id = str(exception.get("intent_id") or "").strip()
        if intent_id and intent_id in active_round_ids:
            continue
        source_id = str(exception.get("id") or exception.get("task_id") or f"exception-{index}")
        interaction_id = str(exception.get("interaction_id") or "").strip()
        category = "exceptions"
        # A durable execution record can represent a normal human input boundary
        # (for example external-manual LLM execution), not only a failure.  If the
        # record names a valid Interaction Contract, classify it by that contract
        # so normal requests do not appear as Exceptions.
        if interaction_id:
            try:
                category = _waiting_category(interaction_id)
            except Exception:
                category = "exceptions"
        _, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        items.append(_attention_item(
            category=category,
            interaction_id=interaction_id,
            source_type=str(exception.get("source_type") or "execution_exception"),
            source_id=source_id,
            title=str(exception.get("title") or "Execution exception"),
            summary=str(exception.get("summary") or exception.get("error") or ""),
            priority=str(exception.get("priority") or ("medium" if category != "exceptions" else "high")),
            created_at=str(exception.get("created_at") or ""),
            payload=exception,
            subject_title=rq_title,
            round_id=intent_id,
            round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
            phase_label=_phase_label(interaction_id) if intent_id else "",
        ))

    # Recover durable run-level external LLM boundaries when an older/partial
    # workspace is missing the companion failure/checkpoint projection.
    items.extend(_synthetic_external_llm_attention(ledger, items))

    # De-duplicate by stable attention identity. For external-LLM execution records,
    # also collapse repeated durable failures for the same logical run/stage/item so
    # an obsolete prompt does not crowd out the current request.
    by_id: dict[str, AttentionItem] = {}
    external_by_logical_key: dict[tuple[str, str, str], AttentionItem] = {}
    for item in items:
        if item.source_type == "auto_research_failure" and item.interaction_id == "provide_external_llm_result":
            payload = dict(item.payload or {})
            logical_key = (
                str(payload.get("run_id") or ""),
                str(payload.get("stage") or ""),
                str(payload.get("item_key") or ""),
            )
            prior = external_by_logical_key.get(logical_key)
            if prior is None or item.created_at > prior.created_at:
                external_by_logical_key[logical_key] = item
            continue
        prior = by_id.get(item.attention_id)
        if prior is None or item.created_at > prior.created_at:
            by_id[item.attention_id] = item
    for item in external_by_logical_key.values():
        by_id[item.attention_id] = item

    # Human attention is operational work: within one priority/category, show the
    # newest request first so a newly submitted Research Question is immediately
    # visible instead of being buried under historical unresolved records.
    ordered = sorted(by_id.values(), key=lambda item: str(item.created_at or ""), reverse=True)
    return sorted(
        ordered,
        key=lambda item: (_PRIORITY_ORDER[item.priority], ATTENTION_CATEGORIES.index(item.category)),
    )


def attention_queue_snapshot(
    ledger: Ledger,
    *,
    waiting_workflows: Iterable[Mapping[str, Any]] = (),
    exceptions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    items = build_attention_queue(ledger, waiting_workflows=waiting_workflows, exceptions=exceptions)
    groups = {category: [] for category in ATTENTION_CATEGORIES}
    for item in items:
        groups[item.category].append(item.as_dict())
    return {
        "total": len(items),
        "counts": {key: len(value) for key, value in groups.items()},
        "groups": groups,
        "items": [item.as_dict() for item in items],
    }
