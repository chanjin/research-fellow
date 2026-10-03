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
    rq_id: str = ""

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
            "rq_id": self.rq_id,
        }


def _class_levels(interaction_id: str) -> tuple[str, str]:
    item = load_autonomy_classification().get(interaction_id)
    return (item.current_level, item.target_level) if item else ("H1", "H1")


def _attention_item(
    *, category: str, interaction_id: str, source_type: str, source_id: str,
    title: str, summary: str = "", priority: str = "medium", created_at: str = "",
    payload: Mapping[str, Any] | None = None,
    subject_title: str = "", round_id: str = "", round_label: str = "", phase_label: str = "", rq_id: str = "",
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
        rq_id=rq_id.strip(),
    )



def _rq_for_intent(ledger: Ledger, intent_id: str) -> tuple[str, str, int]:
    if not intent_id:
        return ("", "", 0)
    linked = ledger.research_questions_for_intent(intent_id)
    if not linked:
        # Research-answer execution uses the Research Question id itself as the
        # durable run intent_id.  Treat that as an RQ-scoped Attention item so
        # the queue can still show the owning question instead of a generic task.
        direct_rq = ledger.research_question(intent_id)
        if direct_rq:
            rq = dict(direct_rq)
            return (
                str(rq.get("rq_id") or intent_id),
                str(rq.get("question") or rq.get("title") or "").strip(),
                0,
            )
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


def _phase_label(interaction_id: str, *, stage: str = "") -> str:
    if interaction_id == "provide_external_llm_result":
        return {
            "research_answer_first_draft": "RQ 답변 작성 · 1/2 직접 근거 초안",
            "research_answer_enrichment": "RQ 답변 작성 · 2/2 최종 답변 보강",
            "single_paper_first_review": "논문 원문 리뷰",
            "search_strategy": "문헌 발견 · 외부 LLM 검색",
        }.get(stage, "외부 LLM 실행")
    return {
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
    rq_id = str(resume_metadata.get("rq_id") or "").strip()
    round_no = 0
    if ledger is not None and intent_id:
        linked_rq_id, linked_title, round_no = _rq_for_intent(ledger, intent_id)
        rq_id = linked_rq_id or rq_id
        rq_title = linked_title or rq_title
    subject_title = rq_title
    phase = _phase_label(interaction_id)
    interaction_title = contract.raw.get("purpose", interaction_id).split("\n", 1)[0].strip()
    title = subject_title or interaction_title
    rq_scoped_execution = bool(ledger is not None and rq_id and rq_id == intent_id and not ledger.research_questions_for_intent(intent_id))
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
        round_label=("" if rq_scoped_execution else (f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else ""))),
        phase_label=phase,
        rq_id=rq_id,
    )




def _completed_literature_intent_ids(ledger: Ledger) -> set[str]:
    """Return literature Intent ids that already reached durable completion.

    Historical runs/failures can outlive their workflow checkpoint.  Once the
    curation Intent or its auto-literature report is completed, those execution
    leftovers must not be projected back into Attention as a new search Input.
    """
    completed: set[str] = set()
    for row in ledger.phenomena(type_="curation_intent", status="completed"):
        payload = dict(row.get("payload") or {})
        intent = dict(payload.get("intent") or {})
        intent_id = str(intent.get("intent_id") or row.get("subject_id") or "").strip()
        if intent_id:
            completed.add(intent_id)
    for row in ledger.phenomena(type_="advice_report", status="completed"):
        if str(row.get("subject_type") or "") != "auto_literature_report":
            continue
        payload = dict(row.get("payload") or {})
        intent_id = str(payload.get("intent_id") or row.get("subject_id") or "").strip()
        if intent_id:
            completed.add(intent_id)
    return completed


def _synthetic_external_llm_attention(ledger: Ledger, existing: list[AttentionItem]) -> list[AttentionItem]:
    """Recover a missing external-LLM Attention item from durable run state.

    Older or partially migrated workspaces can contain an auto_research_run marked
    needs_attention even when the companion failure row/checkpoint was not persisted.
    The run is still durable execution state, so project a minimal actionable Input
    rather than leaving Research saying "exploring" with an empty Attention queue.
    """
    represented_runs: set[str] = set()
    waiting_intents: set[str] = set()
    completed_intents = _completed_literature_intent_ids(ledger)
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
        if intent_id and intent_id in completed_intents:
            # The paper-review/literature round already finished.  Do not revive
            # a stale search_strategy run as a brand-new Inputs card.
            continue
        if error_type != "external_llm_required":
            continue
        # search_strategy is reconstructible from the durable SearchProfile. Other
        # stages should normally have a failure row or workflow checkpoint and are
        # intentionally not guessed here.
        if stage != "search_strategy":
            continue
        rq_id, rq_title, round_no = _rq_for_intent(ledger, intent_id)
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
            phase_label=_phase_label("provide_external_llm_result", stage=stage),
            rq_id=rq_id,
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
        # Attention is a read model: enrich knowledge-card decisions with the
        # paper x research-question interpretation without copying it into the
        # durable decision request or card itself.
        request_view = dict(request)
        payload = dict(request_view.get("payload") or {})
        if str(request_view.get("subject_type") or "") == "knowledge_card":
            paper_id = str(payload.get("paper_id") or "").strip()
            rq_id = str(payload.get("rq_id") or "").strip()
            analysis = ledger.paper_question_analysis(paper_id, rq_id) if paper_id and rq_id else None
            if analysis is None and paper_id:
                analysis = ledger.paper_analysis(paper_id)
            paper = ledger.shelf_paper(paper_id) if paper_id else None
            payload["review_summary"] = str((analysis or {}).get("summary") or "")
            payload["review_paper_title"] = str((paper or {}).get("title") or "")
            request_view["payload"] = payload
        source_id = str(request_view.get("phenomenon_id") or request_view.get("request_id") or "")
        intent_id = str(payload.get("intent_id") or "").strip()
        linked_rq_id, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        request_rq_id = str(payload.get("rq_id") or "").strip() or linked_rq_id
        if request_rq_id and not rq_title:
            rq = ledger.research_question(request_rq_id) or {}
            rq_title = str(rq.get("question") or rq.get("title") or "").strip()
        items.append(_attention_item(
            category="decisions",
            interaction_id="resolve_pending_decision_requests",
            source_type="decision_request",
            source_id=source_id,
            title=str(payload.get("title") or request.get("subject_type") or "Decision request"),
            summary=str(payload.get("next_action") or payload.get("reason") or ""),
            priority="high" if str(request.get("subject_type") or "") in {"ontology", "research_question"} else "medium",
            created_at=str(request.get("created_at") or ""),
            payload=request_view,
            subject_title=rq_title,
            round_id=intent_id,
            round_label=(f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else "")),
            phase_label=("지식카드 검토" if intent_id else ""),
            rq_id=request_rq_id,
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
        rq_id, rq_title, round_no = _rq_for_intent(ledger, intent_id)
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
            rq_id=rq_id,
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

    completed_literature_intents = _completed_literature_intent_ids(ledger)
    for index, exception in enumerate(exceptions):
        intent_id = str(exception.get("intent_id") or "").strip()
        if intent_id and intent_id in active_round_ids:
            continue
        if intent_id and intent_id in completed_literature_intents:
            # Completed literature rounds may retain historical run/failure rows
            # for audit, but they are no longer actionable Attention items.
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
        linked_rq_id, rq_title, round_no = _rq_for_intent(ledger, intent_id)
        stage = str(exception.get("stage") or exception.get("current_stage") or "").strip()
        rq_scoped_execution = bool(linked_rq_id and linked_rq_id == intent_id and not ledger.research_questions_for_intent(intent_id))
        items.append(_attention_item(
            category=category,
            interaction_id=interaction_id,
            source_type=str(exception.get("source_type") or "execution_exception"),
            source_id=source_id,
            title=rq_title or str(exception.get("title") or "Execution exception"),
            summary=str(exception.get("summary") or exception.get("error") or ""),
            priority=str(exception.get("priority") or ("medium" if category != "exceptions" else "high")),
            created_at=str(exception.get("created_at") or ""),
            payload=exception,
            subject_title=rq_title,
            round_id=intent_id,
            round_label=("" if rq_scoped_execution else (f"문헌조사 #{round_no}" if round_no else ("문헌조사" if intent_id else ""))),
            phase_label=_phase_label(interaction_id, stage=stage) if intent_id else "",
            rq_id=linked_rq_id,
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
