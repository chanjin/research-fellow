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
        }


def _class_levels(interaction_id: str) -> tuple[str, str]:
    item = load_autonomy_classification().get(interaction_id)
    return (item.current_level, item.target_level) if item else ("H1", "H1")


def _attention_item(
    *, category: str, interaction_id: str, source_type: str, source_id: str,
    title: str, summary: str = "", priority: str = "medium", created_at: str = "",
    payload: Mapping[str, Any] | None = None,
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
    )


def _waiting_category(interaction_id: str) -> str:
    mode = interaction_contract(interaction_id).mode
    if mode == "decide":
        return "decisions"
    if mode == "review":
        return "reviews"
    return "inputs"


def attention_from_waiting_workflow(waiting_result: Mapping[str, Any]) -> AttentionItem | None:
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
    return _attention_item(
        category=_waiting_category(interaction_id),
        interaction_id=interaction_id,
        source_type="workflow_interaction",
        source_id=source_id,
        title=contract.raw.get("purpose", interaction_id).split("\n", 1)[0].strip(),
        summary=f"{workflow_id} · {contract.mode}",
        priority="medium",
        payload={
            "workflow_id": workflow_id,
            "interaction": dict(request),
            "resume_metadata": dict(waiting_result.get("resume_metadata") or {}) if isinstance(waiting_result.get("resume_metadata"), Mapping) else {},
        },
    )


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

    for result in waiting_workflows:
        item = attention_from_waiting_workflow(result)
        if item is not None:
            items.append(item)

    for index, exception in enumerate(exceptions):
        source_id = str(exception.get("id") or exception.get("task_id") or f"exception-{index}")
        items.append(_attention_item(
            category="exceptions",
            interaction_id=str(exception.get("interaction_id") or ""),
            source_type=str(exception.get("source_type") or "execution_exception"),
            source_id=source_id,
            title=str(exception.get("title") or "Execution exception"),
            summary=str(exception.get("summary") or exception.get("error") or ""),
            priority=str(exception.get("priority") or "high"),
            created_at=str(exception.get("created_at") or ""),
            payload=exception,
        ))

    # De-duplicate by stable attention identity and prefer higher priority/latest item.
    by_id: dict[str, AttentionItem] = {}
    for item in items:
        prior = by_id.get(item.attention_id)
        if prior is None or (_PRIORITY_ORDER[item.priority], item.created_at) < (_PRIORITY_ORDER[prior.priority], prior.created_at):
            by_id[item.attention_id] = item
    return sorted(
        by_id.values(),
        key=lambda item: (_PRIORITY_ORDER[item.priority], ATTENTION_CATEGORIES.index(item.category), item.created_at),
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
