"""Job-centred research workspace read model.

This module intentionally owns no research state. It projects existing durable
research questions, literature-discovery runs, paper-shelf state, and shared
phenomena into a single view of the Research Fellow's current job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.storage import Ledger


@dataclass(frozen=True)
class ResearchQuestionView:
    rq_id: str
    question: str
    status: str
    rationale: str
    research_context: str
    exploration_need: str
    updated_at: str
    source_type: str = ""
    intent_count: int = 0
    evidence_source_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EvidenceWorkView:
    item_id: str
    kind: str
    title: str
    status: str
    updated_at: str
    detail: str = ""
    count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KnowledgeProgressView:
    update_id: str
    title: str
    created_at: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AdvisoryView:
    rq_id: str
    question: str
    source_type: str
    status: str
    updated_at: str
    requester: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class NextActionView:
    action_id: str
    priority: str
    title: str
    reason: str
    target_type: str
    target_id: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _active_questions(ledger: Ledger, *, limit: int) -> list[ResearchQuestionView]:
    rows = ledger.research_question_backlog(
        statuses=["exploring", "interested", "candidate", "hold"], limit=limit
    )
    result: list[ResearchQuestionView] = []
    for row in rows:
        rq_id = str(row.get("rq_id") or "")
        thread = ledger.research_question_thread(rq_id) or {}
        result.append(ResearchQuestionView(
            rq_id=rq_id,
            question=str(row.get("question") or ""),
            status=str(row.get("status") or "candidate"),
            rationale=str(row.get("rationale") or ""),
            research_context=str(row.get("research_context") or ""),
            exploration_need=str(row.get("exploration_need") or ""),
            updated_at=str(row.get("updated_at") or ""),
            source_type=str(thread.get("source_type") or ""),
            intent_count=len(ledger.research_question_intents(rq_id)),
            evidence_source_count=len(ledger.research_question_sources(rq_id)),
        ))
    return result


def _evidence_work(ledger: Ledger, *, limit: int) -> list[EvidenceWorkView]:
    result: list[EvidenceWorkView] = []
    for run in ledger.unified_literature_discovery_runs(limit=limit):
        results = list(run.get("results") or [])
        result.append(EvidenceWorkView(
            item_id=str(run.get("run_id") or run.get("session_id") or ""),
            kind="literature_discovery",
            title=str(run.get("topic") or "Literature discovery"),
            status=str(run.get("status") or "completed"),
            updated_at=str(run.get("updated_at") or run.get("created_at") or ""),
            detail=str(run.get("search_summary") or run.get("query") or ""),
            count=len(results),
        ))
    for paper in ledger.shelf_papers(limit=limit):
        reading = str(paper.get("reading_status") or "unread")
        if reading not in {"unread", "reading"}:
            continue
        result.append(EvidenceWorkView(
            item_id=str(paper.get("paper_id") or ""),
            kind="paper_reading",
            title=str(paper.get("title") or "Untitled paper"),
            status=reading,
            updated_at=str(paper.get("updated_at") or ""),
            detail=str(paper.get("shelf_status") or "reference"),
            count=0,
        ))
    result.sort(key=lambda item: item.updated_at, reverse=True)
    return result[:limit]


def _knowledge_progress(ledger: Ledger, *, limit: int) -> list[KnowledgeProgressView]:
    reviewed = ledger.reviewed_knowledge_update_ids()
    result: list[KnowledgeProgressView] = []
    for row in ledger.phenomena(type_="knowledge_update")[:limit]:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        update_id = str(row.get("phenomenon_id") or "")
        title = str(
            payload.get("title")
            or payload.get("claim")
            or payload.get("finding")
            or "Knowledge updated"
        )
        detail = str(payload.get("summary") or payload.get("change_summary") or "")
        result.append(KnowledgeProgressView(
            update_id=update_id,
            title=title,
            created_at=str(row.get("created_at") or ""),
            status="reviewed" if update_id in reviewed else "new",
            detail=detail,
        ))
    return result


def _advisory_requests(ledger: Ledger, questions: list[ResearchQuestionView], *, limit: int) -> list[AdvisoryView]:
    result: list[AdvisoryView] = []
    for question in questions:
        source_type = question.source_type
        if source_type not in {"external_advisory", "researcher", "researcher_question"}:
            continue
        thread = ledger.research_question_thread(question.rq_id) or {}
        payload = thread.get("source_payload") if isinstance(thread.get("source_payload"), dict) else {}
        result.append(AdvisoryView(
            rq_id=question.rq_id,
            question=question.question,
            source_type=source_type,
            status=question.status,
            updated_at=question.updated_at,
            requester=str(payload.get("requester") or payload.get("recipient") or ""),
        ))
    return result[:limit]


def _next_actions(
    questions: list[ResearchQuestionView],
    evidence: list[EvidenceWorkView],
    knowledge: list[KnowledgeProgressView],
    *,
    limit: int,
) -> list[NextActionView]:
    actions: list[NextActionView] = []
    for rq in questions:
        if rq.status in {"interested", "candidate"} and rq.intent_count == 0:
            actions.append(NextActionView(
                action_id=f"explore:{rq.rq_id}", priority="high",
                title=f"Explore research question: {rq.question}",
                reason=rq.exploration_need or "No exploration intent is linked yet.",
                target_type="research_question", target_id=rq.rq_id,
            ))
        elif rq.status == "exploring" and rq.evidence_source_count == 0:
            actions.append(NextActionView(
                action_id=f"evidence:{rq.rq_id}", priority="medium",
                title=f"Acquire evidence for: {rq.question}",
                reason="The question is being explored but no evidence source is linked yet.",
                target_type="research_question", target_id=rq.rq_id,
            ))
    for item in evidence:
        if item.kind == "paper_reading" and item.status == "unread":
            actions.append(NextActionView(
                action_id=f"read:{item.item_id}", priority="medium",
                title=f"Review paper: {item.title}",
                reason="The paper is in the research shelf but has not been read yet.",
                target_type="paper", target_id=item.item_id,
            ))
    new_updates = [item for item in knowledge if item.status == "new"]
    if new_updates:
        actions.append(NextActionView(
            action_id="review-new-knowledge", priority="medium",
            title=f"Review {len(new_updates)} new knowledge update(s)",
            reason="New M1 knowledge has not yet been consumed by a completed M2 research-state review.",
            target_type="knowledge_update", target_id=new_updates[0].update_id,
        ))
    rank = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda item: (rank.get(item.priority, 9), item.title.lower()))
    return actions[:limit]


def research_workspace_snapshot(ledger: Ledger, *, limit: int = 20) -> dict[str, Any]:
    """Return a read-only job view of the current research lifecycle."""
    limit = max(1, min(int(limit), 100))
    questions = _active_questions(ledger, limit=limit)
    evidence = _evidence_work(ledger, limit=limit)
    knowledge = _knowledge_progress(ledger, limit=limit)
    advisory = _advisory_requests(ledger, questions, limit=limit)
    next_actions = _next_actions(questions, evidence, knowledge, limit=limit)

    question_counts: dict[str, int] = {}
    for item in questions:
        question_counts[item.status] = question_counts.get(item.status, 0) + 1
    evidence_counts = {
        "discovery": sum(1 for item in evidence if item.kind == "literature_discovery"),
        "papers_to_read": sum(1 for item in evidence if item.kind == "paper_reading" and item.status == "unread"),
        "papers_reading": sum(1 for item in evidence if item.kind == "paper_reading" and item.status == "reading"),
    }
    return {
        "counts": {
            "active_questions": len(questions),
            "exploring_questions": question_counts.get("exploring", 0),
            "evidence_items": len(evidence),
            "new_knowledge": sum(1 for item in knowledge if item.status == "new"),
            "advisory_requests": len(advisory),
            "next_actions": len(next_actions),
        },
        "question_counts": question_counts,
        "evidence_counts": evidence_counts,
        "questions": [item.as_dict() for item in questions],
        "evidence": [item.as_dict() for item in evidence],
        "knowledge_progress": [item.as_dict() for item in knowledge],
        "advisory": [item.as_dict() for item in advisory],
        "next_actions": [item.as_dict() for item in next_actions],
    }
