"""Promotion and source-intrinsic asset helpers for paper reading."""
from __future__ import annotations
import re
import uuid
from typing import Any
from research_fellow.domain.knowledge import KnowledgeCard
from research_fellow.storage import Ledger

def independent_card_context(item: dict[str, Any], paper_summary: str = "", *, max_chars: int = 1200) -> str:
    """Build reusable card context from the source paper, never from project lineage.

    The short, claim-specific context proposed during paper reading is preferred.
    A paper reading summary is only a fallback, so the card does not accumulate a
    duplicate essay or the current short-paper/Revision-To-do context.
    """
    suggested = re.sub(r"\s+", " ", str(item.get("suggested_context") or "")).strip()
    summary = re.sub(r"\s+", " ", str(paper_summary or "")).strip()
    context = suggested or summary
    if not context:
        return ""
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"


def promote_question(ledger: Ledger, paper: dict[str, Any], item: dict[str, Any], comment: str) -> str:
    """Turn a reviewed reading interpretation into a claim candidate, not a question card."""
    analysis = ledger.paper_analysis(paper["paper_id"]) or {}
    card_context = independent_card_context(item, str(analysis.get("summary") or ""))
    card = KnowledgeCard(
        card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}", title=item["question"][:72], source_kind="external_paper",
        claim=item["tentative_answer"], explanation=comment,
        context=card_context,
        implication=str(item.get("suggested_implication") or item.get("research_relevance") or ""),
        source_excerpt=str(item.get("suggested_source_excerpt") or "\n".join(item.get("evidence", [])))[:3200],
        labels=[], evidence_excerpt="\n".join(item["evidence"])[:1600],
        evidence_pages=[], citation_markers=[], conditions="not_assessed", limits=item["uncertainty"],
        provenance={
            "source_name": paper["title"], "paper_id": paper["paper_id"], "grounding": "paper_reading_review",
            "reading_question": item["question"],
        },
        origin_links=list(paper.get("origin_links", [])),
    ).model_dump(mode="json")
    case_id = ledger.create_case("research", f"Paper reading promotion: {paper['title'][:72]}")
    return ledger.record(case_id, "decision_request", "m1", ["researcher"], "knowledge_card", {
        "title": f"논문 읽기 기반 주장(Claim) 후보 승인: {card['title']}", "card": card,
        "paper_id": paper["paper_id"], "reading_question_id": item["question_id"],
        "next_action": "원문 근거와 연구자 첨삭을 확인한 뒤 승인 또는 보완 요청",
    }, subject_id=card["card_id"])


def promote_ontology_candidate(ledger: Ledger, paper: dict[str, Any], candidate: dict[str, Any], comment: str) -> str:
    """Request researcher approval for an abstraction before it enters ontology work."""
    case_id = ledger.create_case("research", f"Paper ontology proposal: {paper['title'][:72]}")
    payload = {
        "candidate_id": candidate["candidate_id"],
        "paper_id": paper["paper_id"],
        "paper_title": paper["title"],
        "statement": candidate["candidate_text"],
        "evidence": candidate["evidence"],
        "researcher_comment": comment,
    }
    return ledger.record(case_id, "decision_request", "m1", ["researcher"], "ontology_candidate", {
        "title": f"논문 기반 온톨로지 후보 승인: {candidate['candidate_text'][:72]}",
        "ontology_candidate": payload,
        "next_action": "일반화가 원문 근거와 연구 주제에 맞는지 검토하고, 승인 시 온톨로지 정리 대상으로 보냅니다.",
    }, subject_id=candidate["candidate_id"])
