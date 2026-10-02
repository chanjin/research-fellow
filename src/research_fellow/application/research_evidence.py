"""RQ-centred evidence lineage and supporting-evidence projection."""
from __future__ import annotations

import re
from typing import Any, Iterable

from research_fellow.origin_lineage import normalize_origin_links
from research_fellow.storage import Ledger


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[\w가-힣]{2,}", str(text or ""))}


def _card_text(card: dict[str, Any]) -> str:
    return " ".join([
        str(card.get("title") or ""), str(card.get("claim") or ""),
        str(card.get("context") or ""), str(card.get("implication") or ""),
        " ".join(str(x) for x in card.get("labels") or []),
        " ".join(str(x) for x in card.get("concepts") or []),
        str(card.get("conditions") or ""), str(card.get("limits") or ""),
    ])


def _paper_text(paper: dict[str, Any], analysis: dict[str, Any] | None = None) -> str:
    analysis = analysis or {}
    return " ".join([
        str(paper.get("title") or ""), str(paper.get("abstract") or ""),
        str(analysis.get("summary") or ""),
        " ".join(str(x) for x in paper.get("labels") or []),
    ])


def _analysis_for_research_question(ledger: Ledger, paper_id: str, rq_id: str) -> dict[str, Any]:
    """Return analysis valid for this RQ without leaking another RQ's interpretation.

    Legacy paper-level analysis is used only when the workspace has no RQ-scoped
    analyses for this paper, preserving old workspaces without reintroducing the
    single-analysis overwrite model.
    """
    scoped = ledger.paper_question_analysis(paper_id, rq_id) if rq_id else None
    if scoped:
        return scoped
    if ledger.paper_question_analyses(paper_id):
        return {}
    return ledger.paper_analysis(paper_id) or {}


def has_rq_origin(item: dict[str, Any], rq_id: str) -> bool:
    return any(
        str(link.get("origin_id") or "") == rq_id
        for link in normalize_origin_links(item.get("origin_links") or [])
    )


def _lexical_score(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(text)) / len(query_tokens)


def rq_evidence_bundle(
    ledger: Ledger,
    cards: Iterable[dict[str, Any]],
    rq_id: str,
    *,
    supporting_card_limit: int = 6,
    supporting_paper_limit: int = 6,
) -> dict[str, Any]:
    """Return direct RQ evidence first and prior workspace support second."""
    rq = ledger.research_question(rq_id) or {}
    question = str(rq.get("question") or "")
    all_cards = list(cards)
    all_papers = ledger.shelf_papers(limit=500)

    direct_cards = [card for card in all_cards if has_rq_origin(card, rq_id)]
    direct_card_ids = {str(card.get("card_id") or "") for card in direct_cards}
    direct_papers = [paper for paper in all_papers if has_rq_origin(paper, rq_id)]
    direct_paper_ids = {str(paper.get("paper_id") or "") for paper in direct_papers}

    explicit_support_ids = {str(x) for x in rq.get("source_card_ids") or [] if str(x)}
    ranked_cards: list[tuple[float, dict[str, Any], str]] = []
    for card in all_cards:
        card_id = str(card.get("card_id") or "")
        if card_id in direct_card_ids:
            continue
        score = _lexical_score(question, _card_text(card))
        reason = "question_relevance"
        if card_id in explicit_support_ids:
            score = max(score, 1.0)
            reason = "linked_existing_knowledge"
        if score > 0:
            ranked_cards.append((score, card, reason))
    ranked_cards.sort(key=lambda item: (-item[0], str(item[1].get("title") or "")))
    supporting_cards = [
        {**card, "support_reason": reason, "support_score": round(score, 3)}
        for score, card, reason in ranked_cards[:supporting_card_limit]
    ]

    supporting_source_names = {
        str((card.get("provenance") or {}).get("source_name") or "").strip().lower()
        for card in supporting_cards
        if str((card.get("provenance") or {}).get("source_name") or "").strip()
    }
    ranked_papers: list[tuple[float, dict[str, Any], str]] = []
    for paper in all_papers:
        paper_id = str(paper.get("paper_id") or "")
        if paper_id in direct_paper_ids:
            continue
        analysis = _analysis_for_research_question(ledger, paper_id, rq_id)
        title = str(paper.get("title") or "").strip().lower()
        score = _lexical_score(question, _paper_text(paper, analysis))
        reason = "question_relevance"
        if title and title in supporting_source_names:
            score = max(score, 1.0)
            reason = "supports_existing_knowledge"
        if score > 0:
            ranked_papers.append((
                score,
                {**paper, "summary": analysis.get("summary") or paper.get("abstract") or ""},
                reason,
            ))
    ranked_papers.sort(key=lambda item: (-item[0], str(item[1].get("title") or "")))
    supporting_papers = [
        {**paper, "support_reason": reason, "support_score": round(score, 3)}
        for score, paper, reason in ranked_papers[:supporting_paper_limit]
    ]

    direct_papers_with_summary = []
    for paper in direct_papers:
        analysis = _analysis_for_research_question(ledger, str(paper.get("paper_id") or ""), rq_id)
        direct_papers_with_summary.append({
            **paper,
            "summary": analysis.get("summary") or paper.get("abstract") or "",
        })

    return {
        "rq_id": rq_id,
        "question": question,
        "direct": {"cards": direct_cards, "papers": direct_papers_with_summary},
        "supporting": {"cards": supporting_cards, "papers": supporting_papers},
        "counts": {
            "direct_cards": len(direct_cards),
            "direct_papers": len(direct_papers_with_summary),
            "supporting_cards": len(supporting_cards),
            "supporting_papers": len(supporting_papers),
        },
    }
