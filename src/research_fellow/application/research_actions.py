"""Researcher-initiated follow-up actions for an existing Research Question."""
from __future__ import annotations
from typing import Any, Iterable, Mapping

from research_fellow.application.advising_rq_intents import create_auto_exploration_intent_for_rq
from research_fellow.application.research_answer import request_research_answer_update
from research_fellow.application.research_evidence import rq_evidence_bundle
from research_fellow.storage import Ledger


def followup_literature_context(
    ledger: Ledger,
    memory_cards: Iterable[Mapping[str, Any]],
    rq_id: str,
) -> dict[str, Any]:
    """Build the researcher-facing context used to direct a new literature round."""
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")
    evidence = rq_evidence_bundle(ledger, [dict(x) for x in memory_cards], rq_id)
    direct = dict(evidence.get("direct") or {})
    papers = [dict(x) for x in direct.get("papers") or []]
    cards = [dict(x) for x in direct.get("cards") or []]
    return {
        "research_question": dict(rq),
        "current_evidence_summary": {
            "paper_count": len(papers),
            "knowledge_count": len(cards),
            "paper_titles": [str(x.get("title") or "") for x in papers[:8] if x.get("title")],
            "knowledge_titles": [str(x.get("title") or "") for x in cards[:8] if x.get("title")],
        },
    }


def _evidence_context_text(summary: Mapping[str, Any]) -> str:
    paper_titles = [str(x) for x in summary.get("paper_titles") or [] if str(x).strip()]
    knowledge_titles = [str(x) for x in summary.get("knowledge_titles") or [] if str(x).strip()]
    lines = [
        f"Direct papers already collected for this RQ: {int(summary.get('paper_count') or 0)}",
        f"Approved knowledge already derived for this RQ: {int(summary.get('knowledge_count') or 0)}",
    ]
    if paper_titles:
        lines.append("Previously collected papers: " + " | ".join(paper_titles))
    if knowledge_titles:
        lines.append("Existing direct knowledge: " + " | ".join(knowledge_titles))
    return "\n".join(lines)


def request_additional_literature(
    ledger: Ledger,
    memory_cards: Iterable[Mapping[str, Any]],
    rq_id: str,
    direction: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one explicit follow-up M1 round from researcher-provided direction."""
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")
    direction_text = str(direction.get("direction") or "").strip()
    if not direction_text:
        raise ValueError("추가 문헌 조사 방향을 입력하세요.")
    specific_questions = str(direction.get("specific_questions") or "").strip()
    expected_evidence = str(direction.get("expected_evidence") or "").strip()

    current = followup_literature_context(ledger, memory_cards, rq_id)
    evidence_summary = dict(current.get("current_evidence_summary") or {})
    base_context = str(rq.get("research_context") or "").strip()
    followup_context = "\n".join(
        part for part in [
            base_context,
            "FOLLOW-UP LITERATURE ROUND",
            f"Researcher-directed exploration: {direction_text}",
            f"Specific questions to verify: {specific_questions}" if specific_questions else "",
            _evidence_context_text(evidence_summary),
            "Avoid simply repeating papers already collected for this research question; prioritize literature that extends, challenges, or fills gaps in the current evidence.",
        ] if part
    )
    exploration_need = expected_evidence or (
        f"추가 탐색 방향 '{direction_text}'에 대해 현재 RQ의 기존 근거를 보완·반박·확장하는 출처 기반 근거"
    )
    forced = {
        **rq,
        "_force_followup": True,
        "_followup_direction": direction_text,
        "_followup_specific_questions": specific_questions,
        "research_context": followup_context,
        "exploration_need": exploration_need,
    }
    dispatched = create_auto_exploration_intent_for_rq(
        ledger,
        forced,
        score=4,
        selection_reason=(
            f"연구자가 현재 연구질문에 대해 추가 문헌 조사 방향을 지정했습니다: {direction_text}"
        ),
    )
    intent = dispatched["intent"]
    ledger.add_research_question_change(
        rq_id,
        "additional_literature_requested",
        (
            f"연구자가 추가 문헌 조사 방향을 지정하여 M1 탐색 Intent {intent.intent_id}를 새로 연결했습니다. "
            f"방향: {direction_text}"
        ),
    )
    return {
        "status": "ready",
        "rq_id": rq_id,
        "intent_id": intent.intent_id,
        "title": intent.title,
        "direction": direction_text,
        "specific_questions": specific_questions,
        "expected_evidence": exploration_need,
        "current_evidence_summary": evidence_summary,
        "reused": bool(dispatched.get("reused")),
    }


def request_answer_update(ledger: Ledger, memory_cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    return request_research_answer_update(ledger, memory_cards, rq_id)
