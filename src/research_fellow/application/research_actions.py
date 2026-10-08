"""Researcher-initiated follow-up actions for an existing Research Question."""
from __future__ import annotations
from typing import Any, Iterable, Mapping

from research_fellow.application.advising_rq_intents import create_auto_exploration_intent_for_rq
from research_fellow.application.advising_state import recent_knowledge_updates
from research_fellow.application.research_answer import latest_research_answer, request_initial_research_answer, request_research_answer_update
from research_fellow.application.research_evidence import rq_evidence_bundle
from research_fellow.storage import Ledger


def _compact(value: object, limit: int = 900) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def followup_literature_context(
    ledger: Ledger,
    memory_cards: Iterable[Mapping[str, Any]],
    rq_id: str,
) -> dict[str, Any]:
    """Build accumulated RQ context before a researcher approves another round.

    This is intentionally a read model over existing durable state: approved
    knowledge, RQ-scoped paper interpretations, and the latest answer draft.
    It does not create a new M2 task or Attention item.
    """
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")
    all_cards = [dict(x) for x in memory_cards]
    evidence = rq_evidence_bundle(ledger, all_cards, rq_id)
    direct = dict(evidence.get("direct") or {})
    papers = [dict(x) for x in direct.get("papers") or []]
    cards = [dict(x) for x in direct.get("cards") or []]
    latest_answer = latest_research_answer(ledger, rq_id)
    answer_payload = dict((latest_answer or {}).get("payload") or {})
    answer_draft = str(answer_payload.get("report") or "").strip()

    direct_card_ids = {str(x.get("card_id") or "") for x in cards if str(x.get("card_id") or "")}
    pending_updates = [
        item for item in recent_knowledge_updates(ledger, limit=500)
        if str((item.get("payload") or {}).get("card_id") or "") in direct_card_ids
    ]

    return {
        "research_question": dict(rq),
        "current_evidence_summary": {
            "paper_count": len(papers),
            "knowledge_count": len(cards),
            "paper_titles": [str(x.get("title") or "") for x in papers[:8] if x.get("title")],
            "knowledge_titles": [str(x.get("title") or "") for x in cards[:8] if x.get("title")],
            "paper_summaries": [
                {"title": str(x.get("title") or ""), "summary": _compact(x.get("summary"), 1200)}
                for x in papers[:6] if str(x.get("summary") or "").strip()
            ],
            "knowledge_summaries": [
                {
                    "card_id": str(x.get("card_id") or ""),
                    "title": str(x.get("title") or ""),
                    "claim": _compact(x.get("claim"), 650),
                    "implication": _compact(x.get("implication"), 500),
                    "conditions": _compact(x.get("conditions"), 350),
                    "limits": _compact(x.get("limits"), 350),
                }
                for x in cards[:10]
            ],
            "pending_knowledge_update_count": len(pending_updates),
            "pending_knowledge_update_ids": [str(x.get("phenomenon_id") or "") for x in pending_updates if x.get("phenomenon_id")],
            "latest_answer_version": int(answer_payload.get("answer_version") or (1 if latest_answer else 0)),
            "latest_answer_draft": _compact(answer_draft, 4500),
        },
    }


def _evidence_context_text(summary: Mapping[str, Any]) -> str:
    paper_titles = [str(x) for x in summary.get("paper_titles") or [] if str(x).strip()]
    knowledge_titles = [str(x) for x in summary.get("knowledge_titles") or [] if str(x).strip()]
    latest_answer = str(summary.get("latest_answer_draft") or "").strip()
    lines = []
    if latest_answer:
        lines.extend([
            "CURRENT RESEARCH ANSWER — PRIMARY FOLLOW-UP CONTEXT",
            f"Answer version: v{int(summary.get('latest_answer_version') or 1)}",
            latest_answer,
            "",
            "FOLLOW-UP SEARCH GOAL",
            "Use the current answer to identify what could change, qualify, weaken, strengthen, or extend it.",
            "Prioritize unsupported claims, weak evidence, unresolved contradictions, boundary conditions, missing mechanisms, and open questions.",
            "Do not merely retrieve more papers that repeat or confirm claims already well supported in the current answer.",
            "",
        ])
    lines.extend([
        "ACCUMULATED RQ EVIDENCE",
        f"Direct papers already collected for this RQ: {int(summary.get('paper_count') or 0)}",
        f"Approved knowledge already derived for this RQ: {int(summary.get('knowledge_count') or 0)}",
    ])
    if paper_titles:
        lines.append("Previously collected papers: " + " | ".join(paper_titles))
    if knowledge_titles:
        lines.append("Existing direct knowledge: " + " | ".join(knowledge_titles))

    knowledge_summaries = [dict(x) for x in summary.get("knowledge_summaries") or [] if isinstance(x, Mapping)]
    if knowledge_summaries:
        lines.append("APPROVED KNOWLEDGE SYNTHESIS")
        for item in knowledge_summaries:
            parts = [
                f"- {item.get('title') or 'Knowledge'}: {item.get('claim') or ''}",
                f"  Implication: {item.get('implication')}" if item.get("implication") else "",
                f"  Conditions: {item.get('conditions')}" if item.get("conditions") else "",
                f"  Limits: {item.get('limits')}" if item.get("limits") else "",
            ]
            lines.extend(part for part in parts if part)

    paper_summaries = [dict(x) for x in summary.get("paper_summaries") or [] if isinstance(x, Mapping)]
    if paper_summaries:
        lines.append("RQ-SPECIFIC PAPER INTERPRETATIONS")
        for item in paper_summaries:
            if item.get("summary"):
                lines.append(f"- {item.get('title') or 'Paper'}\n{item.get('summary')}")

    return "\n".join(lines)


def request_additional_literature(
    ledger: Ledger,
    memory_cards: Iterable[Mapping[str, Any]],
    rq_id: str,
    direction: Mapping[str, Any],
) -> dict[str, Any]:
    """Revise the research question and start a fresh literature round.

    The function name is retained for application/UI compatibility, but the
    boundary is now a Research Question revision rather than an accumulated
    follow-up search. The revised question is versioned first, then dispatched
    through the same fresh M1 literature-discovery path used by a new RQ.
    """
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")

    revised_question = str(direction.get("direction") or "").strip()
    if not revised_question:
        raise ValueError("연구질문을 입력하세요.")
    current_question = str(rq.get("question") or "").strip()
    if revised_question == current_question:
        raise ValueError("기존 연구질문과 다른 심화 연구질문을 입력하세요.")

    change_reason = "승인된 연구답변을 바탕으로 다음 라운드를 위해 연구질문을 구체화했습니다."
    if not ledger.refine_research_question(rq_id, revised_question, change_reason):
        raise ValueError("연구질문을 수정하지 못했습니다.")

    refreshed = ledger.research_question(rq_id) or {**rq, "question": revised_question}
    exploration_need = (
        "수정된 연구질문 자체와 연구자가 제공한 기존 연구 맥락을 기준으로 외부 문헌에서 "
        "기초 근거, 반대 근거, 적용 조건, 관련 방법과 사례를 탐색한다."
    )
    fresh = {
        **refreshed,
        "_fresh_intake": True,
        "research_context": str(refreshed.get("research_context") or "").strip(),
        "exploration_need": exploration_need,
        "rationale": str(refreshed.get("rationale") or "").strip(),
        "gap_or_tension": str(refreshed.get("gap_or_tension") or "").strip(),
    }
    dispatched = create_auto_exploration_intent_for_rq(
        ledger,
        fresh,
        score=4,
        selection_reason=(
            "연구자가 기존 연구답변을 검토한 뒤 연구질문을 수정했으며, "
            "수정된 질문에 대해 새로운 첫 문헌탐색을 시작합니다."
        ),
    )
    intent = dispatched["intent"]
    ledger.add_research_question_change(
        rq_id,
        "revised_question_literature_started",
        f"연구질문이 수정되어 새로운 초기 문헌탐색 라운드 Intent {intent.intent_id}를 시작했습니다: {revised_question}",
    )
    return {
        "status": "ready",
        "rq_id": rq_id,
        "intent_id": intent.intent_id,
        "title": intent.title,
        "direction": revised_question,
        "reused": bool(dispatched.get("reused")),
        "round_mode": "fresh_intake",
    }

def request_initial_answer(ledger: Ledger, memory_cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    return request_initial_research_answer(ledger, memory_cards, rq_id)


def request_answer_update(ledger: Ledger, memory_cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    return request_research_answer_update(ledger, memory_cards, rq_id)
