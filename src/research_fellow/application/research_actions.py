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
    """Create a follow-up M1 round only after explicit researcher approval.

    New knowledge does not call an LLM or create Attention by itself.  At this
    boundary the researcher supplies/approves the exploration direction, and the
    resulting M1 intent receives the accumulated RQ evidence and latest answer.
    """
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
            "FOLLOW-UP LITERATURE ROUND — RESEARCHER APPROVED",
            f"Researcher-approved exploration: {direction_text}",
            f"Specific questions to verify: {specific_questions}" if specific_questions else "",
            _evidence_context_text(evidence_summary),
            "Use the accumulated knowledge and answer draft as the starting state. Do not repeat already collected papers unless needed for verification; prioritize literature that challenges, extends, or fills concrete gaps in the current research answer.",
        ] if part
    )
    exploration_need = expected_evidence or (
        f"추가 탐색 방향 '{direction_text}'에 대해 현재 RQ의 기존 근거와 답변 초안을 보완·반박·확장하는 출처 기반 근거"
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
            f"연구자가 축적된 지식과 현재 연구결과를 확인한 뒤 추가 문헌 조사 방향을 승인했습니다: {direction_text}"
        ),
    )
    intent = dispatched["intent"]

    # Existing review tables mark only the knowledge updates actually consumed
    # by this researcher-approved M2→M1 handoff. No new workflow state is added.
    pending_ids = {str(x) for x in evidence_summary.get("pending_knowledge_update_ids") or [] if str(x)}
    if pending_ids:
        relevant_updates = []
        for item in recent_knowledge_updates(ledger, limit=500):
            item_ids = {str(x) for x in (item.get("pending_update_ids") or [item.get("phenomenon_id")]) if str(x)}
            if item_ids & pending_ids:
                relevant_updates.append(item)
        review_id = ledger.create_research_state_review("manual", relevant_updates)
        ledger.link_research_question_review(review_id, rq_id, "followup_literature_approved")
        ledger.complete_research_state_review(
            review_id,
            generated_rq_count=0,
            selected_rq_count=1,
            summary=f"연구자가 누적 지식 {len(relevant_updates)}건을 반영한 추가 문헌 탐색을 승인했습니다.",
        )

    ledger.add_research_question_change(
        rq_id,
        "additional_literature_requested",
        (
            f"연구자가 누적 지식과 현재 연구결과를 확인한 뒤 M1 탐색 Intent {intent.intent_id}를 승인했습니다. "
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


def request_initial_answer(ledger: Ledger, memory_cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    return request_initial_research_answer(ledger, memory_cards, rq_id)


def request_answer_update(ledger: Ledger, memory_cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    return request_research_answer_update(ledger, memory_cards, rq_id)
