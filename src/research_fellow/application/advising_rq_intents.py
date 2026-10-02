"""M1 exploration-intent creation, duplicate detection, and dispatch for M2 research questions."""
from __future__ import annotations
import re
import uuid
from typing import Any
from research_fellow.domain.research import CurationIntent
from research_fellow.storage import Ledger

def create_exploration_intent_for_rq(ledger: Ledger, rq: dict[str, Any], priority: str = "보통") -> tuple[str, str]:
    """Turn one backlog RQ into a concrete M1 search contract awaiting researcher approval."""
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    question = str(rq.get("question", "")).strip()
    rationale = str(rq.get("rationale", "")).strip()
    gap = str(rq.get("gap_or_tension", "")).strip()
    context = str(rq.get("research_context", "")).strip()
    need = str(rq.get("exploration_need", "")).strip()
    rq_id = str(rq.get("rq_id") or intent_id)
    source_card_ids = [str(value) for value in rq.get("source_card_ids", []) if str(value)]
    origin_links = [{
        "origin_type": "researcher_question", "origin_id": rq_id,
        "label": question[:100], "source_card_ids": source_card_ids,
    }]
    if source_card_ids:
        origin_links.append({
            "origin_type": "m2_knowledge", "origin_id": rq_id,
            "label": f"{question[:80]} 관련 지식 보완", "source_card_ids": source_card_ids,
        })
    intent = CurationIntent(
        intent_id=intent_id,
        title=f"RQ 탐색 · {question[:45]}",
        purpose=need or f"이 연구질문이 제기된 근거와 지식 공백을 선행연구에서 확인한다: {rationale}",
        question=question,
        research_context="\n".join(item for item in [
            f"Research question: {question}",
            f"Why this question emerged: {rationale}",
            f"Gap or tension: {gap}" if gap else "",
            f"Research context: {context}" if context else "",
        ] if item),
        labels=[], priority=priority if priority in {"높음", "보통", "낮음"} else "보통",
        expected_evidence=need or "질문의 전제, 반대 근거, 적용 조건, 관련 방법 및 사례를 확인할 수 있는 출처 기반 근거",
        completion_condition="질문의 핵심 공백에 대해 출처가 확인된 지식카드 또는 명시적인 미해결 지식 공백을 M2에 보고한다.",
        execution_mode="manual", created_by="m2",
        origin_links=origin_links,
    )
    case_id = ledger.create_case("research", question[:80])
    request_id = ledger.record(
        case_id, "decision_request", "m2", ["researcher"], "curation_intent",
        {"title": f"M1 탐색 Intent 승인: {intent.title}", "intent": intent.model_dump(mode="json"),
         "next_action": "승인 시 M1 실행함으로 전달", "research_question_id": rq.get("rq_id", "")},
        subject_id=intent_id,
    )
    ledger.link_research_question_intent(str(rq.get("rq_id", "")), intent_id, request_id)
    return intent_id, request_id


def auto_exploration_candidates(ledger: Ledger, limit: int = 100) -> list[dict[str, Any]]:
    """Questions eligible for auto mode: candidate/interested and not already handed to M1."""
    backlog = ledger.research_question_backlog(statuses=["candidate", "interested"], limit=limit)
    return [rq for rq in backlog if not ledger.research_question_intents(str(rq.get("rq_id", "")))]


def _intent_terms(value: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "research", "study", "방법", "연구", "탐색", "관련"}
    return {token.lower() for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", value) if token.lower() not in stop}


def find_duplicate_curation_intent(ledger: Ledger, question: str, purpose: str, threshold: float = 0.72) -> dict[str, Any] | None:
    """Find a materially duplicate M1 task already recorded in the ledger."""
    target = _intent_terms(f"{question} {purpose}")
    if not target:
        return None
    normalized_question = " ".join(re.findall(r"[A-Za-z0-9가-힣]+", question.lower()))
    for event in ledger.phenomena(type_="curation_intent"):
        payload = event.get("payload") or {}
        other_question = str(payload.get("question", ""))
        other_purpose = str(payload.get("purpose", ""))
        if not other_question:
            continue
        if normalized_question == " ".join(re.findall(r"[A-Za-z0-9가-힣]+", other_question.lower())):
            return event
        other = _intent_terms(f"{other_question} {other_purpose}")
        union = target | other
        similarity = len(target & other) / len(union) if union else 0.0
        if similarity >= threshold:
            return event
    return None


def create_auto_exploration_intent_for_rq(
    ledger: Ledger, rq: dict[str, Any], *, score: int, selection_reason: str,
) -> dict[str, Any]:
    """Create a ready M2→M1 Intent directly, bypassing researcher approval in explicit auto mode."""
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    question = str(rq.get("question", "")).strip()
    rationale = str(rq.get("rationale", "")).strip()
    gap = str(rq.get("gap_or_tension", "")).strip()
    context = str(rq.get("research_context", "")).strip()
    need = str(rq.get("exploration_need", "")).strip()
    priority = "높음" if score >= 4 else "보통"
    # A newly submitted RQ must begin with its own clean first-round intent.
    # Follow-up rounds also intentionally create a new intent. Duplicate reuse is
    # reserved for older auto-dispatch paths where neither boundary applies.
    duplicate = (
        None
        if rq.get("_force_followup") or rq.get("_fresh_intake")
        else find_duplicate_curation_intent(ledger, question, need or rationale)
    )
    if duplicate:
        existing_payload = duplicate.get("payload") or {}
        existing_intent_id = str(existing_payload.get("intent_id") or duplicate.get("subject_id") or "")
        if existing_intent_id:
            ledger.link_research_question_intent(str(rq.get("rq_id", "")), existing_intent_id, "")
        return {
            "rq_id": str(rq.get("rq_id", "")), "question": question, "score": score,
            "selection_reason": selection_reason, "intent": CurationIntent.model_validate(existing_payload),
            "phenomenon_id": duplicate["phenomenon_id"], "case_id": duplicate["case_id"], "reused": True,
        }

    rq_id = str(rq.get("rq_id") or intent_id)
    source_card_ids = [str(value) for value in rq.get("source_card_ids", []) if str(value)]
    origin_links = [{
        "origin_type": "researcher_question", "origin_id": rq_id,
        "label": question[:100], "source_card_ids": source_card_ids,
    }]
    if source_card_ids:
        origin_links.append({
            "origin_type": "m2_knowledge", "origin_id": rq_id,
            "label": f"{question[:80]} 관련 지식 보완", "source_card_ids": source_card_ids,
        })
    followup_direction = str(rq.get("_followup_direction") or "").strip()
    intent_title = (
        f"추가 탐색 · {followup_direction[:45]}"
        if followup_direction else f"자동 RQ 탐색 · {question[:45]}"
    )
    intent = CurationIntent(
        intent_id=intent_id, title=intent_title,
        purpose=need or f"이 연구질문의 핵심 지식 공백을 선행연구에서 확인한다: {rationale}",
        question=question,
        research_context="\n".join(item for item in [
            f"Research question: {question}",
            f"Why this question emerged: {rationale}",
            f"Gap or tension: {gap}" if gap else "",
            f"Research context: {context}" if context else "",
            f"Auto selection reason: {selection_reason}" if selection_reason else "",
        ] if item),
        labels=[], priority=priority,
        expected_evidence=need or "질문의 전제, 반대 근거, 적용 조건, 관련 방법 및 사례를 확인할 수 있는 출처 기반 근거",
        completion_condition="질문의 핵심 공백에 대해 출처가 확인된 지식카드 또는 명시적인 미해결 지식 공백을 M2와 연구자에게 보고한다.",
        execution_mode="auto", created_by="m2_auto",
        origin_links=origin_links,
    )
    case_id = ledger.create_case("research", f"AUTO · {question[:72]}")
    phenomenon_id = ledger.record(
        case_id, "curation_intent", "m2", ["m1"], "curation_intent",
        intent.model_dump(mode="json"), subject_id=intent_id, status="ready",
    )
    ledger.create_search_profile(intent.model_dump(mode="json"))
    ledger.link_research_question_intent(str(rq.get("rq_id", "")), intent_id, "")
    return {
        "rq_id": str(rq.get("rq_id", "")), "question": question, "score": score,
        "selection_reason": selection_reason, "intent": intent, "phenomenon_id": phenomenon_id, "case_id": case_id, "reused": False,
    }


def dispatch_top_research_questions(
    ledger: Ledger, ranked: list[dict[str, Any]], *, limit: int = 3, review_id: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch selected RQs to M1 and leave one researcher-visible audit report."""
    selected = ranked[:limit]
    if review_id:
        for item in ranked:
            rq_id = str(item["rq"].get("rq_id", ""))
            is_selected = item in selected
            ledger.update_review_question_selection(
                review_id, rq_id, score=int(item["score"]),
                reason=str(item.get("reason", "")), selected=is_selected,
            )
            if is_selected:
                reason = str(item.get("reason", "")).strip()
                suffix = f" 선정 이유: {reason}" if reason else ""
                ledger.add_research_question_change(
                    rq_id, "priority_changed", f"이번 연구상태 검토에서 중요도 {int(item['score'])}/5로 평가되어 자동 후속 탐색 대상으로 선정되었습니다.{suffix}", review_id=review_id,
                )
    dispatched = [
        create_auto_exploration_intent_for_rq(
            ledger, item["rq"], score=int(item["score"]), selection_reason=str(item.get("reason", "")),
        )
        for item in selected
    ]
    if dispatched:
        case_id = ledger.create_case("research", "M2 자동 연구질문 탐색")
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "auto_exploration_dispatch",
            {
                "title": "M2 · 중요 연구질문 자동 탐색 보고",
                "report": "\n".join(
                    f"{idx}. [{item['score']}/5] {item['question']} → {item['intent'].title}"
                    for idx, item in enumerate(dispatched, 1)
                ),
                "selected": [
                    {
                        "rq_id": item["rq_id"], "question": item["question"], "score": item["score"],
                        "selection_reason": item["selection_reason"], "intent_id": item["intent"].intent_id,
                    }
                    for item in dispatched
                ],
            },
            status="completed",
        )
    return dispatched
