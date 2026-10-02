"""Durable and inline per-paper first-review helpers for literature rounds."""
from __future__ import annotations

import uuid
from typing import Any, Mapping

from research_fellow.application.auto_literature import selected_paper_review_prompt, _parse_selected_paper_review, _knowledge_candidate_title
from research_fellow.domain.knowledge import KnowledgeCard
from research_fellow.storage import Ledger


def _profile_for_intent(ledger: Ledger, intent_id: str) -> dict[str, Any]:
    return next((dict(x) for x in ledger.search_profiles(include_deleted=True) if str(x.get("intent_id") or "") == intent_id), {})


def _research_question_for_intent(ledger: Ledger, intent_id: str) -> dict[str, Any]:
    linked = ledger.research_questions_for_intent(intent_id)
    return dict(linked[0]) if linked else {}


def ensure_paper_review_task(ledger: Ledger, *, intent_id: str, paper: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy/durable Input task path kept for existing workspaces.

    New candidate reviews normally run inline inside Attention > Reviews.  Older
    ready tasks remain actionable and are not migrated destructively.
    """
    paper_id = str(paper.get("paper_id") or "").strip()
    if not paper_id:
        raise ValueError("서재함 논문 ID가 없어 논문 리뷰 작업을 만들 수 없습니다.")
    existing = [x for x in ledger.phenomena(type_="research_task") if x.get("subject_type") == "paper_first_review" and str(x.get("subject_id") or "") == paper_id and str((x.get("payload") or {}).get("intent_id") or "") == intent_id and x.get("status") in {"ready", "completed"}]
    if existing:
        return existing[0]
    profile = _profile_for_intent(ledger, intent_id)
    prompt = selected_paper_review_prompt(profile, [dict(candidate)])
    linked = ledger.research_questions_for_intent(intent_id)
    rq_id = str((linked[0] if linked else {}).get("rq_id") or "")
    case_id = ledger.create_case("research", f"Paper first review: {str(paper.get('title') or '')[:72]}")
    task_id = ledger.record(
        case_id, "research_task", "m1", ["researcher"], "paper_first_review",
        {
            "title": f"논문 원문 리뷰: {paper.get('title') or 'Untitled paper'}",
            "task_type": "paper_first_review", "intent_id": intent_id, "rq_id": rq_id,
            "paper_id": paper_id, "paper": dict(candidate), "prompt": prompt,
            "expected_output": "JSON papers[] with executive_summary and source-grounded claims",
        },
        subject_id=paper_id, status="ready",
    )
    return ledger.phenomenon(task_id) or {"phenomenon_id": task_id}


def _existing_knowledge_requests(ledger: Ledger, *, intent_id: str, paper_id: str) -> list[dict[str, Any]]:
    rows = []
    for row in ledger.phenomena(type_="decision_request"):
        if str(row.get("subject_type") or "") != "knowledge_card":
            continue
        payload = dict(row.get("payload") or {})
        if str(payload.get("intent_id") or "") == intent_id and str(payload.get("paper_id") or "") == paper_id:
            rows.append(row)
    return rows


def apply_inline_paper_review_response(
    ledger: Ledger,
    *,
    intent_id: str,
    paper_id: str,
    candidate: Mapping[str, Any],
    response: str,
) -> dict[str, Any]:
    """Persist one paper summary and stage its Knowledge Card candidates.

    This is the canonical write path for both the new inline Review UI and the
    legacy durable paper_first_review task.  Re-submission is idempotent with
    respect to already-created knowledge decision requests for this paper/round.
    """
    profile = _profile_for_intent(ledger, intent_id)
    rq = _research_question_for_intent(ledger, intent_id)
    rq_text = str(rq.get("question") or profile.get("question") or "")
    reviewed = _parse_selected_paper_review(response, [dict(candidate)], research_question=rq_text)[0]
    shelf = ledger.shelf_paper(paper_id)
    if not shelf:
        raise ValueError("서재함 논문을 찾을 수 없습니다.")
    rq_id = str(rq.get("rq_id") or "")
    summary = str(reviewed.get("paper_summary") or "").strip()
    review_note = str(reviewed.get("review_note") or "").strip()
    # Research-question interpretation is canonical when the review belongs to an RQ.
    # Do not overwrite the legacy paper-level analysis: that slot is retained only
    # for old workspaces/unscoped reading flows and researcher-owned paper notes.
    if rq_id:
        ledger.save_paper_question_analysis(
            paper_id,
            research_question_id=rq_id,
            intent_id=intent_id,
            research_question=rq_text,
            summary=summary,
            reading_raw_output=str(reviewed.get("full_text_review") or response),
            generated=True,
        )
    else:
        ledger.save_paper_analysis(
            paper_id,
            research_question=rq_text,
            summary=summary,
            reading_raw_output=str(reviewed.get("full_text_review") or response),
            generated=True,
        )
    ledger.update_shelf_paper(
        paper_id,
        shelf_status=str(shelf.get("shelf_status") or "reference"),
        reading_status="read",
    )

    prior = _existing_knowledge_requests(ledger, intent_id=intent_id, paper_id=paper_id)
    request_ids = [str(row.get("phenomenon_id") or "") for row in prior]
    cards = [dict((row.get("payload") or {}).get("card") or {}) for row in prior if isinstance((row.get("payload") or {}).get("card"), Mapping)]
    if not prior:
        for claim in reviewed.get("knowledge_candidates") or []:
            claim_text = str(claim.get("claim") or "").strip()
            evidence = str(claim.get("evidence") or "").strip()
            quotes = [str(x).strip() for x in (claim.get("source_quotes") or []) if str(x).strip()][:3]
            evidence_detail = evidence
            if quotes:
                evidence_detail = "원문 근거 문장\n" + "\n".join(f"- {quote}" for quote in quotes) + "\n\n근거 요약\n" + evidence
            if len(claim_text) < 8 or len(evidence) < 8:
                continue
            card = KnowledgeCard(
                card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}",
                title=_knowledge_candidate_title(claim, claim_text),
                source_kind="external_paper",
                claim=claim_text,
                # The paper x RQ 1-page summary lives in paper_question_analyses.
                # Do not duplicate it into every knowledge-card candidate.
                context="",
                implication="",
                source_excerpt=evidence_detail[:3200],
                labels=list(profile.get("labels") or []),
                evidence_level="provisional",
                status="verified",
                evidence_excerpt=evidence_detail[:1600],
                conditions="",
                limits=str(claim.get("limits") or ""),
                provenance={
                    "source_name": str(shelf.get("title") or "paper"),
                    "paper_id": paper_id,
                    "research_question_id": rq_id,
                    "intent_id": intent_id,
                    "grounding": "m1_single_paper_review",
                },
                origin_links=list(shelf.get("origin_links") or []),
            ).model_dump(mode="json")
            case_id = ledger.create_case("research", f"Paper knowledge candidate: {str(shelf.get('title') or '')[:72]}")
            req = ledger.record(
                case_id,
                "decision_request",
                "m1",
                ["researcher"],
                "knowledge_card",
                {
                    "title": f"논문 지식카드 후보 승인: {card['title']}",
                    "card": card,
                    "paper_id": paper_id,
                    "intent_id": intent_id,
                    "rq_id": rq_id,
                    "research_question": rq_text,
                    "review_note": review_note,
                    "next_action": "승인 시 연구질문에 연결된 지식카드로 등록합니다.",
                },
                subject_id=card["card_id"],
            )
            request_ids.append(req)
            cards.append(card)
    return {
        "paper_id": paper_id,
        "summary": summary,
        "review_note": review_note,
        "knowledge_request_ids": request_ids,
        "knowledge_cards": cards,
        "reviewed": reviewed,
    }


def apply_paper_review_response(ledger: Ledger, task: Mapping[str, Any], response: str) -> dict[str, Any]:
    payload = dict(task.get("payload") or {})
    paper_id = str(payload.get("paper_id") or task.get("subject_id") or "")
    intent_id = str(payload.get("intent_id") or "")
    candidate = dict(payload.get("paper") or {})
    result = apply_inline_paper_review_response(
        ledger,
        intent_id=intent_id,
        paper_id=paper_id,
        candidate=candidate,
        response=response,
    )
    task_id = str(task.get("phenomenon_id") or "")
    if task_id and str(task.get("status") or "") == "ready":
        ledger.transition(task_id, "ready", "completed")
    return result
