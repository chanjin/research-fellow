"""Research-question answer lifecycle with two-stage researcher-facing documents.

Stage 1 anchors the answer in the previous answer (when updating) plus evidence
collected specifically for the active research question. Stage 2 enriches that
first document with relevant prior workspace knowledge and literature. Both
LLM-backed stages remain external-manual under the current execution policy.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.llm_execution import execute_llm_stage
from research_fellow.application.llm_retry import LLMRetryExhausted
from research_fellow.application.run_tracking import ExecutionRunTracker
from research_fellow.origin_lineage import cards_for_origin
from research_fellow.application.research_evidence import rq_evidence_bundle
from research_fellow.storage import Ledger
from research_fellow.infrastructure.prompt_renderer import apply_review_language_policy

WORKFLOW_PATH = "m2/research_question_answer.yaml"
Draft = Callable[[str], str | None]


def relevant_cards_for_rq(cards: list[dict[str, Any]], rq_id: str) -> list[dict[str, Any]]:
    return cards_for_origin(cards, [rq_id])


def latest_research_answer(ledger: Ledger, rq_id: str) -> dict[str, Any] | None:
    items = [
        item for item in ledger.phenomena(type_="advice_report")
        if item.get("subject_type") == "research_question_response" and str(item.get("subject_id") or "") == rq_id
    ]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            int((item.get("payload") or {}).get("answer_version") or 0),
            str(item.get("created_at") or ""),
        ),
    )


def latest_research_answer_approval(ledger: Ledger, rq_id: str) -> dict[str, Any] | None:
    items = [
        item for item in ledger.phenomena(type_="advisory_exchange")
        if item.get("subject_type") == "research_answer_approval" and str(item.get("subject_id") or "") == rq_id
    ]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            int((item.get("payload") or {}).get("answer_version") or 0),
            str(item.get("created_at") or ""),
        ),
    )


def research_answer_is_approved(ledger: Ledger, rq_id: str) -> bool:
    latest = latest_research_answer(ledger, rq_id)
    approval = latest_research_answer_approval(ledger, rq_id)
    if latest is None or approval is None:
        return False
    return str((approval.get("payload") or {}).get("answer_id") or "") == str(latest.get("phenomenon_id") or "")


def save_research_answer_revision(ledger: Ledger, rq_id: str, markdown_text: str) -> dict[str, Any]:
    """Persist a researcher-edited Markdown answer as the next answer version."""
    text = str(markdown_text or "").strip()
    if not text:
        raise ValueError("답변 문서는 비어 있을 수 없습니다.")
    latest = latest_research_answer(ledger, rq_id)
    if latest is None:
        raise ValueError("수정할 연구질문 답변 Draft가 없습니다.")
    payload = dict(latest.get("payload") or {})
    if text == str(payload.get("report") or "").strip():
        return {"status": "unchanged", "answer_id": str(latest.get("phenomenon_id") or ""), "answer_version": int(payload.get("answer_version") or 1)}
    version = int(payload.get("answer_version") or 1) + 1
    previous_id = str(latest.get("phenomenon_id") or "")
    revised_payload = {
        **payload,
        "title": f"Research Question 답변 v{version}",
        "report": text,
        "answer_mode": "researcher_revision",
        "answer_version": version,
        "previous_answer_id": previous_id,
    }
    answer_id = ledger.record(
        str(latest.get("case_id") or ledger.create_case("research", f"Research answer revision: {rq_id}")),
        "advice_report", "researcher", ["m2"], "research_question_response", revised_payload,
        subject_id=rq_id, status="completed",
    )
    ledger.add_research_question_change(rq_id, "answer_draft_revised", f"연구자가 답변 Draft를 수정하여 v{version}으로 저장했습니다.")
    return {"status": "saved", "answer_id": answer_id, "answer_version": version}


def approve_research_answer_draft(
    ledger: Ledger,
    rq_id: str,
    *,
    markdown_text: str,
    next_question: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Approve the latest answer and optionally refine the RQ for the next round."""
    saved = save_research_answer_revision(ledger, rq_id, markdown_text)
    latest = latest_research_answer(ledger, rq_id)
    if latest is None:
        raise ValueError("승인할 연구질문 답변 Draft가 없습니다.")
    rq = ledger.research_question(rq_id) or {}
    current_question = str(rq.get("question") or "").strip()
    refined = str(next_question or "").strip()
    if refined and refined != current_question:
        ledger.refine_research_question(
            rq_id, refined,
            change_reason="승인된 Research Answer를 바탕으로 다음 문헌 라운드의 연구질문을 보완함",
        )
        current_question = refined
    payload = dict(latest.get("payload") or {})
    prior = latest_research_answer_approval(ledger, rq_id)
    if prior is not None and str((prior.get("payload") or {}).get("answer_id") or "") == str(latest.get("phenomenon_id") or ""):
        return {
            "status": "already_approved",
            "answer_id": str(latest.get("phenomenon_id") or ""),
            "answer_version": int(payload.get("answer_version") or 1),
            "question": current_question,
        }
    approved_intents = ledger.research_question_intents(rq_id)
    approved_intent_id = str((approved_intents[0] if approved_intents else {}).get("intent_id") or "")
    approval_id = ledger.record(
        str(latest.get("case_id") or ledger.create_case("research", f"Research answer approval: {rq_id}")),
        "advisory_exchange", "researcher", ["m2"], "research_answer_approval",
        {
            "rq_id": rq_id,
            "answer_id": str(latest.get("phenomenon_id") or ""),
            "answer_version": int(payload.get("answer_version") or 1),
            "approved_question": current_question,
            "approved_intent_id": approved_intent_id,
            "note": str(note or "").strip(),
        },
        subject_id=rq_id, status="completed",
    )
    ledger.add_research_question_change(
        rq_id, "answer_draft_approved",
        f"연구자가 Research Answer v{int(payload.get('answer_version') or 1)}을 승인했습니다. 다음 문헌 라운드 진입이 가능합니다.",
    )
    return {
        **saved,
        "status": "approved",
        "approval_id": approval_id,
        "answer_id": str(latest.get("phenomenon_id") or ""),
        "answer_version": int(payload.get("answer_version") or 1),
        "question": current_question,
    }


def evidence_fingerprint(bundle: dict[str, Any]) -> str:
    """Stable identity of the evidence set used to decide answer freshness."""
    payload = {
        "direct_cards": sorted(str(x.get("card_id") or "") for x in bundle.get("direct", {}).get("cards", []) if x.get("card_id")),
        "direct_papers": sorted(str(x.get("paper_id") or "") for x in bundle.get("direct", {}).get("papers", []) if x.get("paper_id")),
        "supporting_cards": sorted(str(x.get("card_id") or "") for x in bundle.get("supporting", {}).get("cards", []) if x.get("card_id")),
        "supporting_papers": sorted(str(x.get("paper_id") or "") for x in bundle.get("supporting", {}).get("papers", []) if x.get("paper_id")),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def answer_update_available(ledger: Ledger, cards: list[dict[str, Any]], rq_id: str) -> bool:
    latest = latest_research_answer(ledger, rq_id)
    if latest is None:
        return False
    previous = str((latest.get("payload") or {}).get("evidence_fingerprint") or "")
    current = evidence_fingerprint(rq_evidence_bundle(ledger, cards, rq_id))
    # Legacy answer reports did not persist a fingerprint. Treat them as stale so
    # the researcher can deliberately regenerate them under the current model.
    return not previous or previous != current


def ensure_answer_confirmation(ledger: Ledger, cards: list[dict[str, Any]], rq_id: str) -> str | None:
    rq = ledger.research_question(rq_id)
    if not rq:
        return None
    relevant = relevant_cards_for_rq(cards, rq_id)
    if not relevant:
        return None
    for item in ledger.phenomena(type_="decision_request"):
        if item.get("subject_type") == "research_answer_draft" and str(item.get("subject_id") or "") == rq_id and item.get("status") in {"proposed", "approved"}:
            return None
    if latest_research_answer(ledger, rq_id) is not None:
        return None
    case_id = ledger.create_case("research", f"Research answer draft: {str(rq.get('question') or '')[:72]}")
    payload = {
        "title": "현재 연구질문에 대한 지식이 업데이트되었습니다 · 답변 초안 작성 확인",
        "draft_request": {"rq_id": rq_id, "question": str(rq.get("question") or ""), "knowledge_card_ids": [c.get("card_id") for c in relevant], "mode": "initial"},
        "knowledge_card_count": len(relevant),
        "next_action": "승인하면 이번 연구질문에서 직접 확보한 근거로 1차 문서를 만들고, 기존 워크스페이스 지식과 문헌으로 2차 문서를 보강합니다.",
    }
    request_id = ledger.record(case_id, "decision_request", "m2", ["researcher"], "research_answer_draft", payload, subject_id=rq_id)
    ledger.add_research_question_change(rq_id, "knowledge_ready_for_answer", f"승인 지식 {len(relevant)}건이 연결되어 답변 초안 작성 여부를 연구자에게 확인합니다.")
    return request_id



def request_initial_research_answer(ledger: Ledger, cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    """Explicit researcher approval to draft the first answer for an RQ.

    Knowledge updates by themselves no longer create a Decision Attention item.
    The researcher starts drafting from Research; only the subsequent external
    LLM boundary may create Attention.
    """
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")
    if latest_research_answer(ledger, rq_id) is not None:
        raise ValueError("이미 연구질문 답변이 있습니다. 답변 업데이트를 사용하세요.")
    relevant = relevant_cards_for_rq(cards, rq_id)
    if not relevant:
        raise ValueError("답변 초안을 작성할 승인 지식이 아직 없습니다.")
    for event in ledger.phenomena(recipient="m2", type_="advisory_exchange", status="ready"):
        if event.get("subject_type") == "research_answer_draft_request" and str((event.get("payload") or {}).get("rq_id") or "") == rq_id:
            return {"status": "already_requested", "rq_id": rq_id, "request_id": event.get("phenomenon_id")}
    case_id = ledger.create_case("research", f"Research answer draft: {str(rq.get('question') or '')[:72]}")
    payload = {
        "rq_id": rq_id,
        "question": str(rq.get("question") or ""),
        "mode": "initial",
        "knowledge_card_ids": [c.get("card_id") for c in relevant if c.get("card_id")],
    }
    request_id = ledger.record(
        case_id, "advisory_exchange", "researcher", ["m2"],
        "research_answer_draft_request", payload, subject_id=rq_id, status="ready",
    )
    ledger.add_research_question_change(
        rq_id, "answer_draft_requested",
        f"연구자가 승인 지식 {len(relevant)}건을 바탕으로 최초 답변 초안 작성을 요청했습니다.",
    )
    return {"status": "ready", "rq_id": rq_id, "request_id": request_id}

def request_research_answer_update(ledger: Ledger, cards: list[dict[str, Any]], rq_id: str) -> dict[str, Any]:
    """Explicit researcher action to regenerate an answer after evidence changed."""
    rq = ledger.research_question(rq_id)
    if not rq:
        raise ValueError("Unknown research question")
    latest = latest_research_answer(ledger, rq_id)
    if latest is None:
        raise ValueError("업데이트할 기존 답변이 없습니다.")
    if not answer_update_available(ledger, cards, rq_id):
        raise ValueError("기존 답변 이후 새 문헌 또는 지식 변화가 없습니다.")
    for event in ledger.phenomena(recipient="m2", type_="advisory_exchange", status="ready"):
        if event.get("subject_type") == "research_answer_draft_request" and str((event.get("payload") or {}).get("rq_id") or "") == rq_id:
            return {"status": "already_requested", "rq_id": rq_id, "request_id": event.get("phenomenon_id")}
    case_id = ledger.create_case("research", f"Research answer update: {str(rq.get('question') or '')[:72]}")
    payload = {
        "rq_id": rq_id,
        "question": str(rq.get("question") or ""),
        "mode": "update",
        "previous_answer_id": str(latest.get("phenomenon_id") or ""),
    }
    request_id = ledger.record(case_id, "advisory_exchange", "researcher", ["m2"], "research_answer_draft_request", payload, subject_id=rq_id, status="ready")
    ledger.add_research_question_change(rq_id, "answer_update_requested", "새 문헌 또는 지식 변화가 확인되어 연구자가 답변 업데이트를 요청했습니다.")
    return {"status": "ready", "rq_id": rq_id, "request_id": request_id}


def execute_research_answer_draft(ledger: Ledger, memory_cards: list[dict[str, Any]], request_event: dict[str, Any], *, drafter: Draft, resume_run_id: str = "") -> dict[str, Any]:
    workflow = prepare_workflow_run(WORKFLOW_PATH, {"ledger": ledger, "memory_cards": memory_cards, "draft_request_event": request_event, "drafter": drafter, "resume_run_id": resume_run_id}, globals())
    try:
        return workflow.execute()
    except LLMRetryExhausted as error:
        context = workflow.context
        tracker = context.get("run_tracker")
        if tracker is not None:
            tracker.needs_attention(stage=error.stage, error_type=error.error_type, error_message=error.message, retry_count=error.attempts)
            tracker.record_llm_failure(error, context={"kind": "research_answer_draft", "rq_id": context.get("rq_id", ""), "answer_mode": context.get("answer_mode", "initial")}, intent_id=str(context.get("rq_id") or ""), item_key=error.item_key)
        return {"status": "needs_attention", "rq_id": context.get("rq_id", ""), "run_id": context.get("answer_run_id", ""), "workflow_id": "m2_research_question_answer", "stage": error.stage}


def _prepare_research_answer_context(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    event = context["draft_request_event"]
    payload = dict(event.get("payload") or {})
    rq_id = str(payload.get("rq_id") or event.get("subject_id") or "")
    rq = ledger.research_question(rq_id) or {}
    bundle = rq_evidence_bundle(ledger, list(context.get("memory_cards") or []), rq_id)
    previous = latest_research_answer(ledger, rq_id)
    previous_payload = dict((previous or {}).get("payload") or {})
    tracker = ExecutionRunTracker.start(ledger, run_id=str(context.get("resume_run_id") or ""), intent_id=rq_id, stage="research_answer_first_draft")
    context.update({
        "rq_id": rq_id,
        "research_question": str(rq.get("question") or payload.get("question") or ""),
        "answer_mode": str(payload.get("mode") or "initial"),
        "previous_answer": str(previous_payload.get("report") or ""),
        "previous_answer_id": str((previous or {}).get("phenomenon_id") or ""),
        "answer_cards": list(bundle["direct"]["cards"]),
        "answer_papers": list(bundle["direct"]["papers"]),
        "supporting_cards": list(bundle["supporting"]["cards"]),
        "supporting_papers": list(bundle["supporting"]["papers"]),
        "evidence_counts": dict(bundle.get("counts") or {}),
        "evidence_fingerprint": evidence_fingerprint(bundle),
        "run_tracker": tracker,
        "answer_run_id": tracker.run_id,
    })


def _draft_research_answer(context: dict[str, Any]) -> None:
    """Create the 1st document from previous answer + RQ-specific new evidence."""
    tracker: ExecutionRunTracker = context["run_tracker"]
    lines = [
        "You are M2 preparing the FIRST research document for an active research question. Write Korean.",
        "The first document must be anchored ONLY in evidence collected specifically for this research question.",
        "If a PREVIOUS ANSWER exists, preserve useful structure and claims but revise them when the new direct evidence requires it.",
        "Do not use unrelated prior workspace knowledge in this first document.",
        "Clearly distinguish findings, limitations, unresolved gaps, and source attribution. Do not invent citations.",
        f"Research question: {context.get('research_question','')}",
        "",
        "PREVIOUS ANSWER (may be empty):",
        str(context.get("previous_answer") or "(none)"),
        "",
        "NEW DIRECT KNOWLEDGE FROM THIS RQ:",
    ]
    for card in context.get("answer_cards") or []:
        lines.append(f"- [{card.get('card_id')}] {card.get('title')}: {card.get('claim')} | limits={card.get('limits','')}")
    lines.append("\nNEW DIRECT PAPERS FROM THIS RQ:")
    for paper in context.get("answer_papers") or []:
        lines.append(f"- {paper.get('title')} | {paper.get('source_url','')} | {str(paper.get('summary') or '')[:1400]}")
    lines.append("\nReturn the complete FIRST DOCUMENT only.")
    prompt = apply_review_language_policy("\n".join(lines))
    stage = "research_answer_first_draft"
    manual = tracker.manual_override(stage=stage)
    result = execute_llm_stage(context["drafter"], prompt, stage=stage, parser=lambda v: v.strip(), accept=lambda v: len(v) >= 80, manual_response=manual)
    # Keep the manual override until the whole two-stage workflow completes.
    # WorkflowRun currently restarts from step 1 after an external-manual boundary,
    # so the first document must remain replayable while the enrichment stage waits.
    tracker.enter(stage, retry_count=result.attempts - 1)
    context["first_answer_document"] = result.value


def _enrich_research_answer(context: dict[str, Any]) -> None:
    """Create the 2nd/final document by enriching the first with prior workspace assets."""
    tracker: ExecutionRunTracker = context["run_tracker"]
    if not (context.get("supporting_cards") or context.get("supporting_papers")):
        context["answer_draft"] = str(context.get("first_answer_document") or "")
        tracker.enter("research_answer_enrichment", retry_count=0)
        return
    lines = [
        "You are M2 completing the SECOND and final researcher-facing document. Write Korean.",
        "Start from the FIRST DOCUMENT below. Preserve its RQ-specific evidence as the backbone.",
        "Use SUPPORTING EXISTING KNOWLEDGE and SUPPORTING EXISTING LITERATURE only to strengthen, compare, qualify, challenge, or contextualize the first document.",
        "Never imply that prior workspace material was collected in the current literature round.",
        "Do not replace strong direct evidence with weaker prior material. Keep provenance and limitations clear. Do not invent citations.",
        f"Research question: {context.get('research_question','')}",
        "",
        "FIRST DOCUMENT:",
        str(context.get("first_answer_document") or ""),
        "",
        "SUPPORTING EXISTING KNOWLEDGE:",
    ]
    for card in context.get("supporting_cards") or []:
        lines.append(f"- [{card.get('card_id')}] {card.get('title')}: {card.get('claim')} | limits={card.get('limits','')} | reason={card.get('support_reason','')}")
    lines.append("\nSUPPORTING EXISTING LITERATURE:")
    for paper in context.get("supporting_papers") or []:
        lines.append(f"- {paper.get('title')} | {paper.get('source_url','')} | {str(paper.get('summary') or '')[:1400]} | reason={paper.get('support_reason','')}")
    lines.append("\nReturn the complete SECOND/FINAL DOCUMENT only.")
    prompt = apply_review_language_policy("\n".join(lines))
    stage = "research_answer_enrichment"
    manual = tracker.manual_override(stage=stage)
    result = execute_llm_stage(context["drafter"], prompt, stage=stage, parser=lambda v: v.strip(), accept=lambda v: len(v) >= 80, manual_response=manual)
    tracker.enter(stage, retry_count=result.attempts - 1)
    context["answer_draft"] = result.value


def _publish_research_answer_draft(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    event = context["draft_request_event"]
    rq_id = str(context.get("rq_id") or "")
    first_document = str(context.get("first_answer_document") or "")
    final_document = str(context.get("answer_draft") or "")
    existing = [x for x in ledger.phenomena(type_="advice_report") if x.get("subject_type") == "research_question_response" and str(x.get("subject_id") or "") == rq_id]
    version = len(existing) + 1
    phenomenon_id = ledger.record(event["case_id"], "advice_report", "m2", ["researcher"], "research_question_response", {
        "title": f"Research Question 답변 v{version}",
        "report": final_document,
        "first_document": first_document,
        "rq_id": rq_id,
        "question": context.get("research_question", ""),
        "answer_mode": context.get("answer_mode", "initial"),
        "answer_version": version,
        "previous_answer_id": context.get("previous_answer_id", ""),
        "evidence_fingerprint": context.get("evidence_fingerprint", ""),
        "evidence_card_ids": [c.get("card_id") for c in context.get("answer_cards") or []],
        "paper_ids": [p.get("paper_id") for p in context.get("answer_papers") or []],
        "supporting_card_ids": [c.get("card_id") for c in context.get("supporting_cards") or []],
        "supporting_paper_ids": [p.get("paper_id") for p in context.get("supporting_papers") or []],
        "evidence_counts": dict(context.get("evidence_counts") or {}),
    }, subject_id=rq_id, status="completed")
    if event.get("status") == "ready":
        ledger.transition(event["phenomenon_id"], "ready", "completed")
    context["run_tracker"].clear_manual_override(stage="research_answer_first_draft")
    context["run_tracker"].clear_manual_override(stage="research_answer_enrichment")
    context["run_tracker"].complete()
    ledger.add_research_question_change(rq_id, "answer_draft_ready", f"직접 근거 기반 1차 문서와 기존 연구자산으로 보강한 2차 답변 문서 v{version}이 준비되었습니다.")
    context.update({"status": "completed", "answer_draft_id": phenomenon_id, "answer_version": version})
