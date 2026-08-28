from __future__ import annotations

import uuid
from typing import Any

from .memory import KnowledgeMemory
from .storage import Ledger
from .application.decisions import decide_request
from .application.curation import create_document_candidates as _create_document_candidates
from .infrastructure.document_reader import ExtractedPage, extract_pages


def extract_text(uploaded_file: Any) -> str:
    """Compatibility wrapper. New code should use page-addressable ``extract_pages``."""
    return "\n".join(page.text for page in extract_pages(uploaded_file))


def create_document_candidate(
    ledger: Ledger,
    title: str,
    source_kind: str,
    source_text: str,
    labels: list[str],
    claim: str,
) -> list[str]:
    pages = [ExtractedPage(1, source_text)]
    request_ids, _ = create_document_candidates(ledger, title, source_kind, pages, labels, f"주장: {claim}")
    return request_ids


def create_document_candidates(
    ledger: Ledger, title: str, source_kind: str, pages: list[ExtractedPage], labels: list[str], text_draft: str = ""
) -> tuple[list[str], list[str]]:
    """Compatibility facade for the P1 curation application service."""
    return _create_document_candidates(ledger, title, source_kind, pages, labels, text_draft)


def request_curation_intent(ledger: Ledger, title: str, question: str, labels: list[str], priority: str) -> str:
    case_id = ledger.create_case("research", title)
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    # Compatibility entry point: make even a manually-created intent a complete
    # search contract rather than letting M1 search from a title and labels alone.
    intent = {
        "intent_id": intent_id,
        "title": title,
        "purpose": "Gather source-grounded evidence that directly informs the stated research question.",
        "question": question,
        "research_context": f"Research question: {question}",
        "labels": labels,
        "priority": priority,
        "expected_evidence": "Methods, results, limitations, and contrary evidence directly relevant to the research question.",
        "completion_condition": "Record source-grounded findings and any unresolved evidence gap for the stated research question.",
    }
    return ledger.record(
        case_id,
        "decision_request",
        "m2",
        ["researcher"],
        "curation_intent",
        {"title": f"M1 탐색 Intent 승인: {title}", "intent": intent, "next_action": "승인 시 M1 실행함으로 전달"},
        subject_id=intent_id,
    )


def complete_intent(ledger: Ledger, intent: dict[str, Any], finding: str) -> bool:
    """Close an approved M1 task once and publish one observable result."""
    if not ledger.transition(intent["phenomenon_id"], "ready", "completed"):
        return False
    ledger.record(
        intent["case_id"], "knowledge_update", "m1", ["m2", "researcher"], "curation_result",
        {"title": f"M1 탐색 결과: {intent['payload']['title']}", "finding": finding},
        intent["phenomenon_id"], status="completed",
    )
    return True


def create_external_case(ledger: Ledger, requester: str, question: str, context: str, interpretation: str) -> str:
    case_id = ledger.create_case("external_advisory", f"외부 자문: {requester}")
    ledger.record(
        case_id, "advisory_exchange", "external_requester", ["m2"], "advisory_request",
        {"requester": requester, "question": question, "context": context}, status="completed",
    )
    ledger.record(
        case_id, "advisory_exchange", "m2", ["external_requester"], "scope_interpretation",
        {"title": "M2 전문성 기반 요청 해석", "interpretation": interpretation}, status="proposed",
    )
    return case_id
