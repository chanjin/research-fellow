"""Researcher submission boundary for progressive curation candidates."""
from __future__ import annotations
from research_fellow.infrastructure.document_reader import ExtractedDocument
from research_fellow.storage import Ledger
from research_fellow.application.progressive_curation_models import ProgressiveCandidate, ConsolidationDecision

def submit_progressive_candidates(
    ledger: Ledger, document: ExtractedDocument, candidates: list[ProgressiveCandidate], decisions: list[ConsolidationDecision], warnings: list[str],
) -> list[str]:
    """Create researcher decision requests only; no candidate enters JSONL here."""
    case_id = ledger.create_case("research", f"점진적 지식화: {document.title}")
    request_ids = []
    for candidate in candidates:
        request_ids.append(ledger.record(
            case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
            {
                "title": f"지식 카드 승인: {candidate.card['title']}", "card": candidate.card,
                "normalization": "progressive_page_curation", "warnings": candidate.warnings,
                "next_action": "승인 시 의미 기억에 저장하고 M2에 통지; 보류 시 보완을 요청",
            },
            subject_id=candidate.candidate_id,
        ))
    return request_ids
