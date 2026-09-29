"""Knowledge-card construction and researcher-submission capabilities."""
from __future__ import annotations
import uuid
from typing import Callable
from research_fellow.infrastructure.document_reader import ExtractedDocument, ExtractedPage
from research_fellow.storage import Ledger
from research_fellow.application.claim_curation_prompts import qualification_prompt

def build_simple_claim_cards(
    document: ExtractedDocument, source_kind: str, candidates: list[CandidateClaim], default_labels: list[str],
) -> list[dict[str, object]]:
    """Build cards from one LLM response without an extra semantic conversion."""
    from research_fellow.application.curation import SOURCE_KINDS
    from research_fellow.domain.knowledge import KnowledgeCard

    cards = []
    for candidate in candidates:
        labels = list(dict.fromkeys([*candidate.labels, *default_labels]))[:5]
        card = KnowledgeCard(
            card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}", title=_compact_title(candidate.claim, limit=64),
            source_kind=SOURCE_KINDS.get(source_kind, source_kind), claim=candidate.claim,
            explanation=candidate.explanation, labels=labels,
            evidence_excerpt="", evidence_pages=[], citation_markers=[], conditions="", limits="",
            provenance={"source_name": document.file_name, "grounding": "source_document_only"},
        )
        cards.append(card.model_dump(mode="json"))
    return cards


def qualify_claims(
    document: ExtractedDocument, source_kind: str, labels: list[str], claims: list[str], draft_for: Callable[[str], str | None],
) -> tuple[list[dict[str, object]], list[str]]:
    """Apply evidence/condition constraints in bounded batches of three claims."""
    cards: list[dict[str, object]] = []
    warnings: list[str] = []
    source_text = _source_text(document)
    anchor = document.pages[0] if document.pages else ExtractedPage(1, source_text)
    for offset in range(0, len(claims), 3):
        batch = claims[offset : offset + 3]
        draft = draft_for(qualification_prompt(document, batch, source_kind))
        if not draft:
            warnings.append(f"Claims {offset + 1}-{offset + len(batch)}: no qualification draft was returned.")
            continue
        blocks = [part.strip() for part in re.split(r"(?m)^---+\s*$", draft) if part.strip()]
        for index, block in enumerate(blocks[: len(batch)], start=offset + 1):
            result = normalize_candidate_draft(
                title=document.file_name, source_kind=source_kind, page=anchor, labels=labels,
                index=index, text_draft=block, source_text=source_text,
            )
            cards.append(result.card)
            warnings.extend(f"Claim {index}: {warning}" for warning in result.warnings)
    return cards, warnings


def submit_claim_cards(
    ledger: Ledger, document: ExtractedDocument, cards: list[dict[str, object]], warnings: list[str],
    existing_cards: list[dict[str, object]] | None = None,
) -> list[str]:
    """Send non-authoritative candidate cards to the researcher approval inbox."""
    non_english = [card for card in cards if not _is_machine_english_card(card)]
    if non_english:
        warnings = [
            *warnings,
            f"기계용 지식카드 필드가 영어가 아닌 후보 {len(non_english)}건은 승인함에 보내지 않았습니다. 영문 주장·설명·레이블로 보정하세요.",
        ]
        cards = [card for card in cards if _is_machine_english_card(card)]
    if existing_cards:
        from research_fellow.application.duplicate_review import similar_approved_cards

        duplicate_matches = similar_approved_cards(cards, existing_cards)
        duplicate_ids = set(duplicate_matches)
        if duplicate_ids:
            warnings = [
                *warnings,
                f"기존 승인 지식과 유사한 자동 후보 {len(duplicate_ids)}건은 신규 카드 승인함에 보내지 않았습니다. 탐색 로그의 원문과 기존 카드를 비교해 근거 보강 여부를 검토하세요.",
            ]
            cards = [card for card in cards if str(card.get("card_id", "")) not in duplicate_ids]
    case_id = ledger.create_case("research", f"Claim-first curation: {document.title}")
    request_ids = []
    for card in cards:
        candidate_id = f"kc-candidate-{uuid.uuid4().hex[:12]}"
        candidate = {**card, "card_id": candidate_id}
        request_ids.append(ledger.record(
            case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
            {
                "title": f"Knowledge card approval: {candidate['title']}", "card": candidate,
                "normalization": "claim_first_curation", "warnings": warnings,
                "next_action": "On approval, save to semantic memory and notify M2.",
            }, subject_id=candidate_id,
        ))
    return request_ids


def _is_machine_english_card(card: dict[str, object]) -> bool:
    """Cards form M1's machine-readable context; source excerpts may stay original."""
    fields = ("title", "claim", "explanation", "conditions", "limits")
    values = [str(card.get(field, "")) for field in fields]
    values.extend(str(label) for label in card.get("labels", []) if isinstance(card.get("labels", []), list))
    return not any(any("가" <= character <= "힣" for character in value) for value in values)
