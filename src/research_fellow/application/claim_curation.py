"""Minimal M1 curation for local reasoning models.

One bounded LLM call returns a claim, short explanation, and labels. No
function here writes semantic memory; researcher approval remains the gate.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Callable

from research_fellow.application.curation import _compact_title, normalize_candidate_draft
from research_fellow.infrastructure.document_reader import ExtractedDocument, ExtractedPage
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class ClaimReview:
    claim_id: str
    claim: str
    decision: str = "retain"
    merge_target: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class CandidateClaim:
    claim: str
    explanation: str = ""
    labels: tuple[str, ...] = ()


def discovery_prompt(document: ExtractedDocument, max_claims: int = 10) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_discovery.j2", source_text=_source_text(document), max_claims=max_claims)


def parse_candidate_claims(text: str, limit: int = 10) -> list[CandidateClaim]:
    """Parse the small text format without depending on LLM-controlled JSON."""
    # Local models often add Markdown and omit the requested item number.  A
    # Claim field is the reliable boundary in both cases.
    starts = list(re.finditer(r"(?im)^\s*(?:\d{1,2}[.)]\s*)?(?:\*{1,2}\s*)?claim\s*(?:\*{1,2})?\s*:", text))
    items = [text[match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)] for index, match in enumerate(starts)]
    if not items:
        items = re.split(r"(?m)^\s*(?:\d{1,2}[.)]|[-*])\s+", text.strip())
    cleaned: list[CandidateClaim] = []
    seen = set()
    for item in items:
        if not item.strip():
            continue
        fields = _fields(item)
        raw_claim = fields.get("claim") or item.splitlines()[0]
        claim = re.sub(r"\s+", " ", raw_claim).strip(" -•\t")
        explanation = re.sub(r"\s+", " ", fields.get("explanation", "")).strip()
        labels = tuple(label.strip() for label in fields.get("labels", "").split(",") if 1 < len(label.strip()) <= 48)[:3]
        key = re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
        if len(claim) < 20 or key in seen or _looks_like_metadata(claim):
            continue
        seen.add(key)
        cleaned.append(CandidateClaim(claim, explanation, labels))
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_discovered_claims(text: str, limit: int = 10) -> list[str]:
    """Compatibility helper for earlier callers."""
    return [candidate.claim for candidate in parse_candidate_claims(text, limit)]


def review_prompt(claims: list[str]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_consolidation.j2", claims=_claim_views(claims))


def label_prompt(claims: list[str]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_labels.j2", claims=_claim_views(claims))


def parse_label_suggestions(text: str, claims: list[str]) -> dict[str, list[str]]:
    """Parse only claim IDs and short labels; malformed output is harmless."""
    valid = {f"C{index}" for index in range(1, len(claims) + 1)}
    result = {claim_id: [] for claim_id in valid}
    for line in text.splitlines():
        match = re.match(r"^\s*(C\d+)\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match or match.group(1).upper() not in valid:
            continue
        labels = [item.strip() for item in match.group(2).split(",") if 1 < len(item.strip()) <= 48]
        result[match.group(1).upper()] = labels[:3]
    return result


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


def parse_claim_review(text: str, claims: list[str]) -> list[ClaimReview]:
    valid = {f"C{index}": claim for index, claim in enumerate(claims, start=1)}
    reviews: dict[str, ClaimReview] = {claim_id: ClaimReview(claim_id, claim) for claim_id, claim in valid.items()}
    for block in re.split(r"(?m)^---+\s*$", text):
        fields = _fields(block)
        claim_id = fields.get("claim")
        decision = fields.get("decision", "retain").lower()
        target = fields.get("merge with") or None
        if claim_id not in valid or decision not in {"retain", "merge", "discard"}:
            continue
        if decision == "merge" and target not in valid:
            continue
        reviews[claim_id] = ClaimReview(claim_id, valid[claim_id], decision, target, fields.get("reason", ""))
    return list(reviews.values())


def retained_claims(reviews: list[ClaimReview]) -> list[str]:
    merged = {review.claim_id for review in reviews if review.decision == "merge"}
    return [review.claim for review in reviews if review.decision == "retain" and review.claim_id not in merged]


def qualification_prompt(document: ExtractedDocument, claims: list[str], source_kind: str) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt(
        "m1_claim_qualification.j2", document_title=document.title, source_kind=source_kind,
        source_text=_source_text(document), claims=_claim_views(claims),
    )


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


def _source_text(document: ExtractedDocument, limit: int = 14_000) -> str:
    return "\n\n".join(page.text for page in document.pages)[:limit]


def _claim_views(claims: list[str]) -> list[dict[str, str]]:
    return [{"claim_id": f"C{index}", "claim": claim} for index, claim in enumerate(claims, start=1)]


def _fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = re.match(r"^\s*(?:[-#]\s*)?(?:\*{1,2}\s*)?([A-Za-z][A-Za-z ]*?)(?:\s*\*{1,2})?\s*:\s*(.*)$", line)
        if match:
            value = re.sub(r"^\*+\s*|\s*\*+$", "", match.group(2).strip())
            fields[match.group(1).strip().lower()] = value
    return fields


def _looks_like_metadata(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in ("creative commons", "all rights reserved", "ieee software", "http://", "https://"))


def _is_machine_english_card(card: dict[str, object]) -> bool:
    """Cards form M1's machine-readable context; source excerpts may stay original."""
    fields = ("title", "claim", "explanation", "conditions", "limits")
    values = [str(card.get(field, "")) for field in fields]
    values.extend(str(label) for label in card.get("labels", []) if isinstance(card.get("labels", []), list))
    return not any(any("가" <= character <= "힣" for character in value) for value in values)
