from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from research_fellow.domain.knowledge import KnowledgeCard, page_markers, parse_text_draft
from research_fellow.infrastructure.document_reader import ExtractedPage
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.storage import Ledger


SOURCE_KINDS = {
    "외부 논문": "external_paper",
    "연구자의 확정 문서": "researcher_published_work",
    "연구자의 아이디어 노트": "researcher_idea_note",
}


@dataclass(frozen=True)
class CandidateDraftResult:
    card: dict[str, object]
    normalized_from: str
    warnings: list[str]


def _first_claim(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?。])\s+", compact)
    return next((sentence for sentence in sentences if len(sentence) >= 8), compact[:420])


def _compact_title(claim: str, limit: int = 42) -> str:
    """Create a concise, claim-centred title for lists and lineage views."""
    title = re.sub(r"\s+", " ", claim).strip()
    title = re.sub(r"[.。!?]+$", "", title)
    if len(title) <= limit:
        return title
    shortened = title[:limit].rsplit(" ", 1)[0]
    return (shortened or title[:limit]) + "…"


def normalize_candidate_draft(
    *, title: str, source_kind: str, page: ExtractedPage, labels: list[str], index: int, text_draft: str = "",
) -> CandidateDraftResult:
    """Normalize a readable M1/LLM draft using only document-derived fallback evidence.

    The normalizer is intentionally conservative: it never invents an excerpt,
    page number, or citation. Missing interpretive fields receive a visible
    warning and a bounded default, so the researcher can request a correction.
    """
    fields = parse_text_draft(text_draft)
    warnings: list[str] = []
    excerpt = fields.get("evidence_excerpt", page.text[:1200]).strip()
    if not fields.get("evidence_excerpt"):
        warnings.append("근거 발췌가 초안에 없어 원문 해당 쪽의 발췌로 보완했습니다.")
    claim = fields.get("claim", "").strip()
    if not claim:
        claim = _first_claim(excerpt)
        warnings.append("주장이 초안에 없어 근거 첫 문장을 후보 주장으로 사용했습니다. 승인 전 보완을 권장합니다.")
    raw_pages = fields.get("evidence_pages", "")
    pages = [int(number) for number in re.findall(r"\d+", raw_pages)] or [page.page_number]
    if not raw_pages:
        warnings.append("쪽수 표기가 없어 추출된 쪽수로 보완했습니다.")
    raw_markers = fields.get("citation_markers", "")
    markers = [part.strip() for part in raw_markers.split(",") if part.strip()] or page_markers(page.page_number, page.text)
    card = KnowledgeCard(
        card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}",
        title=fields.get("title", "").strip() or _compact_title(claim),
        source_kind=SOURCE_KINDS.get(source_kind, source_kind),
        claim=claim,
        labels=[part.strip() for part in fields.get("labels", ",".join(labels)).split(",") if part.strip()],
        evidence_excerpt=excerpt,
        evidence_pages=pages,
        citation_markers=markers,
        conditions=fields.get("conditions", "원문 전체 맥락과 적용 대상을 연구자가 검토해야 합니다."),
        limits=fields.get("limits", "이 카드는 원문 전체의 결론을 대체하지 않는 후보 지식입니다."),
        provenance={"source_name": title, "page_or_section": page.section or f"p.{page.page_number}"},
    )
    return CandidateDraftResult(card=card.model_dump(mode="json"), normalized_from="text_draft" if text_draft.strip() else "document_fallback", warnings=warnings)


def curation_text_prompt(title: str, source_kind: str, pages: list[ExtractedPage]) -> str:
    return render_prompt("m1_curation.j2", document_title=title, source_kind=source_kind, pages=pages[:3], max_cards=3)


def create_document_candidates(
    ledger: Ledger, title: str, source_kind: str, pages: list[ExtractedPage], labels: list[str], text_draft: str = "",
) -> tuple[list[str], list[str]]:
    """Offer up to three independently decidable knowledge cards to the researcher."""
    case_id = ledger.create_case("research", f"지식화: {title}")
    useful_pages = [page for page in pages if page.text.strip()][:3]
    if not useful_pages:
        raise ValueError("추출 가능한 텍스트가 없습니다.")
    request_ids, warnings = [], []
    drafts = [block.strip() for block in re.split(r"(?m)^---+\s*$", text_draft) if block.strip()]
    for index, page in enumerate(useful_pages, 1):
        result = normalize_candidate_draft(
            title=title, source_kind=source_kind, page=page, labels=labels, index=index,
            text_draft=drafts[index - 1] if index <= len(drafts) else "",
        )
        card = result.card
        warnings.extend(f"후보 {index}: {warning}" for warning in result.warnings)
        request_ids.append(ledger.record(
            case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
            {
                "title": f"지식 카드 승인: {card['title']}", "card": card,
                "normalization": result.normalized_from, "warnings": result.warnings,
                "next_action": "승인 시 의미 기억에 저장하고 M2에 통지; 보류 시 보완을 요청",
            },
            subject_id=str(card["card_id"]),
        ))
    return request_ids, warnings
