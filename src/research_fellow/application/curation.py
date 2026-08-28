from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from research_fellow.domain.knowledge import KnowledgeCard, parse_text_draft
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


def _normalise_for_evidence_match(text: str) -> str:
    return re.sub(r"[^\w가-힣]+", " ", text.lower()).strip()


def _evidence_is_grounded(excerpt: str, source_text: str) -> bool:
    """Accept a literal excerpt, or a near-literal bounded fragment only.

    Evidence is deliberately not translated or summarised here.  The card
    claim may be a Korean M1 interpretation; its evidence must remain a
    traceable source fragment.
    """
    excerpt_normalized = _normalise_for_evidence_match(excerpt)
    source_normalized = _normalise_for_evidence_match(source_text)
    if len(excerpt_normalized) < 12 or not source_normalized:
        return False
    if excerpt_normalized in source_normalized:
        return True
    excerpt_terms = [term for term in excerpt_normalized.split() if len(term) >= 2]
    source_terms = set(source_normalized.split())
    return len(excerpt_terms) >= 4 and (sum(term in source_terms for term in excerpt_terms) / len(excerpt_terms)) >= 0.85


def normalize_candidate_draft(
    *, title: str, source_kind: str, page: ExtractedPage, labels: list[str], index: int, text_draft: str = "", source_text: str | None = None,
) -> CandidateDraftResult:
    """Normalize a readable M1/LLM draft using only document-derived fallback evidence.

    The normalizer is intentionally conservative: it never invents an excerpt,
    page number, or citation. Missing interpretive fields receive a visible
    warning and a bounded default, so the researcher can request a correction.
    """
    fields = parse_text_draft(text_draft)
    warnings: list[str] = []
    evidence_source = source_text or page.text
    excerpt = fields.get("evidence_excerpt", evidence_source[:1200]).strip()
    if not fields.get("evidence_excerpt"):
        warnings.append("근거 발췌가 초안에 없어 원문 해당 쪽의 발췌로 보완했습니다.")
    elif not _evidence_is_grounded(excerpt, evidence_source):
        excerpt = evidence_source[:1200].strip()
        warnings.append("초안의 근거가 원문 발췌로 확인되지 않아 추출 원문 발췌로 교체했습니다.")
    claim = fields.get("claim", "").strip()
    if not claim:
        claim = _first_claim(excerpt)
        warnings.append("주장이 초안에 없어 근거 첫 문장을 후보 주장으로 사용했습니다. 승인 전 보완을 권장합니다.")
    card = KnowledgeCard(
        card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}",
        title=fields.get("title", "").strip() or _compact_title(claim),
        source_kind=SOURCE_KINDS.get(source_kind, source_kind),
        claim=claim,
        labels=[part.strip() for part in fields.get("labels", ",".join(labels)).split(",") if part.strip()],
        evidence_excerpt=excerpt,
        evidence_pages=[],
        citation_markers=[],
        conditions=fields.get("conditions", "The researcher should review the full source context and scope of application."),
        limits=fields.get("limits", "This candidate card does not replace the source document's full conclusion."),
        provenance={"source_name": title},
    )
    return CandidateDraftResult(card=card.model_dump(mode="json"), normalized_from="text_draft" if text_draft.strip() else "document_fallback", warnings=warnings)


def curation_text_prompt(title: str, source_kind: str, pages: list[ExtractedPage]) -> str:
    """Make one small-context claim-discovery prompt, not a page-number task."""
    selected, used = [], 0
    for page in pages:
        remaining = 14_000 - used
        if remaining <= 0:
            break
        selected.append(page.text[:remaining])
        used += len(selected[-1])
    return render_prompt("m1_curation.j2", document_title=title, source_kind=source_kind, source_text="\n\n".join(selected), max_cards=3)


def create_document_candidates(
    ledger: Ledger, title: str, source_kind: str, pages: list[ExtractedPage], labels: list[str], text_draft: str = "",
) -> tuple[list[str], list[str]]:
    """Offer up to three independently decidable knowledge cards to the researcher."""
    case_id = ledger.create_case("research", f"지식화: {title}")
    useful_pages = [page for page in pages if page.text.strip()][:3]
    if not useful_pages:
        raise ValueError("추출 가능한 텍스트가 없습니다.")
    if text_draft.strip().upper() == "NO_CANDIDATE":
        return [], ["LLM이 제공된 범위에서 논문 본문 지식으로 쓸 만한 내용을 찾지 못했습니다."]
    request_ids, warnings = [], []
    drafts = [block.strip() for block in re.split(r"(?m)^---+\s*$", text_draft) if block.strip()]
    source_text = "\n\n".join(page.text for page in useful_pages)
    candidate_drafts = drafts or [""]
    for index, draft in enumerate(candidate_drafts[:3], 1):
        page = useful_pages[min(index - 1, len(useful_pages) - 1)]
        result = normalize_candidate_draft(
            title=title, source_kind=source_kind, page=page, labels=labels, index=index,
            text_draft=draft, source_text=source_text,
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
