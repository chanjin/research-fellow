"""Page-level candidate generation capabilities for progressive curation."""
from __future__ import annotations
import re
from typing import Callable
from research_fellow.infrastructure.candidate_draft_cache import CandidateDraftCache
from research_fellow.infrastructure.document_reader import ExtractedDocument, ExtractedPage
from research_fellow.application.progressive_curation_models import ProgressiveCandidate

def page_curation_prompt(document_title: str, source_kind: str, page: ExtractedPage, max_cards: int = 2) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_page_curation.j2", document_title=document_title, source_kind=source_kind, page=page, max_cards=max_cards)


def generate_page_candidates(
    document: ExtractedDocument, source_kind: str, labels: list[str], model: str,
    draft_for: Callable[[str], str | None], cache: CandidateDraftCache, pages: list[ExtractedPage] | None = None,
    on_page: Callable[[int, int, bool], None] | None = None,
) -> tuple[list[ProgressiveCandidate], list[str]]:
    """Call an LLM once per page, reusing cached non-authoritative drafts."""
    from research_fellow.infrastructure.prompt_templates import read_prompt_template
    from research_fellow.application.curation import normalize_candidate_draft

    template_source = read_prompt_template("m1_page_curation.j2") + f"\0{source_kind}\0{','.join(labels)}"
    selected_pages = pages or document.pages
    candidates: list[ProgressiveCandidate] = []
    warnings: list[str] = []
    for ordinal, page in enumerate(selected_pages, start=1):
        draft = cache.get(document_id=document.document_id, page_number=page.page_number, model=model, template_source=template_source)
        cache_hit = draft is not None
        if draft is None:
            draft = draft_for(page_curation_prompt(document.title, source_kind, page))
            if draft:
                cache.put(document_id=document.document_id, page_number=page.page_number, model=model, template_source=template_source, draft=draft)
        if on_page:
            on_page(ordinal, len(selected_pages), cache_hit)
        if not draft:
            warnings.append(f"구간 {ordinal}: LLM 초안을 만들지 못해 건너뛰었습니다.")
            continue
        if draft.strip().upper() == "NO_CANDIDATE":
            warnings.append(f"구간 {ordinal}: 논문 본문 지식으로 쓸 만한 내용을 찾지 못해 후보를 만들지 않았습니다.")
            continue
        blocks = [part.strip() for part in re.split(r"(?m)^---+\s*$", draft) if part.strip()][:2]
        for index, block in enumerate(blocks, start=1):
            result = normalize_candidate_draft(
                title=document.title, source_kind=source_kind, page=page, labels=labels, index=index, text_draft=block,
            )
            candidate_id = f"kc-candidate-{document.document_id.rsplit('-', 1)[-1]}-p{page.page_number:03d}-{index}"
            card = {**result.card, "card_id": candidate_id}
            candidates.append(ProgressiveCandidate(
                candidate_id=candidate_id, card=card, page_number=page.page_number,
                block_ids=[block.block_id for block in page.blocks], warnings=result.warnings,
            ))
            warnings.extend(f"구간 {ordinal} 후보 {index}: {warning}" for warning in result.warnings)
    return candidates, warnings
