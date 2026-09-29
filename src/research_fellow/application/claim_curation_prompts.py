"""Prompt builders for claim-first knowledge curation."""
from __future__ import annotations
from research_fellow.infrastructure.document_reader import ExtractedDocument

def discovery_prompt(document: ExtractedDocument, max_claims: int = 10) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_discovery.j2", source_text=_source_text(document), max_claims=max_claims)


def review_prompt(claims: list[str]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_consolidation.j2", claims=_claim_views(claims))


def label_prompt(claims: list[str]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_claim_labels.j2", claims=_claim_views(claims))


def qualification_prompt(document: ExtractedDocument, claims: list[str], source_kind: str) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt(
        "m1_claim_qualification.j2", document_title=document.title, source_kind=source_kind,
        source_text=_source_text(document), claims=_claim_views(claims),
    )


def _source_text(document: ExtractedDocument, limit: int = 14_000) -> str:
    return "\n\n".join(page.text for page in document.pages)[:limit]


def _claim_views(claims: list[str]) -> list[dict[str, str]]:
    return [{"claim_id": f"C{index}", "claim": claim} for index, claim in enumerate(claims, start=1)]
