"""Jinja prompt entry points for future M1/M2 review tasks.

The functions in this module deliberately create small, reference-addressable
contexts.  They do not create IDs, write memory, or transition phenomena:
those remain deterministic application-service responsibilities.
"""

from __future__ import annotations

from typing import Any

from research_fellow.infrastructure.prompt_renderer import render_prompt


def _card_view(card: dict[str, Any], reference: str) -> dict[str, Any]:
    provenance = card.get("provenance", {})
    return {
        "reference": reference,
        "card_id": str(card.get("card_id", "")),
        "title": str(card.get("title", "")),
        "claim": str(card.get("claim", "")),
        "evidence_excerpt": str(card.get("evidence_excerpt", "")),
        "labels": list(card.get("labels", [])),
        "conditions": str(card.get("conditions", "")),
        "limits": str(card.get("limits", "")),
        "source_name": str(provenance.get("source_name", "미상")),
    }


def evidence_views(cards: list[dict[str, Any]], prefix: str = "E", limit: int = 12) -> list[dict[str, Any]]:
    """Provide only approved-card facts, each with an LLM-safe local reference."""
    return [_card_view(card, f"{prefix}{index}") for index, card in enumerate(cards[:limit], start=1)]


def claim_verification_prompt(assertion: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_claim_verification.j2", assertion=assertion, evidence=evidence_views(cards))


def gap_search_plan_prompt(gap: str, focus: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_gap_search_plan.j2", knowledge_gap=gap, focus=focus, existing_evidence=evidence_views(cards))


def source_triage_prompt(gap: str, sources: list[dict[str, Any]]) -> str:
    """Render only sources that Python has retrieved from an approved tool."""
    candidates = [
        {
            "reference": f"S{index}", "source_id": str(source.get("source_id", "")),
            "title": str(source.get("title", "")), "authors": list(source.get("authors", [])),
            "published": str(source.get("published", "")), "summary": str(source.get("summary", "")),
            "url": str(source.get("url", "")),
        }
        for index, source in enumerate(sources[:15], start=1)
    ]
    return render_prompt("m1_source_triage.j2", knowledge_gap=gap, source_candidates=candidates)


def lineage_review_prompt(topic: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_lineage_review.j2", topic=topic, knowledge=evidence_views(cards, prefix="K", limit=10))


def revalidation_review_prompt(reason: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_revalidation_review.j2", reason=reason, evidence=evidence_views(cards))


def knowledge_update_report_prompt(cards: list[dict[str, Any]], research_question: str = "") -> str:
    return render_prompt(
        "m2_knowledge_update_report.j2", research_question=research_question, evidence=evidence_views(cards)
    )
