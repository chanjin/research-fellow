"""Prompt builders for ontology curation."""
from __future__ import annotations
from typing import Any
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.application.ontology_curation_context import OntologyCurationContext
def type_suggestion_prompt(context: OntologyCurationContext) -> str:
    return render_prompt(
        "m1_ontology_type_suggestion.j2",
        card=context.card,
        similar_cards=context.similar_cards,
        facets=context.existing_facets,
        types=context.existing_types,
        source_paper=context.source_paper,
        source_analysis=context.source_analysis,
    )


def relation_suggestion_prompt(
    approved_types: list[dict[str, Any]],
    all_types: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    card: dict[str, Any],
) -> str:
    return render_prompt(
        "m1_ontology_relation_suggestion.j2",
        card=card,
        approved_types=approved_types,
        types=all_types,
        relations=relations,
    )
