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


def facet_suggestion_prompt(
    *,
    types: list[dict[str, Any]],
    facets: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    max_facets: int = 20,
) -> str:
    return render_prompt(
        "m1_ontology_facet_suggestion.j2",
        types=types,
        facets=facets,
        relations=relations,
        max_facets=max(1, min(int(max_facets), 20)),
    )


def type_graph_suggestion_prompt(
    *,
    cards: list[dict[str, Any]],
    reference_cards: list[dict[str, Any]],
    types: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    knowledge_relations: list[dict[str, Any]],
) -> str:
    return render_prompt(
        "m1_ontology_type_graph_suggestion.j2",
        cards=cards,
        reference_cards=reference_cards,
        types=types,
        relations=relations,
        knowledge_relations=knowledge_relations,
    )
