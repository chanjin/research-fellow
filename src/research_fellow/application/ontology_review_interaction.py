"""UI-independent handling of researcher feedback on ontology change proposals."""
from __future__ import annotations
from typing import Any, Mapping
from research_fellow.storage import Ledger


def apply_ontology_review_feedback(ledger: Ledger, feedback: Mapping[str, Any]) -> dict[str, Any]:
    review_id = str(feedback.get("review_id") or "").strip()
    if not review_id:
        raise ValueError("Ontology review feedback review_id is required")
    combined = str(feedback.get("combined_comment") or "").strip()
    ledger.update_ontology_change_review_comment(review_id, combined)
    return {
        "review_id": review_id,
        "combined_comment": combined,
        "regenerate_requested": bool(feedback.get("regenerate_requested")),
    }


def apply_ontology_review_with_regeneration(
    ledger: Ledger,
    memory: Any,
    feedback: Mapping[str, Any],
    *,
    draft_fn: Any | None = None,
) -> dict[str, Any]:
    """Store ontology feedback and, when requested, create a revised proposal.

    Regeneration is an application concern rather than a Streamlit concern.  The
    original review remains durable; a revised review is created as a new proposal.
    """
    result = apply_ontology_review_feedback(ledger, feedback)
    if not result.get("regenerate_requested"):
        return result
    if draft_fn is None:
        raise ValueError("Ontology regeneration requires a draft function")

    from research_fellow.application.ontology_evolution import ontology_delta_prompt, parse_ontology_delta

    review_id = str(result["review_id"])
    review = next((item for item in ledger.ontology_change_reviews(limit=200) if str(item.get("review_id")) == review_id), None)
    if review is None:
        raise ValueError("Ontology change review no longer exists")
    source_card_ids = [str(item) for item in (review.get("source_card_ids") or [])]
    cards_by_id = {str(card.get("card_id")): card for card in memory.all()}
    review_cards = [cards_by_id[card_id] for card_id in source_card_ids if card_id in cards_by_id]
    current_types = ledger.ontology_types()
    current_relations = ledger.ontology_type_relations()
    prompt = ontology_delta_prompt(
        cards=review_cards,
        facets=ledger.ontology_facets(),
        types=current_types,
        relations=current_relations,
        comment=str(result.get("combined_comment") or ""),
    )
    response = draft_fn(prompt)
    if not response:
        raise ValueError("Ontology revision draft returned no response")
    revised = parse_ontology_delta(str(response), cards=review_cards, types=current_types)
    new_review = ledger.create_ontology_change_review(
        source_card_ids=source_card_ids,
        proposal=revised,
        base_version_id=review.get("base_version_id"),
    )
    ledger.update_ontology_change_review_comment(
        str(new_review["review_id"]), str(result.get("combined_comment") or "")
    )
    return {**result, "revised_review_id": str(new_review["review_id"])}
