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
