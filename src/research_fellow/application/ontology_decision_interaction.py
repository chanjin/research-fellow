"""UI-independent researcher decisions for ontology change reviews."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
from research_fellow.storage import Ledger


def pending_ontology_change_reviews(ledger: Ledger) -> list[dict[str, Any]]:
    return ledger.ontology_change_reviews(status="proposed")


def apply_ontology_change_resolutions(
    ledger: Ledger,
    resolutions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = {"approved": 0, "deferred": 0, "rejected": 0}
    published_versions: list[dict[str, Any]] = []
    for resolution in resolutions:
        review_id = str(resolution.get("review_id") or "").strip()
        decision = str(resolution.get("decision") or "").strip()
        note = str(resolution.get("note") or "")
        if not review_id:
            raise ValueError("Ontology change resolution review_id is required")
        if decision not in counts:
            raise ValueError(f"Unsupported ontology change decision: {decision}")
        result = ledger.resolve_ontology_change_review(review_id, decision, note=note)
        counts[decision] += 1
        if decision == "approved":
            published_versions.append(result)
    return {"counts": counts, "published_versions": published_versions}
