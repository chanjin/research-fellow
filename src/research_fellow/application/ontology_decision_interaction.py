"""UI-independent researcher decisions for ontology change reviews."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
from research_fellow.storage import Ledger


def pending_ontology_change_reviews(ledger: Ledger) -> list[dict[str, Any]]:
    return ledger.ontology_change_reviews(status="proposed")


def apply_ontology_change_resolutions(
    ledger: Ledger,
    resolutions: Iterable[Mapping[str, Any]],
    *,
    memory: Any | None = None,
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
        review = next((item for item in ledger.ontology_change_reviews(limit=200) if str(item.get("review_id") or "") == review_id), None)
        source_card_ids = [str(value) for value in ((review or {}).get("source_card_ids") or []) if str(value)]
        result = ledger.resolve_ontology_change_review(review_id, decision, note=note)
        counts[decision] += 1
        if decision == "approved":
            published_versions.append(result)
            if memory is not None:
                from research_fellow.application.ontology_workflow import ensure_type_graph_work, maybe_enqueue_facet_work
                # The approved batch already includes Type-graph relation proposals.
                # Refresh only if additional untyped cards remain; do not create per-card Type relation tasks.
                ensure_type_graph_work(ledger, memory)
                maybe_enqueue_facet_work(ledger)
    return {"counts": counts, "published_versions": published_versions}
