"""UI-independent application service for researcher decision interactions."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
from research_fellow.application.decisions import VALID_DECISIONS, decide_request
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


def pending_researcher_decision_requests(ledger: Ledger) -> list[dict[str, Any]]:
    """Return current boundary decision requests shown in the researcher approval queue.

    Legacy knowledge-card requests are intentionally excluded because current paper
    reading/curation paths no longer use that queue as their primary authoring flow.
    """
    return [
        item for item in ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
        if item.get("subject_type") != "knowledge_card"
    ]


def apply_researcher_decision_resolutions(
    ledger: Ledger,
    memory: KnowledgeMemory,
    relation_memory: RelationMemory,
    resolutions: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Apply semantic decision responses independently of the concrete UI renderer."""
    counts = {"approved": 0, "deferred": 0, "rejected": 0, "unchanged": 0}
    for resolution in resolutions:
        request_id = str(resolution.get("request_id") or "").strip()
        decision = str(resolution.get("decision") or "").strip()
        note = str(resolution.get("note") or "")
        if not request_id:
            raise ValueError("Decision resolution request_id is required")
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Unsupported researcher decision: {decision}")
        changed = decide_request(ledger, memory, request_id, decision, note, relation_memory)
        counts[decision if changed else "unchanged"] += 1
    return counts
