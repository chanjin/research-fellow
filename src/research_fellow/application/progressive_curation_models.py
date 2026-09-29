"""Data structures for progressive page-level curation."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ProgressiveCandidate:
    candidate_id: str
    card: dict[str, object]
    page_number: int
    block_ids: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ConsolidationDecision:
    first_id: str
    second_id: str
    relation: str
    action: str
    reason: str
