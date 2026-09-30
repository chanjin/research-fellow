"""Job-centred knowledge workspace read model.

The Knowledge Workspace owns no knowledge state. It projects approved semantic
memory, evidence, relations, ontology structure, and recent knowledge-change
phenomena into one view of what the Research Fellow currently knows and where
structure or review is still needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class KnowledgeCardView:
    card_id: str
    title: str
    claim: str
    status: str
    evidence_level: str
    source_kind: str
    supporting_evidence_count: int
    relation_count: int
    ontology_types: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["ontology_types"] = list(self.ontology_types)
        return data


@dataclass(frozen=True)
class KnowledgeChangeView:
    update_id: str
    title: str
    created_at: str
    subject_id: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KnowledgeGapView:
    gap_id: str
    priority: str
    kind: str
    title: str
    reason: str
    target_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _relation_counts(relations: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for relation in relations:
        for key in (str(relation.get("source_card_id") or ""), str(relation.get("target_card_id") or "")):
            if key:
                result[key] = result.get(key, 0) + 1
    return result


def _card_views(
    memory: KnowledgeMemory,
    relation_memory: RelationMemory,
    ledger: Ledger,
    *,
    limit: int,
) -> tuple[list[KnowledgeCardView], list[dict[str, Any]]]:
    cards = memory.all(limit=limit)
    relations = relation_memory.all()
    counts = _relation_counts(relations)
    views: list[KnowledgeCardView] = []
    for card in cards:
        card_id = str(card.get("card_id") or "")
        ontology_names = tuple(
            str(item.get("name") or item.get("type_id") or "")
            for item in ledger.ontology_types_for_card(card_id)
            if item.get("name") or item.get("type_id")
        )
        views.append(KnowledgeCardView(
            card_id=card_id,
            title=str(card.get("title") or "Untitled knowledge card"),
            claim=str(card.get("claim") or ""),
            status=str(card.get("status") or "verified"),
            evidence_level=str(card.get("evidence_level") or "provisional"),
            source_kind=str(card.get("source_kind") or ""),
            supporting_evidence_count=len(card.get("supporting_evidence") or []),
            relation_count=counts.get(card_id, 0),
            ontology_types=ontology_names,
        ))
    return views, relations


def _recent_changes(ledger: Ledger, *, limit: int) -> list[KnowledgeChangeView]:
    reviewed = ledger.reviewed_knowledge_update_ids()
    result: list[KnowledgeChangeView] = []
    for row in ledger.phenomena(type_="knowledge_update")[:limit]:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        update_id = str(row.get("phenomenon_id") or "")
        result.append(KnowledgeChangeView(
            update_id=update_id,
            title=str(payload.get("title") or payload.get("claim") or payload.get("finding") or "Knowledge updated"),
            created_at=str(row.get("created_at") or ""),
            subject_id=str(row.get("subject_id") or payload.get("card_id") or ""),
            status="reviewed" if update_id in reviewed else "new",
            detail=str(payload.get("summary") or payload.get("change_summary") or ""),
        ))
    return result


def _knowledge_gaps(
    cards: list[KnowledgeCardView],
    *,
    pending_ontology_reviews: int,
    limit: int,
) -> list[KnowledgeGapView]:
    gaps: list[KnowledgeGapView] = []
    for card in cards:
        if card.status == "contested":
            gaps.append(KnowledgeGapView(
                gap_id=f"contested:{card.card_id}", priority="high", kind="conflict",
                title=f"Resolve contested knowledge: {card.title}",
                reason="The approved semantic memory marks this claim as contested.", target_id=card.card_id,
            ))
        if card.evidence_level == "provisional":
            gaps.append(KnowledgeGapView(
                gap_id=f"evidence:{card.card_id}", priority="medium", kind="evidence",
                title=f"Strengthen evidence: {card.title}",
                reason="The claim is still supported only at a provisional evidence level.", target_id=card.card_id,
            ))
        if not card.ontology_types:
            gaps.append(KnowledgeGapView(
                gap_id=f"ontology:{card.card_id}", priority="medium", kind="structure",
                title=f"Structure knowledge: {card.title}",
                reason="The card has not yet been assigned to an ontology type.", target_id=card.card_id,
            ))
        if card.relation_count == 0:
            gaps.append(KnowledgeGapView(
                gap_id=f"relation:{card.card_id}", priority="low", kind="structure",
                title=f"Connect knowledge: {card.title}",
                reason="The approved card is isolated from the current knowledge-relation graph.", target_id=card.card_id,
            ))
    if pending_ontology_reviews:
        gaps.append(KnowledgeGapView(
            gap_id="ontology-review", priority="high", kind="ontology_review",
            title=f"Review {pending_ontology_reviews} ontology change proposal(s)",
            reason="Proposed structural changes are waiting for researcher oversight.",
        ))
    rank = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda item: (rank.get(item.priority, 9), item.title.lower()))
    return gaps[:limit]


def knowledge_workspace_snapshot(
    memory: KnowledgeMemory,
    relation_memory: RelationMemory,
    ledger: Ledger,
    *,
    limit: int = 30,
) -> dict[str, Any]:
    """Return a read-only job view of approved knowledge and its structural health."""
    limit = max(1, min(int(limit), 100))
    cards, relations = _card_views(memory, relation_memory, ledger, limit=limit)
    changes = _recent_changes(ledger, limit=limit)
    ontology = ledger.ontology_snapshot()
    pending_reviews = ledger.ontology_change_reviews(status="proposed", limit=100)
    versions = ledger.ontology_versions(limit=5)
    gaps = _knowledge_gaps(cards, pending_ontology_reviews=len(pending_reviews), limit=limit)

    evidence_levels: dict[str, int] = {}
    statuses: dict[str, int] = {}
    source_kinds: dict[str, int] = {}
    for card in cards:
        evidence_levels[card.evidence_level] = evidence_levels.get(card.evidence_level, 0) + 1
        statuses[card.status] = statuses.get(card.status, 0) + 1
        source_kinds[card.source_kind] = source_kinds.get(card.source_kind, 0) + 1

    assigned_ids = {str(item.get("card_id") or "") for item in ontology.get("assignments", [])}
    typed_cards = sum(1 for card in cards if card.card_id in assigned_ids or card.ontology_types)
    reinforced = sum(1 for card in cards if card.supporting_evidence_count > 0)
    contradicted_relations = sum(1 for item in relations if str(item.get("relation_type") or "") == "contradicts")

    return {
        "counts": {
            "cards": len(cards),
            "relations": len(relations),
            "typed_cards": typed_cards,
            "untyped_cards": max(0, len(cards) - typed_cards),
            "reinforced_cards": reinforced,
            "contested_cards": statuses.get("contested", 0),
            "new_changes": sum(1 for item in changes if item.status == "new"),
            "gaps": len(gaps),
        },
        "evidence_levels": evidence_levels,
        "statuses": statuses,
        "source_kinds": source_kinds,
        "cards": [item.as_dict() for item in cards],
        "changes": [item.as_dict() for item in changes],
        "relations": {
            "total": len(relations),
            "contradictions": contradicted_relations,
            "by_type": _count_values(relations, "relation_type"),
        },
        "ontology": {
            "facets": len(ontology.get("facets", [])),
            "types": len(ontology.get("types", [])),
            "relations": len(ontology.get("relations", [])),
            "assignments": len(ontology.get("assignments", [])),
            "pending_reviews": len(pending_reviews),
            "latest_version": versions[0] if versions else None,
            "top_types": [
                {
                    "type_id": str(item.get("type_id") or ""),
                    "name": str(item.get("name") or ""),
                    "facet_name": str(item.get("facet_name") or ""),
                    "card_count": int(item.get("card_count") or 0),
                }
                for item in sorted(ontology.get("types", []), key=lambda row: int(row.get("card_count") or 0), reverse=True)[:12]
            ],
        },
        "gaps": [item.as_dict() for item in gaps],
    }


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        result[value] = result.get(value, 0) + 1
    return result
