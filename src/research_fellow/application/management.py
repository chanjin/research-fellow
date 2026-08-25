from __future__ import annotations

from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


def delete_knowledge_card(ledger: Ledger, memory: KnowledgeMemory, card_id: str, note: str = "") -> bool:
    """Logically delete a card while preserving append-only evidence and audit history."""
    if not memory.remove(card_id, note):
        return False
    case_id = ledger.create_case("knowledge_maintenance", f"지식 카드 삭제: {card_id}")
    ledger.record(
        case_id, "knowledge_update", "researcher", ["m1", "m2"], "knowledge_card",
        {"title": "지식 카드 삭제", "operation": "deleted", "card_id": card_id, "note": note}, card_id, status="completed",
    )
    return True


def delete_knowledge_relation(ledger: Ledger, relations: RelationMemory, relation_id: str, note: str = "") -> bool:
    """Logically delete a relation while retaining a recoverable audit record."""
    if not relations.remove(relation_id, note):
        return False
    case_id = ledger.create_case("knowledge_maintenance", f"지식 관계 삭제: {relation_id}")
    ledger.record(
        case_id, "knowledge_update", "researcher", ["m1", "m2"], "knowledge_relation",
        {"title": "지식 관계 삭제", "operation": "deleted", "relation_id": relation_id, "note": note}, relation_id, status="completed",
    )
    return True
