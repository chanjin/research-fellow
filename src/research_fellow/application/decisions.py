from __future__ import annotations

from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


VALID_DECISIONS = {"approved", "deferred", "rejected"}


def decide_request(
    ledger: Ledger, memory: KnowledgeMemory, request_id: str, decision: str, note: str = "", relation_memory: RelationMemory | None = None,
) -> bool:
    """Apply one researcher decision once and emit its deterministic consequences.

    Returns ``True`` only when this call changed the ledger. Repeated submissions
    are harmless and never append a second knowledge card or execution intent.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"Unsupported researcher decision: {decision}")
    request = ledger.phenomenon(request_id)
    if request is None:
        raise ValueError(f"Unknown decision request: {request_id}")
    if request["phenomenon_type"] != "decision_request":
        raise ValueError("Only decision_request phenomena can be decided.")
    if request["status"] != "proposed":
        return False
    if decision == "approved" and request["subject_type"] == "knowledge_relation" and relation_memory is None:
        raise ValueError("관계 승인에는 RelationMemory가 필요합니다.")
    if decision == "approved" and request["subject_type"] == "knowledge_relation":
        relation = request["payload"]["relation"]
        active_cards = {card["card_id"] for card in memory.all()}
        if relation["source_card_id"] not in active_cards or relation["target_card_id"] not in active_cards:
            raise ValueError("삭제된 지식 카드를 참조하는 관계는 승인할 수 없습니다.")

    # The transition is guarded in SQLite; another click/process can only win once.
    if not ledger.transition(request_id, "proposed", decision):
        return False
    ledger.add_decision(request_id, decision, note)
    ledger.record(
        request["case_id"], "decision", "researcher", [request["producer"]],
        request["subject_type"],
        {"request_id": request_id, "decision": decision, "note": note},
        request["subject_id"], status="completed",
    )
    if decision != "approved":
        return True

    payload = request["payload"]
    if request["subject_type"] == "knowledge_card":
        card = memory.add(payload["card"])
        ledger.record(
            request["case_id"], "knowledge_update", "m1", ["m2", "researcher"],
            "knowledge_card",
            {"title": f"승인 지식 추가: {card['title']}", "card_id": card["card_id"], "labels": card.get("labels", [])},
            card["card_id"], status="completed",
        )
    elif request["subject_type"] == "curation_intent":
        ledger.record(
            request["case_id"], "curation_intent", "m2", ["m1"], "curation_intent",
            payload["intent"], request["subject_id"], status="ready",
        )
        ledger.create_search_profile(payload["intent"])
    elif request["subject_type"] == "knowledge_relation":
        assert relation_memory is not None
        relation = relation_memory.add(payload["relation"])
        ledger.upsert_knowledge_relation(relation)
        ledger.record(
            request["case_id"], "knowledge_update", "m1", ["m2", "researcher"], "knowledge_relation",
            {"title": f"승인 관계 추가: {relation['relation_type']}", "relation_id": relation["relation_id"],
             "source_card_id": relation["source_card_id"], "target_card_id": relation["target_card_id"]},
            relation["relation_id"], status="completed",
        )
    return True
