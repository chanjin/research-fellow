from __future__ import annotations

import re
from typing import Any


SOURCE_PRIORITY = {
    "ontology_related_type": 1,
    "ontology_same_type": 2,
    "hybrid_search": 3,
    "inherited_rq": 4,
}


def paper_evidence_query(title: str, research_question: str) -> str:
    """Create the stable context used by lexical and embedding retrieval."""
    parts = [
        f"Paper title: {title.strip()}" if title.strip() else "",
        f"Research question: {research_question.strip()}" if research_question.strip() else "",
    ]
    return "\n".join(part for part in parts if part)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w가-힣]{2,}", value)}


def _card_text(card: dict[str, Any]) -> str:
    return " ".join(
        str(card.get(key, ""))
        for key in ("title", "claim", "context", "implication", "conditions", "limits")
    )


def assemble_paper_evidence_candidates(
    *,
    query: str,
    cards: list[dict[str, Any]],
    search_hits: list[dict[str, Any]],
    inherited_card_ids: list[str],
    seed_type_ids: set[str],
    type_names: dict[str, str],
    type_relations: list[dict[str, Any]],
    card_ids_by_type: dict[str, list[str]],
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Merge inherited, hybrid-search, and one-hop ontology candidates.

    Retrieval determines seed cards. The ontology expands only from their types,
    so graph traversal is deterministic, explainable, and bounded.
    """
    cards_by_id = {str(card.get("card_id", "")): card for card in cards}
    candidates: dict[str, dict[str, Any]] = {}

    def add(card_id: str, source: str, score: float, detail: str = "") -> None:
        if card_id not in cards_by_id:
            return
        item = candidates.setdefault(card_id, {
            "card_id": card_id,
            "source": source,
            "origins": [],
            "score": score,
            "ontology_paths": [],
        })
        if source not in item["origins"]:
            item["origins"].append(source)
        if detail and detail not in item["ontology_paths"]:
            item["ontology_paths"].append(detail)
        if SOURCE_PRIORITY[source] > SOURCE_PRIORITY[item["source"]]:
            item["source"] = source
        item["score"] = max(float(item["score"]), float(score))

    for card_id in dict.fromkeys(str(value) for value in inherited_card_ids if str(value)):
        add(card_id, "inherited_rq", 1000.0)
    for rank, hit in enumerate(search_hits):
        add(str(hit.get("card_id", "")), "hybrid_search", 900.0 - rank + float(hit.get("score") or 0.0))

    related_types: dict[str, list[str]] = {}
    for relation in type_relations:
        source_id = str(relation.get("source_type_id", ""))
        target_id = str(relation.get("target_type_id", ""))
        relation_name = str(relation.get("relation_name", "related to"))
        if source_id in seed_type_ids and target_id:
            related_types.setdefault(target_id, []).append(
                f"{type_names.get(source_id, source_id)} —{relation_name}→ {type_names.get(target_id, target_id)}"
            )
        if target_id in seed_type_ids and source_id:
            related_types.setdefault(source_id, []).append(
                f"{type_names.get(source_id, source_id)} —{relation_name}→ {type_names.get(target_id, target_id)}"
            )

    query_tokens = _tokens(query)
    for type_id in seed_type_ids:
        type_label = type_names.get(type_id, type_id)
        for card_id in card_ids_by_type.get(type_id, []):
            overlap = len(query_tokens & _tokens(_card_text(cards_by_id.get(card_id, {}))))
            add(card_id, "ontology_same_type", 500.0 + overlap, f"same type: {type_label}")
    for type_id, paths in related_types.items():
        for card_id in card_ids_by_type.get(type_id, []):
            overlap = len(query_tokens & _tokens(_card_text(cards_by_id.get(card_id, {}))))
            for path in paths:
                add(card_id, "ontology_related_type", 300.0 + overlap, path)

    inherited = [item for item in candidates.values() if item["source"] == "inherited_rq"]
    searched = [item for item in candidates.values() if item["source"] == "hybrid_search"]
    same_type = [item for item in candidates.values() if item["source"] == "ontology_same_type"]
    related_type = [item for item in candidates.values() if item["source"] == "ontology_related_type"]
    inherited.sort(key=lambda item: (-item["score"], item["card_id"]))
    for group in (searched, same_type, related_type):
        group.sort(key=lambda item: (-item["score"], item["card_id"]))
    selected = list(inherited)
    remaining = max(0, max_candidates - len(selected))
    # Direct retrieval stays the primary signal, but leave room for both kinds
    # of ontology expansion instead of allowing a large same-type cluster to
    # hide every relation-based candidate.
    ontology_reserve = min(6, remaining // 2) if same_type or related_type else 0
    search_take = min(len(searched), max(0, remaining - ontology_reserve))
    selected.extend(searched[:search_take])
    searched = searched[search_take:]
    remaining = max(0, max_candidates - len(selected))
    while remaining and (same_type or related_type):
        if same_type and remaining:
            selected.append(same_type.pop(0)); remaining -= 1
        if related_type and remaining:
            selected.append(related_type.pop(0)); remaining -= 1
    if remaining:
        selected.extend(searched[:remaining])
    return selected
