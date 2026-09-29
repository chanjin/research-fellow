"""Context assembly for ontology curation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from research_fellow.application.ontology import search_cards_for_ontology
from research_fellow.infrastructure.retrieval import KnowledgeRetriever
@dataclass(frozen=True)
class OntologyCurationContext:
    card: dict[str, Any]
    similar_cards: list[dict[str, Any]]
    existing_facets: list[dict[str, Any]]
    existing_types: list[dict[str, Any]]
    existing_relations: list[dict[str, Any]]
    source_paper: dict[str, Any] | None = None
    source_analysis: dict[str, Any] | None = None


def build_curation_context(
    retriever: KnowledgeRetriever,
    card: dict[str, Any],
    all_cards: list[dict[str, Any]],
    knowledge_relations: list[dict[str, Any]],
    ledger: Any,
    *,
    embedding_model: str,
    semantic: bool = True,
    limit: int = 10,
) -> OntologyCurationContext:
    """Build a local context around one untyped card.

    Similar cards are enriched with their approved Facet-Type assignments so the
    LLM first considers reuse before proposing new ontology concepts.
    """
    query = " ".join(
        part for part in [str(card.get("title", "")), str(card.get("claim", "")), " ".join(card.get("labels", []) or [])]
        if part.strip()
    )
    hits = search_cards_for_ontology(
        retriever,
        [item for item in all_cards if item.get("card_id") != card.get("card_id")],
        knowledge_relations,
        query,
        use_keyword=True,
        use_embedding=semantic,
        use_relations=True,
        embedding_model=embedding_model,
        limit=limit,
    )
    shelf_papers = ledger.shelf_papers()
    shelf_by_id = {str(p.get("paper_id")): p for p in shelf_papers if p.get("paper_id")}
    shelf_by_title = {str(p.get("title", "")).strip().casefold(): p for p in shelf_papers if str(p.get("title", "")).strip()}

    def _source_paper_for_card(card_item: dict[str, Any]) -> dict[str, Any] | None:
        provenance = card_item.get("provenance") or {}
        paper_id = str(provenance.get("paper_id") or "").strip()
        if paper_id and paper_id in shelf_by_id:
            return shelf_by_id[paper_id]
        source_name = str(provenance.get("source_name") or "").strip().casefold()
        return shelf_by_title.get(source_name) if source_name else None

    similar_cards: list[dict[str, Any]] = []
    for hit in hits:
        item = dict(hit.result.card)
        item["similarity_score"] = round(float(hit.result.score), 3)
        item["selection_reason"] = hit.result.reason
        item["relation_distance"] = hit.relation_distance
        item["ontology_types"] = ledger.ontology_types_for_card(str(item.get("card_id", "")))
        similar_paper = _source_paper_for_card(item)
        item["source_paper_title"] = str((similar_paper or {}).get("title") or (item.get("provenance") or {}).get("source_name") or "").strip()
        item["source_paper_labels"] = list((similar_paper or {}).get("labels") or [])
        similar_cards.append(item)
    provenance = card.get("provenance") or {}
    source_paper = _source_paper_for_card(card)
    source_analysis = ledger.paper_analysis(source_paper["paper_id"]) if source_paper else None
    return OntologyCurationContext(
        card=card,
        similar_cards=similar_cards,
        existing_facets=ledger.ontology_facets(),
        existing_types=ledger.ontology_types(),
        existing_relations=ledger.ontology_type_relations(),
        source_paper=source_paper,
        source_analysis=source_analysis,
    )
