from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from research_fellow.application.ontology import search_cards_for_ontology
from research_fellow.infrastructure.prompt_renderer import render_prompt
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


def type_suggestion_prompt(context: OntologyCurationContext) -> str:
    return render_prompt(
        "m1_ontology_type_suggestion.j2",
        card=context.card,
        similar_cards=context.similar_cards,
        facets=context.existing_facets,
        types=context.existing_types,
        source_paper=context.source_paper,
        source_analysis=context.source_analysis,
    )


def relation_suggestion_prompt(
    approved_types: list[dict[str, Any]],
    all_types: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    card: dict[str, Any],
) -> str:
    return render_prompt(
        "m1_ontology_relation_suggestion.j2",
        card=card,
        approved_types=approved_types,
        types=all_types,
        relations=relations,
    )


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM 응답이 비어 있습니다.")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start_candidates = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
        if not start_candidates:
            raise ValueError("JSON 응답을 찾을 수 없습니다.")
        start = min(start_candidates)
        end = max(raw.rfind("}"), raw.rfind("]"))
        if end <= start:
            raise ValueError("JSON 응답이 중간에서 잘린 것으로 보입니다.")
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON 형식을 해석할 수 없습니다: {error}") from error


def parse_type_suggestions(
    text: str,
    *,
    existing_facets: list[dict[str, Any]],
    existing_types: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        raise ValueError("타입 후보 응답은 JSON object여야 합니다.")
    recommendations = payload.get("recommendations") or []
    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError("recommendations가 하나 이상 필요합니다.")

    type_by_id = {str(x["type_id"]): x for x in existing_types}
    type_by_name = {str(x["name"]).casefold(): x for x in existing_types}
    facet_by_name = {str(x["name"]).casefold(): x for x in existing_facets}
    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []

    for idx, item in enumerate(recommendations[:8], start=1):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        if action not in {"assign_existing", "create_new", "needs_review"}:
            continue
        candidate = dict(item)
        if action == "assign_existing":
            found = None
            type_id = str(item.get("type_id") or "").strip()
            type_name = str(item.get("type") or item.get("type_name") or "").strip()
            if type_id:
                found = type_by_id.get(type_id)
            if found is None and type_name:
                found = type_by_name.get(type_name.casefold())
            if found is None:
                warnings.append(f"후보 {idx}: 기존 타입을 찾을 수 없어 needs_review로 전환했습니다.")
                candidate["action"] = "needs_review"
            else:
                candidate["type_id"] = found["type_id"]
                candidate["type"] = found["name"]
                candidate["facet"] = found.get("facet_name") or ""
                candidate["facet_id"] = found.get("facet_id")
        elif action == "create_new":
            name = str(item.get("type") or item.get("type_name") or "").strip()
            if not name:
                continue
            duplicate = type_by_name.get(name.casefold())
            if duplicate:
                warnings.append(f"'{name}'은(는) 이미 존재해 기존 타입 후보로 전환했습니다.")
                candidate.update({
                    "action": "assign_existing",
                    "type_id": duplicate["type_id"],
                    "type": duplicate["name"],
                    "facet": duplicate.get("facet_name") or "",
                    "facet_id": duplicate.get("facet_id"),
                })
            else:
                facet_name = str(item.get("facet") or "").strip()
                facet = facet_by_name.get(facet_name.casefold()) if facet_name else None
                candidate["facet_id"] = facet.get("facet_id") if facet else None
                if facet_name and not facet:
                    warnings.append(f"신규 타입 '{name}'의 Facet '{facet_name}'은 기존 Facet이 아닙니다. 새 Facet 또는 미지정으로 검토하세요.")
                comparisons = item.get("similar_types") or []
                if not isinstance(comparisons, list):
                    candidate["similar_types"] = []
                else:
                    candidate["similar_types"] = comparisons[:5]
        normalized.append(candidate)

    if not normalized:
        raise ValueError("사용 가능한 타입 후보를 찾지 못했습니다.")
    return {
        "summary": str(payload.get("summary") or "").strip(),
        "recommendations": normalized,
        "warnings": warnings,
    }


def parse_relation_suggestions(text: str, *, existing_types: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _extract_json(text)
    if not isinstance(payload, dict):
        raise ValueError("관계 후보 응답은 JSON object여야 합니다.")
    suggestions = payload.get("relations") or []
    if not isinstance(suggestions, list):
        raise ValueError("relations는 배열이어야 합니다.")
    by_id = {str(x["type_id"]): x for x in existing_types}
    by_name = {str(x["name"]).casefold(): x for x in existing_types}
    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    for idx, item in enumerate(suggestions[:8], start=1):
        if not isinstance(item, dict):
            continue
        def resolve(prefix: str) -> dict[str, Any] | None:
            item_id = str(item.get(f"{prefix}_type_id") or "").strip()
            name = str(item.get(f"{prefix}_type") or "").strip()
            return by_id.get(item_id) if item_id else by_name.get(name.casefold())
        source = resolve("source")
        target = resolve("target")
        rel_name = str(item.get("relation_name") or "").strip()
        if not source or not target or source["type_id"] == target["type_id"] or not rel_name:
            warnings.append(f"관계 후보 {idx}는 타입 또는 관계명이 유효하지 않아 제외했습니다.")
            continue
        normalized.append({
            **item,
            "source_type_id": source["type_id"],
            "source_type": source["name"],
            "target_type_id": target["type_id"],
            "target_type": target["name"],
            "relation_name": rel_name,
        })
    return {"relations": normalized, "warnings": warnings}
