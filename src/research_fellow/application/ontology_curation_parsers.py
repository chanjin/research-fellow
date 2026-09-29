"""Structured-output parsers for ontology curation."""
from __future__ import annotations
from typing import Any
from research_fellow.application.structured_output import extract_json_value
def parse_type_suggestions(
    text: str,
    *,
    existing_facets: list[dict[str, Any]],
    existing_types: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = extract_json_value(text)
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
    payload = extract_json_value(text)
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
