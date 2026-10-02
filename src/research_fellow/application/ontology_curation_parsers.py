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


def parse_facet_suggestions(text: str, *, existing_types: list[dict[str, Any]]) -> dict[str, Any]:
    payload = extract_json_value(text)
    if not isinstance(payload, dict):
        raise ValueError("Facet 후보 응답은 JSON object여야 합니다.")
    facets = payload.get("facets") or []
    assignments = payload.get("assignments") or []
    if not isinstance(facets, list) or not facets:
        raise ValueError("facets가 하나 이상 필요합니다.")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("assignments가 하나 이상 필요합니다.")
    clean_facets: dict[str, dict[str, Any]] = {}
    for item in facets[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        clean_facets[name.casefold()] = {
            "name": name,
            "description": str(item.get("description") or "").strip(),
            "color": str(item.get("color") or "#DCEBFF").strip(),
            "reason": str(item.get("reason") or "").strip(),
        }
    if not clean_facets:
        raise ValueError("사용 가능한 Facet 후보가 없습니다.")
    type_by_id = {str(item.get("type_id") or ""): item for item in existing_types}
    type_updates: list[dict[str, Any]] = []
    seen: set[str] = set()
    warnings = [str(x) for x in payload.get("warnings") or []][:8]
    for item in assignments:
        if not isinstance(item, dict):
            continue
        type_id = str(item.get("type_id") or "").strip()
        current = type_by_id.get(type_id)
        facet_name = str(item.get("facet") or "").strip()
        facet = clean_facets.get(facet_name.casefold())
        if not current or not facet or type_id in seen:
            continue
        seen.add(type_id)
        type_updates.append({
            "type_id": type_id,
            "name": str(current.get("name") or ""),
            "description": str(current.get("description") or ""),
            "facet": facet["name"],
            "facet_description": facet["description"],
            "facet_color": facet["color"],
            "reason": str(item.get("reason") or facet.get("reason") or "").strip(),
        })
    missing = [type_id for type_id in type_by_id if type_id not in seen]
    if missing:
        raise ValueError(f"모든 Type에 Facet이 필요합니다. 누락 Type {len(missing)}개")
    return {
        "summary": str(payload.get("summary") or "Ontology Facet 구조 제안").strip(),
        "reviewed_card_ids": [],
        "assignments": [],
        "new_types": [],
        "relations": [],
        "type_updates": type_updates,
        "relation_changes": [],
        "warnings": warnings,
    }



def parse_type_graph_suggestions(
    text: str,
    *,
    cards: list[dict[str, Any]],
    reference_cards: list[dict[str, Any]],
    existing_types: list[dict[str, Any]],
    existing_knowledge_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a batch Type-graph proposal covering every input card."""
    payload = extract_json_value(text)
    if not isinstance(payload, dict):
        raise ValueError("타입 그래프 응답은 JSON object여야 합니다.")

    card_ids = {str(item.get("card_id") or "") for item in cards if str(item.get("card_id") or "")}
    all_card_ids = set(card_ids)
    all_card_ids.update(str(item.get("card_id") or "") for item in reference_cards if str(item.get("card_id") or ""))
    existing_card_pairs = {
        (str(item.get("source_card_id") or ""), str(item.get("target_card_id") or ""), str(item.get("relation_type") or "").casefold())
        for item in existing_knowledge_relations
    }
    type_by_id = {str(item.get("type_id") or ""): item for item in existing_types}
    type_by_name = {str(item.get("name") or "").casefold(): item for item in existing_types}

    raw_new_types = payload.get("new_types") or []
    if not isinstance(raw_new_types, list):
        raise ValueError("new_types는 배열이어야 합니다.")
    new_types: list[dict[str, Any]] = []
    new_names: dict[str, dict[str, Any]] = {}
    warnings = [str(item) for item in (payload.get("warnings") or [])][:12]
    for item in raw_new_types[:40]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        existing = type_by_name.get(name.casefold())
        if existing:
            warnings.append(f"'{name}'은(는) 기존 Type이라 신규 Type에서 제외했습니다.")
            continue
        if name.casefold() in new_names:
            continue
        clean = {
            "name": name,
            "description": str(item.get("description") or "").strip(),
            "facet": "",
            "facet_description": "",
            "facet_color": "#DCEBFF",
            "reason": str(item.get("reason") or "").strip(),
        }
        new_names[name.casefold()] = clean
        new_types.append(clean)

    raw_assignments = payload.get("assignments") or []
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise ValueError("assignments가 하나 이상 필요합니다.")
    assignments: list[dict[str, Any]] = []
    assigned_cards: set[str] = set()
    seen_assignment: set[tuple[str, str]] = set()
    for item in raw_assignments[: max(80, len(card_ids) * 4)]:
        if not isinstance(item, dict):
            continue
        card_id = str(item.get("card_id") or "").strip()
        if card_id not in card_ids:
            continue
        type_id = str(item.get("type_id") or "").strip()
        type_name = str(item.get("type") or item.get("type_name") or "").strip()
        if type_id:
            existing = type_by_id.get(type_id)
            if not existing:
                warnings.append(f"카드 {card_id}: 존재하지 않는 type_id {type_id} 배정을 제외했습니다.")
                continue
            type_name = str(existing.get("name") or "")
        else:
            existing = type_by_name.get(type_name.casefold()) if type_name else None
            if existing:
                type_id = str(existing.get("type_id") or "")
                type_name = str(existing.get("name") or "")
            elif not type_name or type_name.casefold() not in new_names:
                warnings.append(f"카드 {card_id}: 정의되지 않은 신규 Type '{type_name}' 배정을 제외했습니다.")
                continue
        key = (card_id, type_id or type_name.casefold())
        if key in seen_assignment:
            continue
        seen_assignment.add(key)
        assigned_cards.add(card_id)
        assignments.append({
            "card_id": card_id,
            "type_id": type_id,
            "type": type_name,
            "reason": str(item.get("reason") or "").strip(),
            "confidence": str(item.get("confidence") or "medium").strip() or "medium",
        })

    missing = sorted(card_ids - assigned_cards)
    if missing:
        raise ValueError(f"모든 지식카드에 Type 배정이 필요합니다. 누락 카드 {len(missing)}개: {', '.join(missing[:6])}")

    raw_relations = payload.get("relations") or []
    if not isinstance(raw_relations, list):
        raise ValueError("relations는 배열이어야 합니다.")
    relations: list[dict[str, Any]] = []
    seen_relations: set[tuple[str, str, str]] = set()

    def resolve_type(item: dict[str, Any], prefix: str) -> tuple[str, str] | None:
        type_id = str(item.get(f"{prefix}_type_id") or "").strip()
        type_name = str(item.get(f"{prefix}_type") or "").strip()
        if type_id:
            existing = type_by_id.get(type_id)
            if not existing:
                return None
            return type_id, str(existing.get("name") or "")
        existing = type_by_name.get(type_name.casefold()) if type_name else None
        if existing:
            return str(existing.get("type_id") or ""), str(existing.get("name") or "")
        if type_name and type_name.casefold() in new_names:
            return "", new_names[type_name.casefold()]["name"]
        return None

    for item in raw_relations[:80]:
        if not isinstance(item, dict):
            continue
        source = resolve_type(item, "source")
        target = resolve_type(item, "target")
        relation_name = str(item.get("relation_name") or "").strip()
        if not source or not target or not relation_name:
            continue
        if source == target:
            continue
        key = (source[0] or source[1].casefold(), target[0] or target[1].casefold(), relation_name.casefold())
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append({
            "source_type_id": source[0],
            "source_type": source[1],
            "target_type_id": target[0],
            "target_type": target[1],
            "relation_name": relation_name,
            "description": str(item.get("description") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        })

    raw_card_relations = payload.get("card_relations") or []
    if not isinstance(raw_card_relations, list):
        raise ValueError("card_relations는 배열이어야 합니다.")
    allowed_card_relations = {"supports", "extends", "contradicts", "qualifies", "uses_method", "addresses_gap"}
    knowledge_relations: list[dict[str, Any]] = []
    seen_card_relations: set[tuple[str, str, str]] = set()
    for item in raw_card_relations[:120]:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_card_id") or "").strip()
        target_id = str(item.get("target_card_id") or "").strip()
        relation_type = str(item.get("relation_type") or "").strip().casefold()
        if not source_id or not target_id or source_id == target_id:
            continue
        if source_id not in all_card_ids or target_id not in all_card_ids:
            continue
        # At least one endpoint must be in the current batch; do not rewrite old-old relations.
        if source_id not in card_ids and target_id not in card_ids:
            continue
        if relation_type not in allowed_card_relations:
            warnings.append(f"지원하지 않는 카드 관계 '{relation_type}'를 제외했습니다.")
            continue
        if (source_id, target_id, relation_type) in existing_card_pairs:
            continue
        key = (source_id, target_id, relation_type)
        if key in seen_card_relations:
            continue
        evidence = str(item.get("evidence") or "").strip()
        if not evidence:
            continue
        confidence = str(item.get("confidence") or "medium").strip().casefold()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        seen_card_relations.add(key)
        knowledge_relations.append({
            "source_card_id": source_id,
            "target_card_id": target_id,
            "relation_type": relation_type,
            "evidence": evidence,
            "conditions": str(item.get("conditions") or "").strip() or "두 지식카드의 적용 범위와 근거를 함께 검토해야 합니다.",
            "confidence": confidence,
        })

    return {
        "summary": str(payload.get("summary") or "지식카드 묶음 기반 Type graph 제안").strip(),
        "reviewed_card_ids": sorted(card_ids),
        "assignments": assignments,
        "new_types": new_types,
        "relations": relations,
        "knowledge_relations": knowledge_relations,
        "type_updates": [],
        "relation_changes": [],
        "warnings": warnings,
    }
