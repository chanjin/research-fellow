"""LLM-assisted, researcher-approved ontology change reviews."""

from __future__ import annotations

import json
import re
from typing import Any

from research_fellow.infrastructure.prompt_renderer import render_prompt


def ontology_delta_prompt(*, cards: list[dict[str, Any]], facets: list[dict[str, Any]], types: list[dict[str, Any]], relations: list[dict[str, Any]], comment: str = "") -> str:
    return render_prompt("m1_ontology_delta.j2", cards=cards, facets=facets, types=types, relations=relations, comment=comment)


def _json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("온톨로지 변경안 JSON을 찾지 못했습니다.")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("온톨로지 변경안은 JSON object여야 합니다.")
    return value


def parse_ontology_delta(text: str, *, cards: list[dict[str, Any]], types: list[dict[str, Any]]) -> dict[str, Any]:
    raw = _json(text)
    card_ids = {str(card.get("card_id")) for card in cards}
    type_by_id = {str(item.get("type_id")): item for item in types}
    type_by_name = {str(item.get("name", "")).casefold(): item for item in types}
    new_types: list[dict[str, Any]] = []
    for item in raw.get("new_types") or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        if str(item["name"]).casefold() not in type_by_name:
            new_types.append({
                "name": str(item["name"]).strip(), "description": str(item.get("description") or "").strip(),
                "facet": str(item.get("facet") or "").strip(), "facet_description": str(item.get("facet_description") or "").strip(),
                "facet_color": str(item.get("facet_color") or "#DCEBFF").strip(), "reason": str(item.get("reason") or "").strip(),
            })
    assignments: list[dict[str, Any]] = []
    for item in raw.get("assignments") or []:
        if not isinstance(item, dict) or str(item.get("card_id") or "") not in card_ids:
            continue
        found = type_by_id.get(str(item.get("type_id") or "")) or type_by_name.get(str(item.get("type") or "").casefold())
        type_name = str((found or {}).get("name") or item.get("type") or "").strip()
        if type_name:
            assignments.append({"card_id": str(item["card_id"]), "type_id": str((found or {}).get("type_id") or ""), "type": type_name, "reason": str(item.get("reason") or "").strip(), "confidence": str(item.get("confidence") or "medium")})
    reviewed_card_ids = [str(value) for value in raw.get("reviewed_card_ids") or [] if str(value) in card_ids]
    if not reviewed_card_ids:
        reviewed_card_ids = sorted({str(item["card_id"]) for item in assignments})
    relations: list[dict[str, Any]] = []
    for item in raw.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = type_by_id.get(str(item.get("source_type_id") or "")) or type_by_name.get(str(item.get("source_type") or "").casefold())
        target = type_by_id.get(str(item.get("target_type_id") or "")) or type_by_name.get(str(item.get("target_type") or "").casefold())
        name = str(item.get("relation_name") or "").strip()
        if source and target and source["type_id"] != target["type_id"] and name:
            relations.append({"source_type_id": source["type_id"], "source_type": source["name"], "target_type_id": target["type_id"], "target_type": target["name"], "relation_name": name, "description": str(item.get("description") or "").strip(), "reason": str(item.get("reason") or "").strip()})
    type_updates: list[dict[str, Any]] = []
    for item in raw.get("type_updates") or []:
        if not isinstance(item, dict):
            continue
        current = type_by_id.get(str(item.get("type_id") or ""))
        name = str(item.get("name") or "").strip()
        if current and name:
            type_updates.append({"type_id": str(current["type_id"]), "name": name, "description": str(item.get("description") or current.get("description") or "").strip(), "reason": str(item.get("reason") or "").strip()})
    relation_changes: list[dict[str, Any]] = []
    for item in raw.get("relation_changes") or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        relation_id = str(item.get("relation_id") or "").strip()
        source = type_by_id.get(str(item.get("source_type_id") or "")) or type_by_name.get(str(item.get("source_type") or "").casefold())
        target = type_by_id.get(str(item.get("target_type_id") or "")) or type_by_name.get(str(item.get("target_type") or "").casefold())
        name = str(item.get("relation_name") or "").strip()
        if action in {"update", "delete"} and not relation_id:
            continue
        if action == "add" and (not source or not target or source["type_id"] == target["type_id"] or not name):
            continue
        relation_changes.append({
            "action": action, "relation_id": relation_id,
            "source_type_id": str((source or {}).get("type_id") or ""),
            "target_type_id": str((target or {}).get("type_id") or ""),
            "relation_name": name, "description": str(item.get("description") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        })
    return {"summary": str(raw.get("summary") or "").strip(), "reviewed_card_ids": reviewed_card_ids, "assignments": assignments, "new_types": new_types, "relations": relations, "type_updates": type_updates, "relation_changes": relation_changes, "warnings": [str(x) for x in raw.get("warnings") or []][:8]}
