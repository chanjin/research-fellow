"""Stable origin lineage shared by M1 discovery, papers, and knowledge cards."""

from __future__ import annotations

from typing import Any, Iterable


ORIGIN_TYPE_LABELS = {
    "researcher_question": "연구자 질문",
    "m2_knowledge": "M2 지식카드 처리",
    "paper_writing": "논문 작성",
}


def normalize_origin_links(values: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, str], int] = {}
    for value in values or []:
        if not isinstance(value, dict):
            continue
        origin_type = str(value.get("origin_type") or "").strip()
        origin_id = str(value.get("origin_id") or "").strip()
        origin_sub_id = str(value.get("origin_sub_id") or "").strip()
        if origin_type not in ORIGIN_TYPE_LABELS or not origin_id:
            continue
        key = (origin_type, origin_id, origin_sub_id)
        default_label = ORIGIN_TYPE_LABELS[origin_type]
        detail = str(value.get("label") or "").strip()
        label = detail if detail.startswith(default_label) else f"{default_label} · {detail}" if detail else default_label
        source_card_ids = list(dict.fromkeys(
            str(item).strip() for item in value.get("source_card_ids", []) if str(item).strip()
        ))[:24]
        candidate = {
            "origin_type": origin_type,
            "origin_id": origin_id,
            "origin_sub_id": origin_sub_id,
            "label": label[:160],
            "source_card_ids": source_card_ids,
            "research_title": str(value.get("research_title") or "").strip()[:500],
            "research_question": str(value.get("research_question") or "").strip()[:2000],
            "research_context": str(value.get("research_context") or "").strip()[:8000],
        }
        if key in positions:
            prior = normalized[positions[key]]
            prior["source_card_ids"] = list(dict.fromkeys([
                *prior.get("source_card_ids", []), *candidate["source_card_ids"],
            ]))[:24]
            if prior.get("label") == default_label and candidate.get("label") != default_label:
                prior["label"] = candidate["label"]
            for field in ("research_title", "research_question", "research_context"):
                if candidate.get(field) and not prior.get(field):
                    prior[field] = candidate[field]
            continue
        positions[key] = len(normalized)
        normalized.append(candidate)
    return normalized


def merge_origin_links(*groups: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return normalize_origin_links([item for group in groups for item in (group or [])])


def origin_labels(values: Iterable[dict[str, Any]] | None) -> list[str]:
    return [str(item["label"]) for item in normalize_origin_links(values)]


def origin_research_context(values: Iterable[dict[str, Any]] | None) -> str:
    """Render deduplicated, human-editable research context from task lineage."""
    blocks: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for item in normalize_origin_links(values):
        title = str(item.get("research_title") or "").strip()
        question = str(item.get("research_question") or "").strip()
        context = str(item.get("research_context") or "").strip()
        key = (title, question, context)
        if key in seen or not any(key):
            continue
        seen.add(key)
        lines = [f"연구 제목: {title}" if title else "", f"연구 질문: {question}" if question else "", f"의도 맥락: {context}" if context else ""]
        blocks.append("\n".join(line for line in lines if line))
    return "\n\n".join(blocks)


def cards_for_origin(cards: Iterable[dict[str, Any]], origin_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {str(value).strip() for value in origin_ids if str(value).strip()}
    if not wanted:
        return []
    return [
        card for card in cards
        if any(str(link.get("origin_id") or "") in wanted for link in normalize_origin_links(card.get("origin_links", [])))
    ]
