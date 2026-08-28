from __future__ import annotations

import re
import uuid
from itertools import combinations
from typing import Callable

from research_fellow.domain.knowledge import KnowledgeRelation
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.storage import Ledger


RELATION_TYPES = ("supports", "extends", "contradicts", "qualifies", "uses_method", "addresses_gap")


def relation_text_prompt(source: dict[str, object], target: dict[str, object]) -> str:
    return render_prompt("m1_relation_proposal.j2", source=source, target=target)


def lineage_overview_prompt(cards: list[dict[str, object]], relations: list[dict[str, object]]) -> str:
    """Prompt only; the graph and approved relations remain the source of truth."""
    visible = cards[:10]
    card_index = {str(card["card_id"]): card for card in visible}
    edges = [
        {**relation, "source_title": card_index[str(relation["source_card_id"])]["title"], "target_title": card_index[str(relation["target_card_id"])]["title"]}
        for relation in relations if str(relation["source_card_id"]) in card_index and str(relation["target_card_id"]) in card_index
    ]
    return render_prompt("m1_lineage_overview.j2", cards=visible, relations=edges)


def _parse_relation_text(text: str) -> dict[str, str]:
    aliases = {"관계": "relation_type", "relation": "relation_type", "근거": "evidence", "evidence": "evidence",
               "조건": "conditions", "conditions": "conditions", "신뢰": "confidence", "confidence": "confidence"}
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if normalized := aliases.get(key.strip().lower()):
                result[normalized] = value.strip()
    return result


def normalize_relation_draft(
    source_card_id: str, target_card_id: str, relation_type: str, text_draft: str, evidence: str, conditions: str, confidence: str,
) -> tuple[dict[str, object], list[str]]:
    fields = _parse_relation_text(text_draft)
    warnings: list[str] = []
    selected_type = fields.get("relation_type", relation_type).strip().lower()
    if selected_type not in RELATION_TYPES:
        raise ValueError(f"지원하지 않는 관계 유형입니다: {selected_type}")
    normalized_evidence = fields.get("evidence", evidence).strip()
    if not normalized_evidence:
        raise ValueError("관계 근거를 입력하거나 초안에 근거 항목을 포함하세요.")
    if not fields.get("evidence") and text_draft.strip():
        warnings.append("초안에 관계 근거가 없어 입력란의 근거를 사용했습니다.")
    relation = KnowledgeRelation(
        relation_id=f"kr-candidate-{uuid.uuid4().hex[:12]}", source_card_id=source_card_id,
        target_card_id=target_card_id, relation_type=selected_type, evidence=normalized_evidence,
        conditions=fields.get("conditions", conditions).strip() or "두 카드의 원문 적용 조건을 함께 검토해야 합니다.",
        confidence=fields.get("confidence", confidence).strip().lower() or "medium",
    )
    return relation.model_dump(mode="json"), warnings


def create_relation_candidate(
    ledger: Ledger, source_card_id: str, target_card_id: str, relation_type: str, text_draft: str, evidence: str, conditions: str, confidence: str,
    source_card: dict[str, object] | None = None, target_card: dict[str, object] | None = None,
) -> tuple[str, list[str]]:
    relation, warnings = normalize_relation_draft(
        source_card_id, target_card_id, relation_type, text_draft, evidence, conditions, confidence
    )
    case_id = ledger.create_case("research", f"지식 관계: {source_card_id} → {target_card_id}")
    source_summary = _card_summary(source_card) if source_card else {"card_id": source_card_id}
    target_summary = _card_summary(target_card) if target_card else {"card_id": target_card_id}
    relation_summary = _relation_summary(relation, source_summary, target_summary)
    request_id = ledger.record(
        case_id, "decision_request", "m1", ["researcher"], "knowledge_relation",
        {"title": f"지식 관계 승인: {relation['relation_type']}", "relation": relation,
         "source_card_summary": source_summary, "target_card_summary": target_summary,
         "relation_summary": relation_summary, "warnings": warnings,
         "next_action": "승인 시 관계 기억에 저장하고 M2·연구자에게 통지"},
        subject_id=str(relation["relation_id"]),
    )
    return request_id, warnings


def _card_summary(card: dict[str, object]) -> dict[str, object]:
    """Readable approval snapshot; card ID remains the relation's canonical reference."""
    return {key: card.get(key, "") for key in ("card_id", "title", "claim", "explanation", "labels")}


def _relation_summary(relation: dict[str, object], source: dict[str, object], target: dict[str, object]) -> str:
    return (
        f"‘{source.get('title') or source.get('card_id')}’의 주장이 "
        f"‘{target.get('title') or target.get('card_id')}’의 주장과 "
        f"{relation['relation_type']} 관계인지 검토합니다."
    )


def _heuristic_relation(source: dict[str, object], target: dict[str, object]) -> tuple[str, str, str, str]:
    source_labels, target_labels = set(source.get("labels", [])), set(target.get("labels", []))
    relation_type = "supports" if source_labels & target_labels else "qualifies"
    evidence = f"출발 카드 주장: {source['claim']} / 도착 카드 주장: {target['claim']}"
    conditions = "두 카드의 출처·적용 대상·근거 범위를 연구자가 함께 검토할 때"
    return relation_type, evidence, conditions, "low"


def propose_relation_candidates(
    ledger: Ledger, cards: list[dict[str, object]], existing_relations: list[dict[str, object]],
    draft_for: Callable[[dict[str, object], dict[str, object]], str | None] | None = None,
    limit: int = 3,
) -> tuple[list[str], list[str]]:
    """Create a small review queue so the researcher confirms rather than authors relations."""
    existing_pairs = {(item["source_card_id"], item["target_card_id"]) for item in existing_relations}
    request_ids, warnings = [], []
    for source, target in combinations(cards[:6], 2):
        if len(request_ids) >= limit:
            break
        pair = (source["card_id"], target["card_id"])
        if pair in existing_pairs:
            continue
        relation_type, evidence, conditions, confidence = _heuristic_relation(source, target)
        draft = draft_for(source, target) if draft_for else None
        try:
            request_id, item_warnings = create_relation_candidate(
                ledger, str(source["card_id"]), str(target["card_id"]), relation_type, draft or "", evidence, conditions, confidence, source, target,
            )
        except ValueError:
            # A malformed LLM draft is ignored; deterministic heuristic proposal remains reviewable.
            request_id, item_warnings = create_relation_candidate(
                ledger, str(source["card_id"]), str(target["card_id"]), relation_type, "", evidence, conditions, confidence, source, target,
            )
            item_warnings.append("Gemma 초안을 해석할 수 없어 근거 카드 기반 보수적 제안으로 대체했습니다.")
        request_ids.append(request_id)
        warnings.extend(item_warnings)
    return request_ids, warnings


def lineage_dot(cards: list[dict[str, object]], relations: list[dict[str, object]]) -> str:
    """A compact projection, not a graph database or source of truth."""
    visible = cards[:10]
    ids = {str(card["card_id"]) for card in visible}
    def quote(value: object) -> str:
        return '"' + str(value).replace('"', "'").replace("\n", " ") + '"'
    lines = ["digraph lineage {", "rankdir=LR; node [shape=box, style=rounded];"]
    for card in visible:
        label = str(card["title"])[:48]
        lines.append(f"{quote(card['card_id'])} [label={quote(label)}];")
    for relation in relations:
        if str(relation["source_card_id"]) in ids and str(relation["target_card_id"]) in ids:
            label = f"{relation['relation_type']} ({relation['confidence']})"
            lines.append(f"{quote(relation['source_card_id'])} -> {quote(relation['target_card_id'])} [label={quote(label)}];")
    return "\n".join([*lines, "}"])
