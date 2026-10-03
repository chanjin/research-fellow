"""Durable M1 ontology-curation task orchestration.

This module connects approved Knowledge Cards to the existing external-manual
LLM + Attention path.  It deliberately reuses ontology change reviews as the
researcher approval boundary instead of introducing another ontology state.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from research_fellow.application.ontology_curation_context import build_curation_context
from research_fellow.application.ontology_curation_parsers import (
    parse_facet_suggestions,
    parse_relation_suggestions,
    parse_type_suggestions,
    parse_type_graph_suggestions,
)
from research_fellow.application.ontology_curation_prompts import (
    facet_suggestion_prompt,
    relation_suggestion_prompt,
    type_suggestion_prompt,
    type_graph_suggestion_prompt,
)
from research_fellow.application.relations import (
    create_relation_candidate,
    parse_relation_batch_drafts,
    relation_batch_prompt,
)
from research_fellow.infrastructure.retrieval import KnowledgeRetriever
from research_fellow.memory import KnowledgeMemory
from research_fellow.storage import Ledger

ONTOLOGY_TYPE_TASK = "ontology_type_suggestion"
KNOWLEDGE_RELATION_TASK = "knowledge_relation_suggestion"
ONTOLOGY_RELATION_TASK = "ontology_relation_suggestion"
ONTOLOGY_FACET_TASK = "ontology_facet_suggestion"
FACET_SUGGESTION_MIN_TYPES = 20


def _task_exists(ledger: Ledger, subject_type: str, subject_id: str) -> bool:
    return any(
        str(item.get("subject_type") or "") == subject_type
        and str(item.get("subject_id") or "") == subject_id
        and str(item.get("status") or "") in {"ready", "completed"}
        for item in ledger.phenomena(type_="research_task")
    )


def _record_task(
    ledger: Ledger,
    *,
    subject_type: str,
    subject_id: str,
    title: str,
    prompt: str,
    expected_output: str,
    payload: dict[str, Any] | None = None,
) -> str:
    case_id = ledger.create_case("research", title[:120])
    return ledger.record(
        case_id,
        "research_task",
        "m1",
        ["researcher"],
        subject_type,
        {
            "title": title,
            "task_type": subject_type,
            "prompt": prompt,
            "expected_output": expected_output,
            "context": dict(payload or {}),
        },
        subject_id=subject_id,
        status="ready",
    )


def _ready_type_tasks(ledger: Ledger) -> list[dict[str, Any]]:
    return [
        item for item in ledger.phenomena(type_="research_task")
        if str(item.get("subject_type") or "") == ONTOLOGY_TYPE_TASK
        and str(item.get("status") or "") == "ready"
    ]


def _untyped_cards(ledger: Ledger, memory: KnowledgeMemory, *, limit: int = 30) -> list[dict[str, Any]]:
    cards = [
        dict(card) for card in memory.all()
        if str(card.get("card_id") or "")
        and not ledger.ontology_types_for_card(str(card.get("card_id") or ""))
    ]
    cards.sort(key=lambda card: str(card.get("created_at") or card.get("card_id") or ""))
    return cards[: max(1, int(limit))]


def _type_graph_fingerprint(cards: list[dict[str, Any]], ledger: Ledger) -> str:
    value = {
        "card_ids": sorted(str(card.get("card_id") or "") for card in cards),
        "types": sorted((str(item.get("type_id") or ""), str(item.get("name") or "")) for item in ledger.ontology_types()),
        "type_relations": sorted((str(item.get("source_type_id") or ""), str(item.get("target_type_id") or ""), str(item.get("relation_name") or "")) for item in ledger.ontology_type_relations()),
        "knowledge_relations": sorted((str(item.get("source_card_id") or ""), str(item.get("target_card_id") or ""), str(item.get("relation_type") or "")) for item in ledger.active_knowledge_relations()),
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _complete_ready_legacy_relation_tasks(ledger: Ledger) -> None:
    """Retire old per-card relation Inputs now superseded by the graph-level task."""
    for item in ledger.phenomena(type_="research_task"):
        if (
            str(item.get("subject_type") or "") == KNOWLEDGE_RELATION_TASK
            and str(item.get("status") or "") == "ready"
        ):
            ledger.set_status(str(item.get("phenomenon_id") or ""), "completed")


def _supersede_pending_reviews_for_scope(ledger: Ledger, card_ids: list[str]) -> list[str]:
    """Close stale pending Ontology reviews for the same batch of source cards."""
    scope = sorted({str(value) for value in card_ids if str(value)})
    if not scope:
        return []
    closed: list[str] = []
    for review in ledger.ontology_change_reviews(status="proposed", limit=200):
        review_scope = sorted({str(value) for value in (review.get("source_card_ids") or []) if str(value)})
        if review_scope != scope:
            continue
        review_id = str(review.get("review_id") or "")
        if not review_id:
            continue
        ledger.resolve_ontology_change_review(
            review_id,
            "rejected",
            note="Superseded by a newer Ontology Graph run for the same knowledge-card scope.",
        )
        closed.append(review_id)
    return closed


def ensure_type_graph_work(ledger: Ledger, memory: KnowledgeMemory, *, limit: int = 30) -> str | None:
    """Keep exactly one graph-level ontology Input for the current untyped-card batch.

    The same Input covers card-to-Type assignment, new Types, Type relations, and
    Knowledge Card relations. Legacy per-card relation Inputs are retired.
    """
    _complete_ready_legacy_relation_tasks(ledger)
    cards = _untyped_cards(ledger, memory, limit=limit)
    ready = _ready_type_tasks(ledger)
    if not cards:
        for task in ready:
            ledger.set_status(str(task.get("phenomenon_id") or ""), "completed")
        return None

    fingerprint = _type_graph_fingerprint(cards, ledger)
    for task in ready:
        context = dict((task.get("payload") or {}).get("context") or {})
        if str(context.get("batch_fingerprint") or "") == fingerprint:
            return str(task.get("phenomenon_id") or "")

    for task in ready:
        ledger.set_status(str(task.get("phenomenon_id") or ""), "completed")

    batch_ids = {str(card.get("card_id") or "") for card in cards}
    all_cards = [dict(card) for card in memory.all() if str(card.get("card_id") or "")]
    reference_cards = [card for card in all_cards if str(card.get("card_id") or "") not in batch_ids][:30]
    card_ids = [str(card.get("card_id") or "") for card in cards]
    reference_card_ids = [str(card.get("card_id") or "") for card in reference_cards]
    # A deliberate graph re-run replaces an older still-pending proposal for the
    # exact same card batch. Otherwise the researcher sees duplicate Reviews.
    _supersede_pending_reviews_for_scope(ledger, card_ids)
    prompt = type_graph_suggestion_prompt(
        cards=cards,
        reference_cards=reference_cards,
        types=ledger.ontology_types(),
        relations=ledger.ontology_type_relations(),
        knowledge_relations=ledger.active_knowledge_relations(),
    )
    return _record_task(
        ledger,
        subject_type=ONTOLOGY_TYPE_TASK,
        subject_id=f"type-graph:{fingerprint}",
        title=f"Ontology Graph 제안 · 미분류 지식 {len(cards)}건",
        prompt=prompt,
        expected_output="JSON graph proposal with assignments[], new_types[], relations[], and card_relations[]",
        payload={
            "card_ids": card_ids,
            "reference_card_ids": reference_card_ids,
            "batch_fingerprint": fingerprint,
            "card_count": len(card_ids),
        },
    )

def enqueue_card_ontology_work(ledger: Ledger, memory: KnowledgeMemory, card_id: str) -> dict[str, Any]:
    """Refresh the single graph-level ontology task after one card is approved."""
    card = next((item for item in memory.all() if str(item.get("card_id") or "") == str(card_id)), None)
    if card is None:
        raise ValueError(f"Knowledge Card를 찾을 수 없습니다: {card_id}")
    before_ready = {str(item.get("phenomenon_id") or "") for item in _ready_type_tasks(ledger)}
    type_task_id = ensure_type_graph_work(ledger, memory)
    created = [type_task_id] if type_task_id and type_task_id not in before_ready else []
    return {"card_id": card_id, "created_task_ids": created}

def enqueue_untyped_cards(ledger: Ledger, memory: KnowledgeMemory, *, limit: int = 30) -> dict[str, Any]:
    """Backfill one graph-level ontology task; never create per-card ontology Inputs."""
    before_ready = {str(item.get("phenomenon_id") or "") for item in _ready_type_tasks(ledger)}
    type_task_id = ensure_type_graph_work(ledger, memory, limit=limit)
    created = [type_task_id] if type_task_id and type_task_id not in before_ready else []
    facet_task = maybe_enqueue_facet_work(ledger)
    if facet_task:
        created.append(facet_task)
    return {"considered_cards": min(len(list(memory.all())), max(1, int(limit))), "created_task_ids": created}

def _type_review_proposal(card_id: str, parsed: dict[str, Any]) -> dict[str, Any]:
    assignments: list[dict[str, Any]] = []
    new_types: list[dict[str, Any]] = []
    warnings = list(parsed.get("warnings") or [])
    for item in parsed.get("recommendations") or []:
        action = str(item.get("action") or "")
        if action == "assign_existing":
            assignments.append({
                "card_id": card_id,
                "type_id": str(item.get("type_id") or ""),
                "type": str(item.get("type") or ""),
                "reason": str(item.get("reason") or ""),
                "confidence": str(item.get("confidence") or "medium"),
            })
        elif action == "create_new":
            name = str(item.get("type") or "").strip()
            if not name:
                continue
            new_types.append({
                "name": name,
                "description": str(item.get("description") or ""),
                # Facets are intentionally not invented here.  They are curated
                # later as a graph-level abstraction once enough Types exist.
                "facet": str(item.get("facet") or ""),
                "facet_description": "",
                "facet_color": "#DCEBFF",
                "reason": str(item.get("reason") or ""),
            })
            assignments.append({
                "card_id": card_id,
                "type_id": "",
                "type": name,
                "reason": str(item.get("reason") or ""),
                "confidence": str(item.get("confidence") or "medium"),
            })
        else:
            warnings.append(f"연구자 추가 판단이 필요한 타입 후보: {item.get('type') or '이름 없음'}")
    if not assignments:
        raise ValueError("승인 검토로 보낼 수 있는 타입 배정 후보가 없습니다.")
    return {
        "summary": str(parsed.get("summary") or "지식카드 Ontology Type 제안"),
        "reviewed_card_ids": [card_id],
        "assignments": assignments,
        "new_types": new_types,
        "relations": [],
        "type_updates": [],
        "relation_changes": [],
        "warnings": warnings,
    }


def apply_ontology_task_response(
    ledger: Ledger,
    memory: KnowledgeMemory,
    task: dict[str, Any],
    response: str,
) -> dict[str, Any]:
    """Validate one ontology-related external LLM response and persist its next boundary."""
    subject_type = str(task.get("subject_type") or "")
    payload = dict(task.get("payload") or {})
    context = dict(payload.get("context") or {})
    task_id = str(task.get("phenomenon_id") or "")
    card_id = str(context.get("card_id") or task.get("subject_id") or "")
    cards_by_id = {str(item.get("card_id") or ""): item for item in memory.all()}

    if subject_type == ONTOLOGY_TYPE_TASK:
        card_ids = [str(value) for value in (context.get("card_ids") or []) if str(value)]
        batch_cards = [cards_by_id[value] for value in card_ids if value in cards_by_id]
        if not batch_cards and card_id and card_id in cards_by_id:
            # Legacy single-card task compatibility.
            batch_cards = [cards_by_id[card_id]]
            card_ids = [card_id]
        reference_ids = [str(value) for value in (context.get("reference_card_ids") or []) if str(value)]
        reference_cards = [cards_by_id[value] for value in reference_ids if value in cards_by_id]
        parsed = parse_type_graph_suggestions(
            response,
            cards=batch_cards,
            reference_cards=reference_cards,
            existing_types=ledger.ontology_types(),
            existing_knowledge_relations=ledger.active_knowledge_relations(),
        )
        relation_drafts = list(parsed.pop("knowledge_relations", []) or [])
        # Card-to-card relation proposals are part of the same graph-level change.
        # Do not fan them back out into one Decision per relation.
        parsed["card_relations"] = relation_drafts

        # Clean up exact legacy relation decisions created by older ontology runs so
        # the researcher sees one graph-level decision rather than duplicated cards.
        proposed_keys = {
            (str(item.get("source_card_id") or ""), str(item.get("target_card_id") or ""), str(item.get("relation_type") or "").casefold())
            for item in relation_drafts
        }
        if proposed_keys:
            for request in ledger.phenomena(type_="decision_request", status="proposed"):
                if str(request.get("subject_type") or "") != "knowledge_relation":
                    continue
                rel = dict((request.get("payload") or {}).get("relation") or {})
                key = (str(rel.get("source_card_id") or ""), str(rel.get("target_card_id") or ""), str(rel.get("relation_type") or "").casefold())
                if key in proposed_keys:
                    ledger.set_status(str(request.get("phenomenon_id") or ""), "completed")

        superseded_review_ids = _supersede_pending_reviews_for_scope(ledger, card_ids)
        review = ledger.create_ontology_change_review(
            source_card_ids=card_ids,
            proposal=parsed,
            base_version_id=(ledger.latest_ontology_version() or {}).get("version_id"),
        )
        ledger.set_status(task_id, "completed")
        return {
            "task_id": task_id,
            "review_id": str(review["review_id"]),
            "kind": subject_type,
            "superseded_review_ids": superseded_review_ids,
        }

    if subject_type == KNOWLEDGE_RELATION_TASK:
        source = context.get("source_card") if isinstance(context.get("source_card"), dict) else cards_by_id.get(card_id, {})
        targets = [dict(item) for item in (context.get("targets") or []) if isinstance(item, dict)]
        drafts = parse_relation_batch_drafts(response, card_id, targets)
        request_ids: list[str] = []
        for draft in drafts:
            target = next((item for item in targets if str(item.get("card_id") or "") == str(draft.get("target_card_id") or "")), None)
            request_id, _ = create_relation_candidate(
                ledger,
                card_id,
                str(draft["target_card_id"]),
                str(draft["relation_type"]),
                "",
                str(draft["evidence"]),
                str(draft["conditions"]),
                str(draft["confidence"]),
                source,
                target,
            )
            request_ids.append(request_id)
        ledger.set_status(task_id, "completed")
        return {"task_id": task_id, "decision_request_ids": request_ids, "kind": subject_type}

    if subject_type == ONTOLOGY_RELATION_TASK:
        parsed = parse_relation_suggestions(response, existing_types=ledger.ontology_types())
        proposal = {
            "summary": f"{context.get('card_title') or card_id}의 승인 Type을 기존 Type graph에 연결하는 관계 제안",
            "reviewed_card_ids": [],
            "assignments": [],
            "new_types": [],
            "relations": list(parsed.get("relations") or []),
            "type_updates": [],
            "relation_changes": [],
            "warnings": list(parsed.get("warnings") or []),
        }
        if not proposal["relations"]:
            ledger.set_status(task_id, "completed")
            return {"task_id": task_id, "review_id": "", "kind": subject_type, "message": "새 Type 관계 제안 없음"}
        review = ledger.create_ontology_change_review(
            source_card_ids=[card_id] if card_id else [],
            proposal=proposal,
            base_version_id=(ledger.latest_ontology_version() or {}).get("version_id"),
        )
        ledger.set_status(task_id, "completed")
        return {"task_id": task_id, "review_id": str(review["review_id"]), "kind": subject_type}

    if subject_type == ONTOLOGY_FACET_TASK:
        parsed = parse_facet_suggestions(response, existing_types=ledger.ontology_types())
        review = ledger.create_ontology_change_review(
            source_card_ids=[],
            proposal=parsed,
            base_version_id=(ledger.latest_ontology_version() or {}).get("version_id"),
        )
        ledger.set_status(task_id, "completed")
        return {"task_id": task_id, "review_id": str(review["review_id"]), "kind": subject_type}

    raise ValueError(f"지원하지 않는 Ontology research task입니다: {subject_type}")


def enqueue_type_relation_work(ledger: Ledger, memory: KnowledgeMemory, card_id: str) -> str | None:
    approved_types = ledger.ontology_types_for_card(card_id)
    all_types = ledger.ontology_types()
    if not approved_types or len(all_types) < 2 or _task_exists(ledger, ONTOLOGY_RELATION_TASK, card_id):
        return None
    card = next((item for item in memory.all() if str(item.get("card_id") or "") == card_id), None)
    if card is None:
        return None
    return _record_task(
        ledger,
        subject_type=ONTOLOGY_RELATION_TASK,
        subject_id=card_id,
        title=f"Ontology Type 관계 제안 · {str(card.get('title') or card_id)[:72]}",
        prompt=relation_suggestion_prompt(approved_types, all_types, ledger.ontology_type_relations(), card),
        expected_output="JSON object with relations[] linking approved Types to the existing Type graph",
        payload={"card_id": card_id, "card_title": str(card.get("title") or "")},
    )


def _facet_fingerprint(types: list[dict[str, Any]]) -> str:
    value = [
        {"type_id": str(item.get("type_id") or ""), "name": str(item.get("name") or ""), "facet_id": str(item.get("facet_id") or "")}
        for item in sorted(types, key=lambda row: str(row.get("type_id") or ""))
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def maybe_enqueue_facet_work(ledger: Ledger) -> str | None:
    types = ledger.ontology_types()
    if len(types) < FACET_SUGGESTION_MIN_TYPES:
        return None
    ungrouped = [item for item in types if not str(item.get("facet_id") or "").strip()]
    if len(ungrouped) < max(5, len(types) // 4):
        return None
    subject_id = f"facet-structure-{_facet_fingerprint(types)}"
    if _task_exists(ledger, ONTOLOGY_FACET_TASK, subject_id):
        return None
    return _record_task(
        ledger,
        subject_type=ONTOLOGY_FACET_TASK,
        subject_id=subject_id,
        title=f"Ontology Facet 구조 제안 · Type {len(types)}개",
        prompt=facet_suggestion_prompt(
            types=types,
            facets=ledger.ontology_facets(),
            relations=ledger.ontology_type_relations(),
            max_facets=20,
        ),
        expected_output="JSON object with facets[] and assignments[] covering the current ontology Types",
        payload={"type_count": len(types), "max_facets": 20},
    )
