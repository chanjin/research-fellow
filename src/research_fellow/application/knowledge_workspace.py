"""Job-centred knowledge workspace read model.

The Knowledge Workspace owns no knowledge state. It projects approved semantic
memory, evidence, relations, ontology structure, and recent knowledge-change
phenomena into one view of what the Research Fellow currently knows and where
structure or review is still needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class KnowledgeCardView:
    card_id: str
    title: str
    claim: str
    status: str
    evidence_level: str
    source_kind: str
    supporting_evidence_count: int
    relation_count: int
    ontology_types: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["ontology_types"] = list(self.ontology_types)
        return data


@dataclass(frozen=True)
class KnowledgeChangeView:
    update_id: str
    title: str
    created_at: str
    subject_id: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KnowledgeGapView:
    gap_id: str
    priority: str
    kind: str
    title: str
    reason: str
    target_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _relation_counts(relations: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for relation in relations:
        for key in (str(relation.get("source_card_id") or ""), str(relation.get("target_card_id") or "")):
            if key:
                result[key] = result.get(key, 0) + 1
    return result


def _card_views(
    memory: KnowledgeMemory,
    relation_memory: RelationMemory,
    ledger: Ledger,
    *,
    limit: int,
) -> tuple[list[KnowledgeCardView], list[dict[str, Any]]]:
    cards = memory.all(limit=limit)
    relations = relation_memory.all()
    counts = _relation_counts(relations)
    views: list[KnowledgeCardView] = []
    for card in cards:
        card_id = str(card.get("card_id") or "")
        ontology_names = tuple(
            str(item.get("name") or item.get("type_id") or "")
            for item in ledger.ontology_types_for_card(card_id)
            if item.get("name") or item.get("type_id")
        )
        views.append(KnowledgeCardView(
            card_id=card_id,
            title=str(card.get("title") or "Untitled knowledge card"),
            claim=str(card.get("claim") or ""),
            status=str(card.get("status") or "verified"),
            evidence_level=str(card.get("evidence_level") or "provisional"),
            source_kind=str(card.get("source_kind") or ""),
            supporting_evidence_count=len(card.get("supporting_evidence") or []),
            relation_count=counts.get(card_id, 0),
            ontology_types=ontology_names,
        ))
    return views, relations


def _recent_changes(ledger: Ledger, *, limit: int) -> list[KnowledgeChangeView]:
    reviewed = ledger.reviewed_knowledge_update_ids()
    result: list[KnowledgeChangeView] = []
    for row in ledger.phenomena(type_="knowledge_update")[:limit]:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        update_id = str(row.get("phenomenon_id") or "")
        result.append(KnowledgeChangeView(
            update_id=update_id,
            title=str(payload.get("title") or payload.get("claim") or payload.get("finding") or "Knowledge updated"),
            created_at=str(row.get("created_at") or ""),
            subject_id=str(row.get("subject_id") or payload.get("card_id") or ""),
            status="reviewed" if update_id in reviewed else "new",
            detail=str(payload.get("summary") or payload.get("change_summary") or ""),
        ))
    return result


def _knowledge_gaps(
    cards: list[KnowledgeCardView],
    *,
    pending_ontology_reviews: int,
    limit: int,
) -> list[KnowledgeGapView]:
    gaps: list[KnowledgeGapView] = []
    for card in cards:
        if card.status == "contested":
            gaps.append(KnowledgeGapView(
                gap_id=f"contested:{card.card_id}", priority="high", kind="conflict",
                title=f"Resolve contested knowledge: {card.title}",
                reason="The approved semantic memory marks this claim as contested.", target_id=card.card_id,
            ))
        if card.evidence_level == "provisional":
            gaps.append(KnowledgeGapView(
                gap_id=f"evidence:{card.card_id}", priority="medium", kind="evidence",
                title=f"Strengthen evidence: {card.title}",
                reason="The claim is still supported only at a provisional evidence level.", target_id=card.card_id,
            ))
        if not card.ontology_types:
            gaps.append(KnowledgeGapView(
                gap_id=f"ontology:{card.card_id}", priority="medium", kind="structure",
                title=f"Structure knowledge: {card.title}",
                reason="The card has not yet been assigned to an ontology type.", target_id=card.card_id,
            ))
        if card.relation_count == 0:
            gaps.append(KnowledgeGapView(
                gap_id=f"relation:{card.card_id}", priority="low", kind="structure",
                title=f"Connect knowledge: {card.title}",
                reason="The approved card is isolated from the current knowledge-relation graph.", target_id=card.card_id,
            ))
    if pending_ontology_reviews:
        gaps.append(KnowledgeGapView(
            gap_id="ontology-review", priority="high", kind="ontology_review",
            title=f"Review {pending_ontology_reviews} ontology change proposal(s)",
            reason="Proposed structural changes are waiting for researcher oversight.",
        ))
    rank = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda item: (rank.get(item.priority, 9), item.title.lower()))
    return gaps[:limit]



def _resolved_paper_origins(ledger: Ledger, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in values:
        item = dict(raw)
        if str(item.get("origin_type") or "") == "researcher_question":
            rq = ledger.research_question_thread(str(item.get("origin_id") or "")) or {}
            if rq:
                item["research_question"] = str(rq.get("question") or item.get("research_question") or "")
                item["researcher_comment"] = str((rq.get("source_payload") or {}).get("researcher_comment") or rq.get("research_context") or "")
        result.append(item)
    return result


def _evidence_library(memory: KnowledgeMemory, ledger: Ledger, *, limit: int) -> list[dict[str, Any]]:
    cards = memory.all(limit=500)
    approved_by_paper: dict[str, list[dict[str, Any]]] = {}
    approved_by_source: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        paper_id = str(provenance.get("paper_id") or "").strip()
        source_name = str(provenance.get("source_name") or "").strip().casefold()
        if paper_id:
            approved_by_paper.setdefault(paper_id, []).append(card)
        if source_name:
            approved_by_source.setdefault(source_name, []).append(card)
        for evidence in list(card.get("supporting_evidence") or []):
            if not isinstance(evidence, dict):
                continue
            ev_paper_id = str(evidence.get("paper_id") or "").strip()
            ev_source = str(evidence.get("source_name") or "").strip().casefold()
            if ev_paper_id:
                approved_by_paper.setdefault(ev_paper_id, []).append(card)
            if ev_source:
                approved_by_source.setdefault(ev_source, []).append(card)

    pending_by_paper: dict[str, list[dict[str, Any]]] = {}
    for req in ledger.phenomena(type_="decision_request", status="proposed"):
        if str(req.get("subject_type") or "") != "knowledge_card":
            continue
        payload = dict(req.get("payload") or {})
        paper_id = str(payload.get("paper_id") or "").strip()
        card = payload.get("card") if isinstance(payload.get("card"), dict) else {}
        if paper_id and card:
            pending_by_paper.setdefault(paper_id, []).append({
                "request_id": str(req.get("phenomenon_id") or ""),
                "rq_id": str(payload.get("rq_id") or (card.get("provenance") or {}).get("research_question_id") or ""),
                "intent_id": str(payload.get("intent_id") or ""),
                "research_question": str(payload.get("research_question") or ""),
                "card_id": str(card.get("card_id") or req.get("subject_id") or ""),
                "title": str(card.get("title") or ""),
                "claim": str(card.get("claim") or ""),
            })

    result: list[dict[str, Any]] = []
    for paper in ledger.shelf_papers(limit=limit):
        paper_id = str(paper.get("paper_id") or "")
        title = str(paper.get("title") or "Untitled paper")
        origins = _resolved_paper_origins(ledger, list(paper.get("origin_links") or []))
        rq_origins = {
            str(item.get("origin_id") or ""): item
            for item in origins
            if str(item.get("origin_type") or "") == "researcher_question" and str(item.get("origin_id") or "")
        }

        approved = approved_by_paper.get(paper_id, []) or approved_by_source.get(title.casefold(), [])
        dedup_approved: list[dict[str, Any]] = []
        seen_cards: set[str] = set()
        for card in approved:
            card_id = str(card.get("card_id") or "")
            if not card_id or card_id in seen_cards:
                continue
            seen_cards.add(card_id)
            provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
            dedup_approved.append({
                "card_id": card_id,
                "rq_id": str(provenance.get("research_question_id") or ""),
                "title": str(card.get("title") or ""),
                "claim": str(card.get("claim") or ""),
            })

        question_analyses = ledger.paper_question_analyses(paper_id)
        legacy_analysis = ledger.paper_analysis(paper_id) or {}
        bundles: dict[str, dict[str, Any]] = {}
        for row in question_analyses:
            rq_id = str(row.get("research_question_id") or "")
            origin = rq_origins.get(rq_id, {})
            rq = ledger.research_question_thread(rq_id) if rq_id else None
            bundles[rq_id] = {
                "rq_id": rq_id,
                "question": str((rq or {}).get("question") or row.get("research_question") or origin.get("research_question") or ""),
                "researcher_comment": str(((rq or {}).get("source_payload") or {}).get("researcher_comment") or origin.get("researcher_comment") or ""),
                "intent_id": str(row.get("intent_id") or ""),
                "summary": str(row.get("summary") or ""),
                "review": str(row.get("reading_raw_output") or ""),
                "updated_at": str(row.get("updated_at") or ""),
                "pending_knowledge_cards": [],
                "approved_knowledge_cards": [],
            }

        # Preserve visibility for linked questions even before a full-text review exists.
        for rq_id, origin in rq_origins.items():
            bundles.setdefault(rq_id, {
                "rq_id": rq_id,
                "question": str(origin.get("research_question") or ""),
                "researcher_comment": str(origin.get("researcher_comment") or ""),
                "intent_id": "", "summary": "", "review": "", "updated_at": "",
                "pending_knowledge_cards": [], "approved_knowledge_cards": [],
            })

        # Legacy workspaces only had one paper-level analysis. Attach it to a matching
        # question when possible, otherwise expose it as a legacy interpretation.
        if legacy_analysis and not question_analyses and str(legacy_analysis.get("summary") or "").strip():
            legacy_question = str(legacy_analysis.get("research_question") or "").strip()
            match_id = next((rq_id for rq_id, b in bundles.items() if legacy_question and b.get("question") == legacy_question), "")
            key = match_id or "legacy"
            bundle = bundles.setdefault(key, {
                "rq_id": match_id, "question": legacy_question or "Legacy review", "researcher_comment": "",
                "intent_id": "", "summary": "", "review": "", "updated_at": "",
                "pending_knowledge_cards": [], "approved_knowledge_cards": [],
            })
            bundle["summary"] = str(legacy_analysis.get("summary") or "")
            bundle["review"] = str(legacy_analysis.get("reading_raw_output") or "")
            bundle["updated_at"] = str(legacy_analysis.get("updated_at") or "")

        for item in pending_by_paper.get(paper_id, []):
            rq_id = str(item.get("rq_id") or "")
            if not rq_id and item.get("intent_id"):
                linked = ledger.research_questions_for_intent(str(item.get("intent_id") or ""))
                rq_id = str((linked[0] if linked else {}).get("rq_id") or "")
            key = rq_id or "unscoped"
            bundle = bundles.setdefault(key, {
                "rq_id": rq_id, "question": str(item.get("research_question") or ""), "researcher_comment": "",
                "intent_id": str(item.get("intent_id") or ""), "summary": "", "review": "", "updated_at": "",
                "pending_knowledge_cards": [], "approved_knowledge_cards": [],
            })
            bundle["pending_knowledge_cards"].append(item)

        for card in dedup_approved:
            rq_id = str(card.get("rq_id") or "")
            if not rq_id and len(rq_origins) == 1:
                rq_id = next(iter(rq_origins))
            key = rq_id or "unscoped"
            bundle = bundles.setdefault(key, {
                "rq_id": rq_id, "question": str(rq_origins.get(rq_id, {}).get("research_question") or ""),
                "researcher_comment": str(rq_origins.get(rq_id, {}).get("researcher_comment") or ""),
                "intent_id": "", "summary": "", "review": "", "updated_at": "",
                "pending_knowledge_cards": [], "approved_knowledge_cards": [],
            })
            bundle["approved_knowledge_cards"].append(card)

        abstract = str(paper.get("abstract") or "").strip()
        common_summary = abstract
        common_summary_source = "abstract" if abstract else ""
        if not common_summary and legacy_analysis:
            common_summary = str(legacy_analysis.get("summary") or "").strip()
            common_summary_source = "legacy_analysis" if common_summary else ""

        questions = ledger.paper_reading_questions(paper_id)
        question_views = sorted(
            bundles.values(),
            key=lambda row: (str(row.get("updated_at") or ""), str(row.get("question") or "")),
            reverse=True,
        )
        result.append({
            "paper_id": paper_id, "title": title,
            "authors": list(paper.get("authors") or []),
            "publication_year": str(paper.get("publication_year") or ""),
            "source_url": str(paper.get("source_url") or ""),
            "abstract_url": str(paper.get("abstract_url") or paper.get("source_url") or ""),
            "full_text_url": str(paper.get("full_text_url") or ""),
            "pdf_url": str(paper.get("pdf_url") or ""),
            "pdf_path": str(paper.get("pdf_path") or ""),
            "source_id": str(paper.get("source_id") or ""),
            "shelf_status": str(paper.get("shelf_status") or "reference"),
            "reading_status": str(paper.get("reading_status") or "unread"),
            "intake_source": str(paper.get("intake_source") or ""),
            "origin_links": origins,
            "labels": list(paper.get("labels") or []),
            "summary": common_summary,
            "summary_source": common_summary_source,
            "researcher_note": str(legacy_analysis.get("researcher_note") or ""),
            "has_analysis": bool(question_analyses or legacy_analysis),
            "has_generated_analysis": bool(question_analyses or str(legacy_analysis.get("generated_at") or "").strip()),
            "reading_question_count": len(questions),
            "knowledge_card_count": len(dedup_approved),
            "pending_knowledge_card_count": sum(len(x.get("pending_knowledge_cards") or []) for x in question_views),
            "knowledge_cards": dedup_approved[:8],
            "question_interpretations": question_views,
            "asset_available": bool(str(paper.get("pdf_path") or "").strip()),
        })
    return result

def knowledge_workspace_snapshot(
    memory: KnowledgeMemory,
    relation_memory: RelationMemory,
    ledger: Ledger,
    *,
    limit: int = 30,
) -> dict[str, Any]:
    """Return a read-only job view of approved knowledge and its structural health."""
    limit = max(1, min(int(limit), 100))
    cards, relations = _card_views(memory, relation_memory, ledger, limit=limit)
    changes = _recent_changes(ledger, limit=limit)
    ontology = ledger.ontology_snapshot()
    pending_reviews = ledger.ontology_change_reviews(status="proposed", limit=100)
    versions = ledger.ontology_versions(limit=5)
    gaps = _knowledge_gaps(cards, pending_ontology_reviews=len(pending_reviews), limit=limit)
    evidence_library = _evidence_library(memory, ledger, limit=max(limit, 500))

    evidence_levels: dict[str, int] = {}
    statuses: dict[str, int] = {}
    source_kinds: dict[str, int] = {}
    for card in cards:
        evidence_levels[card.evidence_level] = evidence_levels.get(card.evidence_level, 0) + 1
        statuses[card.status] = statuses.get(card.status, 0) + 1
        source_kinds[card.source_kind] = source_kinds.get(card.source_kind, 0) + 1

    assigned_ids = {str(item.get("card_id") or "") for item in ontology.get("assignments", [])}
    typed_cards = sum(1 for card in cards if card.card_id in assigned_ids or card.ontology_types)
    reinforced = sum(1 for card in cards if card.supporting_evidence_count > 0)
    contradicted_relations = sum(1 for item in relations if str(item.get("relation_type") or "") == "contradicts")

    return {
        "counts": {
            "cards": len(cards),
            "relations": len(relations),
            "typed_cards": typed_cards,
            "untyped_cards": max(0, len(cards) - typed_cards),
            "reinforced_cards": reinforced,
            "contested_cards": statuses.get("contested", 0),
            "new_changes": sum(1 for item in changes if item.status == "new"),
            "gaps": len(gaps),
            "papers": len(evidence_library),
            "papers_with_knowledge": sum(1 for item in evidence_library if int(item.get("knowledge_card_count") or 0) > 0),
            "analyzed_papers": sum(1 for item in evidence_library if item.get("has_analysis")),
        },
        "evidence_levels": evidence_levels,
        "statuses": statuses,
        "source_kinds": source_kinds,
        "cards": [item.as_dict() for item in cards],
        "changes": [item.as_dict() for item in changes],
        "relations": {
            "total": len(relations),
            "contradictions": contradicted_relations,
            "by_type": _count_values(relations, "relation_type"),
        },
        "ontology": {
            "facets": len(ontology.get("facets", [])),
            "types": len(ontology.get("types", [])),
            "relations": len(ontology.get("relations", [])),
            "assignments": len(ontology.get("assignments", [])),
            "pending_reviews": len(pending_reviews),
            "latest_version": versions[0] if versions else None,
            "top_types": [
                {
                    "type_id": str(item.get("type_id") or ""),
                    "name": str(item.get("name") or ""),
                    "facet_name": str(item.get("facet_name") or ""),
                    "card_count": int(item.get("card_count") or 0),
                }
                for item in sorted(ontology.get("types", []), key=lambda row: int(row.get("card_count") or 0), reverse=True)[:12]
            ],
        },
        "evidence_library": evidence_library,
        "gaps": [item.as_dict() for item in gaps],
    }


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        result[value] = result.get(value, 0) + 1
    return result
