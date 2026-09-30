"""Read-only operational activity feed derived from durable research phenomena."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.storage import Ledger


@dataclass(frozen=True)
class ActivityItem:
    activity_id: str
    created_at: str
    actor: str
    action: str
    title: str
    category: str
    source_type: str
    source_id: str
    autonomy: str = ""
    reversible: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


_LABELS = {
    "knowledge_update": ("M1", "Knowledge updated", "knowledge"),
    "advice_report": ("M2", "Advisory/report produced", "research"),
    "decision_request": ("Agent", "Human decision requested", "attention"),
    "decision": ("Researcher", "Decision recorded", "human"),
    "curation_intent": ("M2", "Literature curation delegated", "research"),
    "research_update": ("Researcher", "Research state updated", "research"),
}


def activity_feed_snapshot(ledger: Ledger, *, limit: int = 50) -> dict[str, Any]:
    rows = ledger.phenomena()[: max(1, min(int(limit), 200))]
    items: list[ActivityItem] = []
    for index, row in enumerate(rows):
        phenomenon = str(row.get("phenomenon_type") or "activity")
        actor, action, category = _LABELS.get(
            phenomenon,
            (str(row.get("producer") or "Agent"), phenomenon.replace("_", " ").title(), "other"),
        )
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        source_id = str(row.get("phenomenon_id") or row.get("case_id") or index)
        title = str(payload.get("title") or payload.get("finding") or payload.get("question") or action)
        items.append(ActivityItem(
            activity_id=f"{phenomenon}:{source_id}",
            created_at=str(row.get("created_at") or ""),
            actor=actor,
            action=action,
            title=title,
            category=category,
            source_type=phenomenon,
            source_id=source_id,
            autonomy=str(payload.get("autonomy") or ""),
            reversible=payload.get("reversible") if isinstance(payload.get("reversible"), bool) else None,
        ))

    # Autonomy decisions are operational audit events in the same Ledger, not new
    # research/domain state. Surface AUTO_NOTIFY and escalations in the activity feed.
    for event in ledger.autonomy_decisions(limit=max(1, min(int(limit), 50))):
        action = str(event.get("action") or "")
        if action not in {"auto_notify", "escalate"}:
            continue
        interaction_id = str(event.get("interaction_id") or "")
        reasons = ", ".join(str(x) for x in (event.get("reasons") or []))
        items.append(ActivityItem(
            activity_id=f"autonomy:{event.get('event_id')}",
            created_at=str(event.get("created_at") or ""),
            actor="Agent",
            action="Autonomy executed + notify" if action == "auto_notify" else "Human escalation requested",
            title=f"{interaction_id}" + (f" · {reasons}" if reasons else ""),
            category="attention" if action == "escalate" else "autonomy",
            source_type="autonomy_decision",
            source_id=str(event.get("event_id") or ""),
            autonomy=action,
        ))

    # Ontology review state is durable outside the generic phenomenon ledger.
    # Project it into Activity rather than duplicating it as another write model.
    for review in ledger.ontology_change_reviews(limit=max(1, min(int(limit), 50))):
        status = str(review.get("status") or "proposed")
        comment = str(review.get("researcher_comment") or "").strip()
        if status == "proposed" and not comment:
            continue
        review_id = str(review.get("review_id") or "")
        items.append(ActivityItem(
            activity_id=f"ontology_change_review:{review_id}:{status}:{bool(comment)}",
            created_at=str(review.get("updated_at") or review.get("created_at") or ""),
            actor="Researcher" if comment or status in {"approved", "rejected"} else "Agent",
            action="Ontology review updated",
            title=str((review.get("proposal") or {}).get("summary") or "Ontology change review"),
            category="human" if comment or status != "proposed" else "knowledge",
            source_type="ontology_change_review",
            source_id=review_id,
        ))

    items.sort(key=lambda item: item.created_at, reverse=True)
    items = items[: max(1, min(int(limit), 200))]
    return {
        "total": len(items),
        "items": [item.as_dict() for item in items],
    }
