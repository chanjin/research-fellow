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
    return {
        "total": len(items),
        "items": [item.as_dict() for item in items],
    }
