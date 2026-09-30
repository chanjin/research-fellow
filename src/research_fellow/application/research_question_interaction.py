"""Application service for researcher triage of durable research questions."""
from __future__ import annotations

from typing import Any, Mapping

_ALLOWED = {"interested", "exploring", "hold", "completed", "rejected"}


def apply_research_question_triage(ledger: Any, resolution: Mapping[str, Any]) -> dict[str, Any]:
    rq_id = str(resolution.get("rq_id") or "").strip()
    status = str(resolution.get("status") or "").strip()
    if not rq_id:
        raise ValueError("research question id is required")
    if status not in _ALLOWED:
        raise ValueError(f"unsupported research question status: {status!r}")
    ledger.update_research_question_status(rq_id, status)
    return {"rq_id": rq_id, "status": status, "note": str(resolution.get("note") or "").strip()}
