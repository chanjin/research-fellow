"""UI-independent application of researcher-reviewed Revision To-do reconciliation."""
from __future__ import annotations
from typing import Any, Mapping
from research_fellow.storage import Ledger


def apply_revision_reconciliation_review(
    ledger: Ledger,
    *,
    project_id: str,
    from_version: int,
    to_version: int,
    source: str,
    reviewed: Mapping[str, Any],
) -> dict[str, Any]:
    existing = [dict(x) for x in (reviewed.get("existing_todos") or [])]
    before_by_id = {str(item.get("todo_id") or ""): item for item in existing}
    assessments = [dict(x) for x in (reviewed.get("assessments") or [])]
    new_todos = [dict(x) for x in (reviewed.get("new_todos") or [])]
    if not assessments:
        raise ValueError("At least one reviewed Revision To-do assessment is required")
    counts = {"resolved": 0, "active": 0, "obsolete": 0, "restructured": 0, "new": len(new_todos)}
    for item in assessments:
        prior = before_by_id.get(str(item.get("todo_id") or ""), {})
        status = str(item.get("status") or "retained")
        stored_status = {"retained": "open", "modified": "modified", "researcher_review": "researcher_review"}.get(status, status)
        ledger.add_paper_project_event(project_id, "revision_todo_update", {
            **prior,
            "todo_id": item.get("todo_id"),
            "sentence_id": item.get("sentence_id") or prior.get("sentence_id", ""),
            "status": stored_status,
            "problem": item.get("updated_problem") or prior.get("problem", ""),
            "completion_criteria": item.get("updated_completion_criteria") or prior.get("completion_criteria", ""),
            "recommended_action": item.get("updated_recommended_action") or prior.get("recommended_action", "researcher_input"),
            "reconciliation_reason": item.get("reason", ""),
            "merged_into": item.get("merged_into", ""),
            "reconciled_from_version": from_version,
            "reconciled_to_version": to_version,
        })
        if status == "resolved": counts["resolved"] += 1
        elif status == "obsolete": counts["obsolete"] += 1
        elif status in {"merged", "split"}: counts["restructured"] += 1
        else: counts["active"] += 1
    for item in new_todos:
        ledger.add_paper_project_event(project_id, "revision_todo_update", {
            **item, "status": "open", "created_by": "todo_reconciliation", "created_at_version": to_version,
        })
    report = {
        "summary": reviewed.get("summary", ""),
        "counts": counts,
        "revision_achievements": reviewed.get("revision_achievements") or [],
        "next_revision_recommendations": reviewed.get("next_revision_recommendations") or [],
        "assessments": assessments,
        "new_todos": new_todos,
    }
    ledger.add_paper_project_event(project_id, "revision_todo_reconciliation_applied", {
        "from_version": from_version,
        "to_version": to_version,
        "source": source,
        "report": report,
    })
    return report
