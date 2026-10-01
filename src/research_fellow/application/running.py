"""Operational read model for currently running or suspended agent work."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from research_fellow.storage import Ledger
from research_fellow.application.dsl.checkpoint import decode_checkpoint_value


@dataclass(frozen=True)
class RunningWorkItem:
    run_id: str
    workflow_id: str
    status: str
    current_step: str
    source: str
    updated_at: str = ""
    waiting_interaction: str = ""
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "current_step": self.current_step,
            "source": self.source,
            "updated_at": self.updated_at,
            "waiting_interaction": self.waiting_interaction,
            "summary": self.summary,
        }


def _checkpoint_items(checkpoint_dir: Path | None) -> list[RunningWorkItem]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return []
    items: list[RunningWorkItem] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "waiting_for_interaction":
            continue
        pending = payload.get("pending_interaction") or {}
        items.append(RunningWorkItem(
            run_id=str(payload.get("checkpoint_id") or path.stem),
            workflow_id=str(payload.get("workflow_id") or "workflow"),
            status="waiting",
            current_step=str(pending.get("step_id") or ""),
            waiting_interaction=str(pending.get("interaction_id") or ""),
            source="workflow_checkpoint",
            updated_at=str(payload.get("created_at") or ""),
            summary=str(pending.get("mode") or "human interaction required"),
        ))
    return items


def _auto_research_items(ledger: Ledger, *, limit: int = 50) -> list[RunningWorkItem]:
    # Ledger exposes execution observation state without leaking DB details.
    rows = ledger.auto_research_runs(statuses=("running", "needs_attention"), limit=limit)
    result: list[RunningWorkItem] = []
    for item in rows:
        status = "attention" if item.get("status") == "needs_attention" else "running"
        summary_parts = [str(item.get("review_id") or item.get("intent_id") or "")]
        if item.get("last_error_message"):
            summary_parts.append(str(item["last_error_message"]))
        result.append(RunningWorkItem(
            run_id=str(item.get("run_id") or ""),
            workflow_id="auto_research",
            status=status,
            current_step=str(item.get("current_stage") or ""),
            source="execution_run",
            updated_at=str(item.get("updated_at") or ""),
            summary=" · ".join(part for part in summary_parts if part),
        ))
    return result



def _queued_evidence_items(ledger: Ledger, *, limit: int = 50) -> list[RunningWorkItem]:
    """Project active M1 search profiles as queued persistent work."""
    result: list[RunningWorkItem] = []
    for profile in ledger.search_profiles():
        if not profile.get("is_active"):
            continue
        result.append(RunningWorkItem(
            run_id=str(profile.get("profile_id") or profile.get("intent_id") or ""),
            workflow_id="m1_evidence_acquisition",
            status="queued",
            current_step="literature_search",
            source="search_profile",
            updated_at=str(profile.get("updated_at") or profile.get("created_at") or ""),
            summary=str(profile.get("title") or profile.get("question") or "Evidence acquisition queued"),
        ))
    return result[:limit]



def waiting_workflow_results(checkpoint_dir: Path | None) -> list[dict[str, Any]]:
    """Return suspended workflow results in the shape consumed by Attention Queue."""
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "waiting_for_interaction":
            continue
        pending = payload.get("pending_interaction")
        if not isinstance(pending, dict):
            continue
        decoded_pending = dict(pending)
        decoded_pending["inputs"] = decode_checkpoint_value(pending.get("inputs") or {})
        results.append({
            "status": "waiting_for_interaction",
            "workflow_id": str(payload.get("workflow_id") or "workflow"),
            "interaction": decoded_pending,
            "checkpoint": {
                "checkpoint_id": str(payload.get("checkpoint_id") or path.stem),
                "next_step_index": payload.get("next_step_index"),
            },
            "resume_metadata": dict(payload.get("resume_metadata") or {}) if isinstance(payload.get("resume_metadata"), dict) else {},
        })
    return results


def running_work_snapshot(
    ledger: Ledger,
    *,
    checkpoint_dir: Path | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    items = _checkpoint_items(checkpoint_dir) + _auto_research_items(ledger, limit=limit) + _queued_evidence_items(ledger, limit=limit)
    items.sort(key=lambda item: item.updated_at, reverse=True)
    counts = {"running": 0, "queued": 0, "waiting": 0, "attention": 0}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "total": len(items),
        "counts": counts,
        "items": [item.as_dict() for item in items[:limit]],
    }
