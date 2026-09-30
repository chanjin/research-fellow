"""UI-independent application of researcher decisions on revision proposals."""
from __future__ import annotations
from typing import Any, Iterable, Mapping
from research_fellow.application.literature_discovery_formats import build_paper_labels
from research_fellow.storage import Ledger


def apply_revision_application_resolutions(
    ledger: Ledger,
    candidates: Iterable[Mapping[str, Any]],
    resolutions: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    by_id = {str(item.get("candidate_id") or ""): dict(item) for item in candidates}
    counts = {"approved": 0, "deferred": 0, "rejected": 0}
    for resolution in resolutions:
        candidate_id = str(resolution.get("candidate_id") or "").strip()
        decision = str(resolution.get("decision") or "").strip()
        note = str(resolution.get("note") or "")
        if not candidate_id or candidate_id not in by_id:
            raise ValueError(f"Unknown revision application candidate: {candidate_id!r}")
        if decision not in counts:
            raise ValueError(f"Unsupported revision application decision: {decision}")
        candidate = by_id[candidate_id]
        project_id = str(candidate["project_id"])
        todo_id = str(candidate["todo_id"])
        if decision != "approved":
            ledger.add_paper_project_event(
                project_id,
                "revision_resolution_decision",
                {"todo_id": todo_id, "candidate_id": candidate_id, "decision": decision, "note": note},
            )
            counts[decision] += 1
            continue

        saved_reference_ids: list[str] = []
        for paper in list(candidate.get("reference_papers") or []):
            reference = ledger.upsert_literature_reference(
                topic=f"{candidate.get('project_title') or '논문 프로젝트'} · References",
                research_context=f"Revision To-do {todo_id}: {candidate.get('todo_problem') or ''}",
                paper={**dict(paper), "paper_project_id": project_id, "revision_todo_id": todo_id},
                labels=list(paper.get("labels", [])) or build_paper_labels(dict(paper)),
                status="selected",
            )
            if reference.get("reference_id"):
                saved_reference_ids.append(str(reference["reference_id"]))

        proposed_manuscript = dict(candidate["proposed_manuscript"])
        proposal_diff = list(candidate.get("proposal_diff") or [])
        todo = dict(candidate["todo"])
        payload = dict(candidate.get("proposal_payload") or {})
        ledger.add_paper_project_event(
            project_id,
            "manuscript_version",
            {
                "manuscript": proposed_manuscript,
                "source": "researcher_approved_resolution",
                "diff": proposal_diff,
                "todo_id": todo_id,
                "todo_label": todo.get("label", ""),
                "todo_priority": todo.get("priority", ""),
            },
        )
        ledger.add_paper_project_event(
            project_id,
            "revision_resolution_applied",
            {
                "todo_id": todo_id,
                "from_version": payload.get("from_version"),
                "to_version": proposed_manuscript.get("version"),
                "paper_ids": payload.get("selected_paper_ids", []),
                "card_ids": payload.get("selected_card_ids", []),
                "reference_ids": saved_reference_ids,
                "connection_note": payload.get("connection_note", ""),
                "researcher_note": note,
            },
        )
        ledger.add_paper_project_event(
            project_id,
            "revision_todo_update",
            {
                **{key: value for key, value in todo.items() if key != "latest_update"},
                "status": "resolved",
                "resolution_source": "researcher_approved_resolution",
                "selected_card_ids": payload.get("selected_card_ids", []),
            },
        )
        counts[decision] += 1
    return counts
