"""Read-only System / Developer Workspace projection.

This module deliberately owns no configuration, runtime, or domain state.  It
projects the existing AJD/DSL assets, contracts/bindings, autonomy policy,
runtime observation state, diagnostic logs, and architecture audit into a
single developer-facing read model.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from research_fellow.application.architecture_audit import audit_application_architecture
from research_fellow.storage import Ledger


def _iter_resources(root: Any, suffix: str) -> Iterable[Any]:
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_resources(child, suffix)
        elif child.name.endswith(suffix):
            yield child


def _yaml_rows(relative_dir: str) -> list[dict[str, Any]]:
    root = resources.files("research_fellow").joinpath(*relative_dir.split("/"))
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for resource in sorted(_iter_resources(root, ".yaml"), key=lambda item: str(item)):
        try:
            raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        except Exception as error:
            rows.append({"file": resource.name, "error": str(error), "raw": {}})
            continue
        rows.append({"file": resource.name, "error": "", "raw": dict(raw) if isinstance(raw, Mapping) else {}})
    return rows


def _agent_specification() -> dict[str, Any]:
    ajds = []
    for row in _yaml_rows("ajd"):
        raw = row["raw"]
        ajds.append({
            "file": row["file"], "id": str(raw.get("id") or ""), "role": str(raw.get("role") or ""),
            "purpose": str(raw.get("purpose") or "").strip(),
            "responsibilities": len(raw.get("responsibilities") or []),
            "memory_items": len(raw.get("memory") or []), "error": row["error"],
        })

    workflows = []
    for row in _yaml_rows("workflows"):
        raw = row["raw"]
        steps = list(raw.get("steps") or [])
        kind_counts: dict[str, int] = {}
        for step in steps:
            if isinstance(step, Mapping):
                kind = str(step.get("kind") or "unknown")
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
        workflows.append({
            "file": row["file"], "id": str(raw.get("id") or ""), "agent": str(raw.get("agent") or ""),
            "revision": raw.get("revision"), "purpose": str(raw.get("purpose") or "").strip(),
            "realizes": list(raw.get("realizes") or []), "step_count": len(steps),
            "step_kinds": kind_counts, "error": row["error"],
        })

    topologies = []
    for row in _yaml_rows("topologies"):
        raw = row["raw"]
        loops = list(raw.get("loops") or [])
        link_count = sum(len(item.get("links") or []) for item in loops if isinstance(item, Mapping))
        topologies.append({
            "file": row["file"], "id": str(raw.get("id") or ""), "revision": raw.get("revision"),
            "purpose": str(raw.get("purpose") or "").strip(), "roots": list(raw.get("roots") or []),
            "loops": len(loops), "links": link_count,
            "boundary_interactions": len(raw.get("boundary_interactions") or []), "error": row["error"],
        })
    return {"ajd": ajds, "workflows": workflows, "topologies": topologies}


def _semantic_types() -> tuple[str, list[dict[str, Any]], str]:
    path = resources.files("research_fellow").joinpath("capabilities", "semantic_types.yaml")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:
        return "", [], str(error)
    if not isinstance(raw, Mapping):
        return "", [], "semantic_types.yaml is not a mapping"
    items = []
    for item in raw.get("types") or raw.get("semantic_types") or []:
        if isinstance(item, Mapping):
            items.append({
                "id": str(item.get("id") or item.get("name") or ""),
                "description": str(item.get("description") or item.get("purpose") or "").strip(),
            })
    version = str(raw.get("semantic_types") or raw.get("version") or "")
    return version, items, ""


def _contracts() -> dict[str, Any]:
    capability_rows = _yaml_rows("capabilities/contracts")
    interaction_rows = _yaml_rows("interactions/contracts")
    capability_contracts = []
    capability_issues = []
    for row in capability_rows:
        raw = row["raw"]
        schema = str(raw.get("capability") or raw.get("interaction") or "")
        item = {
            "file": row["file"], "id": str(raw.get("id") or ""), "schema": schema,
            "purpose": str(raw.get("purpose") or "").strip(),
            "uses": len((raw.get("uses") or {}).get("required") or []) + len((raw.get("uses") or {}).get("optional") or []),
            "produces": len(raw.get("produces") or []), "error": row["error"],
        }
        capability_contracts.append(item)
        if row["error"] or not raw.get("capability"):
            capability_issues.append({
                "file": row["file"], "id": item["id"],
                "issue": row["error"] or f"expected capability contract, found {schema or 'unknown schema'}",
            })

    interaction_contracts = []
    interaction_ids: set[str] = set()
    for row in interaction_rows:
        raw = row["raw"]
        interaction_id = str(raw.get("id") or "")
        if interaction_id:
            interaction_ids.add(interaction_id)
        interaction_contracts.append({
            "file": row["file"], "id": interaction_id, "mode": str(raw.get("mode") or ""),
            "actor": str(raw.get("actor") or ""), "purpose": str(raw.get("purpose") or "").strip(),
            "requires_response": bool((raw.get("completion") or {}).get("requires_response")),
            "error": row["error"],
        })

    capability_bindings = _yaml_rows("capabilities/bindings")
    interaction_bindings = _yaml_rows("interactions/bindings")
    semantic_version, semantic_types, semantic_error = _semantic_types()
    return {
        "capabilities": capability_contracts,
        "capability_issues": capability_issues,
        "interactions": interaction_contracts,
        "interaction_ids": sorted(interaction_ids),
        "semantic_types": semantic_types,
        "semantic_type_version": semantic_version,
        "semantic_type_error": semantic_error,
        "bindings": {
            "capability_profiles": [row["file"] for row in capability_bindings],
            "interaction_profiles": [row["file"] for row in interaction_bindings],
        },
    }


def _autonomy(ledger: Ledger, interaction_ids: set[str], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    classification_rows = _yaml_rows("autonomy")
    classification_raw: dict[str, Any] = {}
    for row in classification_rows:
        if row["file"] == "classification.yaml":
            classification_raw = row["raw"]
            break
    classes = []
    for item in classification_raw.get("interactions") or []:
        if isinstance(item, Mapping):
            classes.append({
                "id": str(item.get("id") or ""), "category": str(item.get("category") or ""),
                "current_level": str(item.get("current_level") or ""), "target_level": str(item.get("target_level") or ""),
                "rationale": str(item.get("rationale") or ""),
                "escalation_dimensions": list(item.get("escalation_dimensions") or []),
            })

    policy_rows = _yaml_rows("autonomy/policies")
    policies: list[dict[str, Any]] = []
    policy_ids: set[str] = set()
    for row in policy_rows:
        raw = row["raw"]
        for item in raw.get("interactions") or []:
            if not isinstance(item, Mapping):
                continue
            interaction_id = str(item.get("id") or "")
            policy_ids.add(interaction_id)
            policies.append({
                "id": interaction_id, "default": str(item.get("default") or "escalate"),
                "required_signals": list(item.get("required_signals") or []),
                "audit": dict(item.get("audit") or {}), "profile": str(raw.get("profile") or row["file"]),
            })

    recent_events: list[dict[str, Any]] = list(ledger.autonomy_decisions(limit=50))
    if not recent_events:
        # Compatibility fallback for pre-R49 waiting checkpoints.
        for checkpoint in checkpoints:
            semantic = checkpoint.get("semantic_context") if isinstance(checkpoint.get("semantic_context"), Mapping) else {}
            for event in semantic.get("autonomy_events") or []:
                if isinstance(event, Mapping):
                    recent_events.append({**dict(event), "checkpoint_id": str(checkpoint.get("checkpoint_id") or "")})
        recent_events = recent_events[-25:][::-1]

    action_counts = {"auto": 0, "auto_notify": 0, "escalate": 0}
    escalation_reasons: dict[str, int] = {}
    for event in recent_events:
        action = str(event.get("action") or "")
        if action in action_counts:
            action_counts[action] += 1
        if action == "escalate":
            for reason in event.get("reasons") or []:
                key = str(reason).split("=", 1)[0].replace("missing:", "missing ")
                escalation_reasons[key] = escalation_reasons.get(key, 0) + 1

    level_counts = {level: 0 for level in ("H0", "H1", "H2", "H3")}
    for item in classes:
        level = item["current_level"]
        if level in level_counts:
            level_counts[level] += 1
    uncovered = sorted(interaction_ids - policy_ids)
    unknown = sorted(policy_ids - interaction_ids)
    return {
        "classification_version": str(classification_raw.get("classification") or ""),
        "classes": classes, "level_counts": level_counts, "policies": policies,
        "policy_coverage": {
            "interaction_count": len(interaction_ids), "policy_count": len(policy_ids),
            "covered": len(interaction_ids & policy_ids), "uncovered": uncovered, "unknown": unknown,
        },
        "recent_decisions": recent_events[:25],
        "decision_counts": action_counts,
        "escalation_reasons": dict(sorted(escalation_reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


def _read_checkpoints(checkpoint_dir: Path | None, *, limit: int) -> list[dict[str, Any]]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(checkpoint_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            rows.append({"checkpoint_id": path.stem, "status": "invalid", "error": str(error), "file": path.name})
            continue
        if isinstance(raw, Mapping):
            rows.append({**dict(raw), "file": path.name, "error": ""})
        if len(rows) >= limit:
            break
    return rows


def _runtime(ledger: Ledger, checkpoints: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    statuses = ("running", "needs_attention", "completed", "failed")
    runs = ledger.auto_research_runs(statuses=statuses, limit=limit)
    failures = ledger.auto_research_failures(status="needs_attention", limit=limit)
    resolved_failures = ledger.auto_research_failures(status="resolved", limit=limit)
    checkpoint_views = []
    for item in checkpoints:
        pending = item.get("pending_interaction") if isinstance(item.get("pending_interaction"), Mapping) else {}
        checkpoint_views.append({
            "checkpoint_id": str(item.get("checkpoint_id") or item.get("file") or ""),
            "workflow_id": str(item.get("workflow_id") or ""), "status": str(item.get("status") or ""),
            "created_at": str(item.get("created_at") or ""), "next_step_index": item.get("next_step_index"),
            "interaction_id": str(pending.get("interaction_id") or ""), "step_id": str(pending.get("step_id") or ""),
            "error": str(item.get("error") or ""),
        })
    return {
        "counts": {
            "runs": len(runs), "running": sum(1 for item in runs if item.get("status") == "running"),
            "waiting_checkpoints": sum(1 for item in checkpoint_views if item.get("status") == "waiting_for_interaction"),
            "failures_needing_attention": len(failures),
        },
        "runs": runs, "checkpoints": checkpoint_views,
        "failures": failures, "resolved_failures": resolved_failures,
    }


def _prompt_logs(ledger: Ledger, prompt_log_path: Path | None, *, limit: int) -> dict[str, Any]:
    records = ledger.llm_calls(limit=limit)
    mirror_count = 0
    mirror_error = ""
    if prompt_log_path is not None and prompt_log_path.exists():
        try:
            with prompt_log_path.open("r", encoding="utf-8") as handle:
                mirror_count = sum(1 for line in handle if line.strip())
        except OSError as error:
            mirror_error = str(error)
    views = []
    for item in records:
        diagnostics = dict(item.get("diagnostics") or {})
        views.append({
            "call_id": str(item.get("call_id") or ""), "created_at": str(item.get("created_at") or ""),
            "profile_name": str(item.get("profile_name") or ""), "model": str(item.get("model") or ""),
            "error": str(item.get("error") or ""),
            "prompt_chars": diagnostics.get("original_prompt_chars", diagnostics.get("request_prompt_chars")),
            "prompt_estimated_tokens": diagnostics.get("original_prompt_estimated_tokens", diagnostics.get("request_prompt_estimated_tokens")),
            "response_chars": diagnostics.get("response_chars"), "elapsed_seconds": diagnostics.get("elapsed_seconds"),
        })
    return {
        "count": len(records), "records": views,
        "jsonl_path": str(prompt_log_path) if prompt_log_path is not None else "",
        "jsonl_count": mirror_count, "jsonl_error": mirror_error,
    }


def _architecture_audit() -> dict[str, Any]:
    try:
        report = audit_application_architecture()
    except Exception as error:
        return {"status": "issue", "error": str(error), "report": {}}
    return {"status": "pass", "error": "", "report": report}


def system_workspace_snapshot(
    ledger: Ledger,
    *,
    checkpoint_dir: Path | None = None,
    prompt_log_path: Path | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Return a read-only projection for design/runtime inspection."""
    limit = max(1, min(int(limit), 200))
    checkpoints = _read_checkpoints(checkpoint_dir, limit=limit)
    contracts = _contracts()
    specification = _agent_specification()
    autonomy = _autonomy(ledger, set(contracts.pop("interaction_ids")), checkpoints)
    runtime = _runtime(ledger, checkpoints, limit=limit)
    prompt_logs = _prompt_logs(ledger, prompt_log_path, limit=limit)
    audit = _architecture_audit()
    return {
        "counts": {
            "ajd": len(specification["ajd"]), "workflows": len(specification["workflows"]),
            "topologies": len(specification["topologies"]), "capabilities": len(contracts["capabilities"]),
            "interactions": len(contracts["interactions"]), "semantic_types": len(contracts["semantic_types"]),
            "architecture_issues": len(contracts["capability_issues"]) + (1 if audit["status"] != "pass" else 0),
        },
        "specification": specification, "contracts": contracts, "autonomy": autonomy,
        "runtime": runtime,
        "development": {"prompt_logs": prompt_logs, "execution_logs": {"runs": runtime["runs"], "failures": runtime["failures"]}, "architecture_audit": audit},
    }
