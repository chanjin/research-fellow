"""Machine-readable AJD contracts and workflow traceability validation.

This module intentionally keeps the AJD contract small.  It does not attempt to
encode the full prose AJD.  It exposes only the stable identifiers needed for
static workflow validation: agent identity, responsibility identifiers, and
memory/knowledge assets with their allowed access modes.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any, Iterable

import yaml


AJD_VERSION = "ajd/v0.1"
_MEMORY_ACCESS = {"read", "write"}


def _iter_yaml_resources(root: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    for child in root.iterdir():
        rel = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            yield from _iter_yaml_resources(child, rel)
        elif child.name.endswith((".yaml", ".yml")):
            yield rel, child


def _validate_ajd(raw: dict[str, Any], relative_path: str) -> None:
    if raw.get("ajd") != AJD_VERSION:
        raise ValueError(f"Unsupported AJD version in {relative_path}: {raw.get('ajd')!r}")
    agent_id = str(raw.get("id") or "").strip()
    if not agent_id:
        raise ValueError(f"AJD needs id: {relative_path}")

    responsibilities = raw.get("responsibilities")
    if not isinstance(responsibilities, list) or not responsibilities:
        raise ValueError(f"AJD responsibilities must be a non-empty list: {relative_path}")
    seen_responsibilities: set[str] = set()
    for item in responsibilities:
        if not isinstance(item, dict) or not str(item.get("id") or "").strip():
            raise ValueError(f"AJD responsibility needs id: {relative_path}")
        responsibility_id = str(item["id"])
        if responsibility_id in seen_responsibilities:
            raise ValueError(f"Duplicate AJD responsibility {responsibility_id}: {relative_path}")
        seen_responsibilities.add(responsibility_id)

    memory = raw.get("memory") or []
    if not isinstance(memory, list):
        raise ValueError(f"AJD memory must be a list: {relative_path}")
    seen_memory: set[str] = set()
    for asset in memory:
        if not isinstance(asset, dict) or not str(asset.get("id") or "").strip():
            raise ValueError(f"AJD memory asset needs id: {relative_path}")
        asset_id = str(asset["id"])
        if asset_id in seen_memory:
            raise ValueError(f"Duplicate AJD memory asset {asset_id}: {relative_path}")
        seen_memory.add(asset_id)
        access = asset.get("access") or []
        if not isinstance(access, list) or not access:
            raise ValueError(f"AJD memory asset {asset_id} needs access: {relative_path}")
        invalid = [mode for mode in access if mode not in _MEMORY_ACCESS]
        if invalid:
            raise ValueError(f"Invalid AJD memory access {invalid} for {asset_id}: {relative_path}")


@lru_cache(maxsize=1)
def load_ajd_catalog() -> dict[str, tuple[str, dict[str, Any]]]:
    root = resources.files("research_fellow").joinpath("ajd")
    catalog: dict[str, tuple[str, dict[str, Any]]] = {}
    for relative_path, resource in _iter_yaml_resources(root):
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid AJD definition: {relative_path}")
        _validate_ajd(raw, relative_path)
        agent_id = str(raw["id"])
        if agent_id in catalog:
            other_path = catalog[agent_id][0]
            raise ValueError(f"Duplicate AJD id {agent_id}: {other_path}, {relative_path}")
        catalog[agent_id] = (relative_path, raw)
    return catalog


def _responsibility_ids(raw: dict[str, Any]) -> set[str]:
    return {str(item["id"]) for item in (raw.get("responsibilities") or [])}


def _memory_permissions(raw: dict[str, Any]) -> dict[str, set[str]]:
    return {
        str(item["id"]): {str(mode) for mode in (item.get("access") or [])}
        for item in (raw.get("memory") or [])
    }


def _realizes_list(raw: dict[str, Any], *, path: str) -> list[str]:
    value = raw.get("realizes")
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Workflow realizes must be a list: {path}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Workflow realizes entries must be non-empty strings: {path}")
        result.append(item)
    return result


def _memory_refs(raw: dict[str, Any], *, path: str) -> list[tuple[str, str, str]]:
    """Return (access, asset, location) refs from workflow and step scopes."""
    refs: list[tuple[str, str, str]] = []

    def add_memory(spec: Any, location: str) -> None:
        if spec is None:
            return
        if not isinstance(spec, dict):
            raise ValueError(f"memory must be a mapping at {location}: {path}")
        unknown = sorted(set(spec) - {"reads", "writes"})
        if unknown:
            raise ValueError(f"Unknown memory keys {unknown} at {location}: {path}")
        for field, access in (("reads", "read"), ("writes", "write")):
            values = spec.get(field) or []
            if not isinstance(values, list):
                raise ValueError(f"memory.{field} must be a list at {location}: {path}")
            for item in values:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"memory.{field} entries must be non-empty strings at {location}: {path}")
                refs.append((access, item, location))

    add_memory(raw.get("memory"), "workflow")
    for step in raw.get("steps") or []:
        add_memory(step.get("memory"), f"step {step.get('id')}")
    return refs


def validate_workflow_against_ajd(
    raw: dict[str, Any],
    path: str,
    *,
    ajd_catalog: dict[str, tuple[str, dict[str, Any]]] | None = None,
    workflow_catalog: dict[str, tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Validate one workflow against its owning agent's AJD contract."""
    agents = ajd_catalog or load_ajd_catalog()
    agent_id = str(raw.get("agent") or "")
    if agent_id not in agents:
        raise ValueError(f"Workflow {raw.get('id')} references unknown AJD agent {agent_id!r}: {path}")
    _ajd_path, ajd = agents[agent_id]

    realizes = _realizes_list(raw, path=path)
    responsibilities = _responsibility_ids(ajd)
    unknown_responsibilities = [item for item in realizes if item not in responsibilities]
    if unknown_responsibilities:
        raise ValueError(
            f"Workflow {raw.get('id')} realizes responsibilities not owned by {agent_id}: "
            f"{unknown_responsibilities}: {path}"
        )

    permissions = _memory_permissions(ajd)
    memory_usage: list[dict[str, str]] = []
    for access, asset, location in _memory_refs(raw, path=path):
        if asset not in permissions:
            raise ValueError(
                f"Workflow {raw.get('id')} references undeclared AJD memory asset {asset!r} "
                f"for {agent_id} at {location}: {path}"
            )
        if access not in permissions[asset]:
            raise ValueError(
                f"Workflow {raw.get('id')} requests {access} access to AJD memory asset {asset!r}, "
                f"allowed={sorted(permissions[asset])} at {location}: {path}"
            )
        memory_usage.append({"asset": asset, "access": access, "location": location})

    workflows = workflow_catalog or {}
    for step in raw.get("steps") or []:
        kind = str(step.get("kind") or "")
        actor = str(step.get("actor") or "").strip()
        if kind in {"action", "decision"} and actor and actor != agent_id:
            raise ValueError(
                f"Workflow {raw.get('id')} {kind} step {step.get('id')} actor {actor!r} "
                f"does not match owning agent {agent_id!r}: {path}"
            )
        if kind == "workflow" and actor and workflows:
            child_id = str(step.get("workflow") or "")
            child_entry = workflows.get(child_id)
            if child_entry is not None:
                child_agent = str(child_entry[1].get("agent") or "")
                if actor != child_agent:
                    raise ValueError(
                        f"Subworkflow step {step.get('id')} actor {actor!r} does not match "
                        f"child workflow {child_id} agent {child_agent!r}: {path}"
                    )

    return {
        "workflow": str(raw.get("id") or ""),
        "agent": agent_id,
        "realizes": realizes,
        "memory_usage": memory_usage,
    }


def validate_ajd_traceability(
    workflow_catalog: dict[str, tuple[str, dict[str, Any]]],
    *,
    ajd_catalog: dict[str, tuple[str, dict[str, Any]]] | None = None,
    strict_coverage: bool = False,
) -> dict[str, Any]:
    """Validate AJD responsibility/memory contracts across all workflows."""
    agents = ajd_catalog or load_ajd_catalog()
    links: list[dict[str, str]] = []
    memory_usage: list[dict[str, str]] = []
    covered: dict[tuple[str, str], list[str]] = {}
    orphan_workflows: list[str] = []

    for workflow_id, (path, raw) in workflow_catalog.items():
        if raw.get("dsl") != "ajd-workflow/v0.2":
            continue
        result = validate_workflow_against_ajd(
            raw,
            path,
            ajd_catalog=agents,
            workflow_catalog=workflow_catalog,
        )
        if not result["realizes"]:
            orphan_workflows.append(workflow_id)
        for responsibility in result["realizes"]:
            links.append({
                "agent": result["agent"],
                "responsibility": responsibility,
                "workflow": workflow_id,
            })
            covered.setdefault((result["agent"], responsibility), []).append(workflow_id)
        for item in result["memory_usage"]:
            memory_usage.append({
                "agent": result["agent"],
                "workflow": workflow_id,
                **item,
            })

    uncovered: list[dict[str, str]] = []
    for agent_id, (_path, raw) in agents.items():
        for responsibility in sorted(_responsibility_ids(raw)):
            if (agent_id, responsibility) not in covered:
                uncovered.append({"agent": agent_id, "responsibility": responsibility})

    if strict_coverage and (orphan_workflows or uncovered):
        raise ValueError(
            f"AJD coverage errors: orphan_workflows={orphan_workflows}, "
            f"uncovered_responsibilities={uncovered}"
        )

    return {
        "responsibility_links": links,
        "memory_usage": memory_usage,
        "orphan_workflows": orphan_workflows,
        "uncovered_responsibilities": uncovered,
    }
