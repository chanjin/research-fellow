"""Architecture-level topology DSL for composing AJD workflows.

A topology is intentionally non-executable. It selects already validated
workflows and asserts the cross-workflow shared-phenomenon loops and human
boundary interactions that constitute a larger persistent job. This avoids
re-introducing Python orchestration merely to draw the whole job architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from research_fellow.application.dsl.workflow import (
    DSL_VERSION,
    _workflow_catalog,
    validate_shared_phenomenon_compatibility,
    validate_workflow_catalog,
    workflow_composition_graph,
)

TOPOLOGY_VERSION = "ajd-topology/v0.1"


@dataclass(frozen=True)
class TopologyDefinition:
    raw: dict[str, Any]

    @property
    def topology_id(self) -> str:
        return str(self.raw["id"])

    @property
    def roots(self) -> list[str]:
        return [str(item) for item in (self.raw.get("roots") or [])]


def _validate_topology_syntax(raw: dict[str, Any], relative_path: str) -> None:
    if raw.get("topology") != TOPOLOGY_VERSION:
        raise ValueError(
            f"Unsupported topology DSL version {raw.get('topology')!r}: {relative_path}"
        )
    required = ("id", "purpose", "roots", "loops", "boundary_interactions")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"AJD topology missing {missing}: {relative_path}")
    roots = raw.get("roots")
    if not isinstance(roots, list) or any(not isinstance(x, str) or not x.strip() for x in roots):
        raise ValueError(f"topology roots must be non-empty workflow ids: {relative_path}")
    for loop in raw.get("loops") or []:
        if not isinstance(loop, dict) or not str(loop.get("id") or "").strip():
            raise ValueError(f"topology loop needs id: {relative_path}")
        links = loop.get("links")
        if not isinstance(links, list) or len(links) < 2:
            raise ValueError(f"topology loop {loop.get('id')} needs at least two links: {relative_path}")
        for link in links:
            if not isinstance(link, dict):
                raise ValueError(f"topology loop links must be mappings: {relative_path}")
            for key in ("phenomenon", "from", "to"):
                if not str(link.get(key) or "").strip():
                    raise ValueError(f"topology loop link needs {key}: {relative_path}")
    for interaction in raw.get("boundary_interactions") or []:
        if not isinstance(interaction, dict):
            raise ValueError(f"boundary interaction must be a mapping: {relative_path}")
        direction = interaction.get("direction")
        if direction not in {"inbound", "outbound"}:
            raise ValueError(f"boundary interaction direction must be inbound/outbound: {relative_path}")
        for key in ("phenomenon", "workflow", "actor"):
            if not str(interaction.get(key) or "").strip():
                raise ValueError(f"boundary interaction needs {key}: {relative_path}")


def _descendants(roots: list[str], subworkflows: list[dict[str, Any]]) -> set[str]:
    children: dict[str, set[str]] = {}
    for edge in subworkflows:
        children.setdefault(str(edge["parent"]), set()).add(str(edge["child"]))
    selected = set(roots)
    frontier = list(roots)
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, set()):
            if child not in selected:
                selected.add(child)
                frontier.append(child)
    return selected


def _link_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (str(item.get("phenomenon") or ""), str(item.get("from") or ""), str(item.get("to") or ""))


def validate_topology(
    raw: dict[str, Any],
    relative_path: str = "<topology>",
) -> dict[str, Any]:
    """Validate a topology against the current workflow catalog.

    Validation is architectural rather than executable: roots and descendants
    must exist, declared loop links must be real compatible shared phenomena,
    and boundary interactions must correspond to producerless triggers or
    consumerless emits in the workflow catalog.
    """
    _validate_topology_syntax(raw, relative_path)
    validate_workflow_catalog(strict_ajd_coverage=True)
    catalog = _workflow_catalog()
    graph = workflow_composition_graph(catalog)
    selected = _descendants([str(x) for x in raw["roots"]], graph["subworkflows"])

    unknown = sorted(selected - set(catalog))
    if unknown:
        raise ValueError(f"Topology references unknown workflows {unknown}: {relative_path}")

    phenomenon_report = validate_shared_phenomenon_compatibility(catalog)
    actual_links = {
        (str(item["phenomenon"]), str(item["producer_workflow"]), str(item["consumer_workflow"]))
        for item in phenomenon_report["links"]
    }
    verified_loops: list[dict[str, Any]] = []
    for loop in raw.get("loops") or []:
        declared = [_link_key(item) for item in loop["links"]]
        missing = [item for item in declared if item not in actual_links]
        if missing:
            raise ValueError(
                f"Topology loop {loop['id']} declares non-existent phenomenon links {missing}: {relative_path}"
            )
        for _phenomenon, source, target in declared:
            if source not in selected or target not in selected:
                raise ValueError(
                    f"Topology loop {loop['id']} link {source}->{target} is outside selected roots/descendants: "
                    f"{relative_path}"
                )
        # A loop must be closed: following the declared edges leaves every node
        # with an incoming and outgoing edge within the loop.
        nodes = {source for _, source, _ in declared} | {target for _, _, target in declared}
        incoming = {node: 0 for node in nodes}
        outgoing = {node: 0 for node in nodes}
        for _phenomenon, source, target in declared:
            outgoing[source] += 1
            incoming[target] += 1
        if any(not incoming[node] or not outgoing[node] for node in nodes):
            raise ValueError(f"Topology loop {loop['id']} is not closed: {relative_path}")
        verified_loops.append({"id": str(loop["id"]), "links": declared})

    boundary_triggers = {
        (str(item["phenomenon"]), str(item["workflow"]), str(item.get("from") or ""))
        for item in phenomenon_report["boundary_triggers"]
    }
    boundary_emits = {
        (str(item["phenomenon"]), str(item["workflow"]), str(item.get("target") or ""))
        for item in phenomenon_report["boundary_emits"]
    }
    verified_boundaries: list[dict[str, str]] = []
    for item in raw.get("boundary_interactions") or []:
        key = (str(item["phenomenon"]), str(item["workflow"]), str(item["actor"]))
        available = boundary_triggers if item["direction"] == "inbound" else boundary_emits
        if key not in available:
            raise ValueError(
                f"Topology boundary interaction {item['direction']} {key} is not present in workflow catalog: "
                f"{relative_path}"
            )
        if str(item["workflow"]) not in selected:
            raise ValueError(
                f"Topology boundary workflow {item['workflow']} is outside selected roots/descendants: {relative_path}"
            )
        verified_boundaries.append({k: str(v) for k, v in item.items()})

    return {
        "topology": str(raw["id"]),
        "roots": [str(x) for x in raw["roots"]],
        "workflows": sorted(selected),
        "loops": verified_loops,
        "boundary_interactions": verified_boundaries,
    }


@lru_cache(maxsize=8)
def load_topology_definition(relative_path: str) -> TopologyDefinition:
    resource = resources.files("research_fellow").joinpath("topologies", relative_path)
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid topology definition: {relative_path}")
    validate_topology(raw, relative_path)
    return TopologyDefinition(raw)


def topology_graph(relative_path: str) -> dict[str, Any]:
    definition = load_topology_definition(relative_path)
    report = validate_topology(definition.raw, relative_path)
    workflow_graph = workflow_composition_graph()
    selected = set(report["workflows"])
    return {
        **report,
        "subworkflows": [
            item for item in workflow_graph["subworkflows"]
            if item["parent"] in selected and item["child"] in selected
        ],
        "phenomenon_links": [
            item for item in workflow_graph["phenomenon_links"]
            if item["producer_workflow"] in selected and item["consumer_workflow"] in selected
        ],
    }
