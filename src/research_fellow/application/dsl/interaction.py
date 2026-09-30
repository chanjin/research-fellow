"""Interaction contracts for human-agent communication in workflow specifications.

Interaction contracts describe *what* a user must see or provide and why. They
intentionally do not describe widgets, layouts, or renderer technology; those
belong to a later UI binding layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Mapping

import yaml

from research_fellow.application.dsl.capability import CapabilityDataSpec, load_semantic_types

INTERACTION_CONTRACT_VERSION = "ajd-interaction/v0.1"
INTERACTION_MODES = {"inform", "request_input", "select", "review", "decide", "confirm"}


@dataclass(frozen=True)
class InteractionContract:
    raw: dict[str, Any]

    @property
    def interaction_id(self) -> str:
        return str(self.raw["id"])

    @property
    def actor(self) -> str:
        return str(self.raw["actor"])

    @property
    def mode(self) -> str:
        return str(self.raw["mode"])

    @property
    def required_inputs(self) -> list[str]:
        return [str(x) for x in ((self.raw.get("uses") or {}).get("required") or [])]

    @property
    def optional_inputs(self) -> list[str]:
        return [str(x) for x in ((self.raw.get("uses") or {}).get("optional") or [])]

    @property
    def outputs(self) -> list[str]:
        return [str(x) for x in (self.raw.get("produces") or [])]

    @property
    def requires_response(self) -> bool:
        return bool((self.raw.get("completion") or {}).get("requires_response"))

    def data_spec(self, name: str) -> CapabilityDataSpec:
        item = (self.raw.get("data") or {}).get(name)
        if not isinstance(item, Mapping):
            raise KeyError(f"Unknown interaction datum {name!r}: {self.interaction_id}")
        return CapabilityDataSpec(name, str(item["type"]), str(item.get("cardinality") or "one"))


def _iter_yaml(root: Any) -> Iterable[Any]:
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_yaml(child)
        elif child.name.endswith((".yaml", ".yml")):
            yield child


def _names(value: Any, *, field: str, interaction_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list: {interaction_id}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings: {interaction_id}")
        if item in result:
            raise ValueError(f"{field} repeats {item!r}: {interaction_id}")
        result.append(item)
    return result


def _validate_contract(raw: dict[str, Any]) -> InteractionContract:
    required_keys = {"interaction", "id", "actor", "purpose", "mode", "uses", "produces", "data", "completion"}
    optional_keys = {"constraints", "authority", "boundary"}
    unknown = set(raw) - required_keys - optional_keys
    missing = required_keys - set(raw)
    if unknown or missing:
        raise ValueError(f"Interaction contract keys invalid; missing={sorted(missing)}, unknown={sorted(unknown)}")
    if raw.get("interaction") != INTERACTION_CONTRACT_VERSION:
        raise ValueError(f"Unsupported interaction contract version {raw.get('interaction')!r}")
    interaction_id = str(raw.get("id") or "").strip()
    if not interaction_id:
        raise ValueError("Interaction contract id is required")
    actor = str(raw.get("actor") or "").strip()
    if not actor:
        raise ValueError(f"Interaction actor is required: {interaction_id}")
    mode = str(raw.get("mode") or "")
    if mode not in INTERACTION_MODES:
        raise ValueError(f"Unsupported interaction mode {mode!r}: {interaction_id}")
    uses = raw.get("uses")
    if not isinstance(uses, dict) or set(uses) != {"required", "optional"}:
        raise ValueError(f"Interaction uses must contain required/optional: {interaction_id}")
    required = _names(uses.get("required"), field="uses.required", interaction_id=interaction_id)
    optional = _names(uses.get("optional"), field="uses.optional", interaction_id=interaction_id)
    outputs = _names(raw.get("produces"), field="produces", interaction_id=interaction_id)
    all_names = required + optional + outputs
    if len(set(all_names)) != len(all_names):
        raise ValueError(f"Interaction input/output names overlap: {interaction_id}")
    data = raw.get("data")
    if not isinstance(data, dict) or set(data) != set(all_names):
        raise ValueError(f"Interaction data must type every uses/produces datum: {interaction_id}")
    semantic_types = load_semantic_types()
    for name, spec in data.items():
        if not isinstance(spec, dict) or not set(spec).issubset({"type", "cardinality"}) or "type" not in spec:
            raise ValueError(f"Invalid interaction data spec {name}: {interaction_id}")
        if str(spec["type"]) not in semantic_types:
            raise ValueError(f"Unknown semantic type {spec['type']!r} for {name}: {interaction_id}")
        if str(spec.get("cardinality") or "one") not in {"one", "many"}:
            raise ValueError(f"Invalid cardinality for {name}: {interaction_id}")
    completion = raw.get("completion")
    if not isinstance(completion, dict) or set(completion) != {"requires_response"} or not isinstance(completion.get("requires_response"), bool):
        raise ValueError(f"Interaction completion needs boolean requires_response: {interaction_id}")
    if mode == "inform" and completion["requires_response"]:
        raise ValueError(f"inform interaction cannot require a response: {interaction_id}")
    if mode != "inform" and not completion["requires_response"]:
        raise ValueError(f"{mode} interaction must require a response in v0.1: {interaction_id}")
    if mode == "decide":
        authority = raw.get("authority")
        if not isinstance(authority, dict) or str(authority.get("holder") or "") != actor:
            raise ValueError(f"decide interaction authority.holder must equal actor: {interaction_id}")
    boundary = raw.get("boundary")
    if boundary is not None:
        if not isinstance(boundary, dict):
            raise ValueError(f"Interaction boundary must be a mapping: {interaction_id}")
        allowed_boundary = {"phenomenon", "recipient", "status"}
        if not set(boundary).issubset(allowed_boundary) or not str(boundary.get("phenomenon") or "").strip():
            raise ValueError(f"Interaction boundary needs phenomenon and known fields only: {interaction_id}")
        recipient = str(boundary.get("recipient") or "").strip()
        if recipient and recipient != actor:
            raise ValueError(f"Interaction boundary recipient must equal actor when set: {interaction_id}")
    return InteractionContract(dict(raw))


@lru_cache(maxsize=1)
def load_interaction_contracts() -> dict[str, InteractionContract]:
    root = resources.files("research_fellow").joinpath("interactions", "contracts")
    result: dict[str, InteractionContract] = {}
    for resource in _iter_yaml(root):
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Interaction contract must be a mapping: {resource.name}")
        contract = _validate_contract(raw)
        if resource.name != f"{contract.interaction_id}.yaml":
            raise ValueError(f"Interaction contract filename must match id: {resource.name}")
        if contract.interaction_id in result:
            raise ValueError(f"Duplicate interaction contract {contract.interaction_id}")
        result[contract.interaction_id] = contract
    return result


def interaction_contract(interaction_id: str) -> InteractionContract:
    try:
        return load_interaction_contracts()[interaction_id]
    except KeyError as exc:
        raise ValueError(f"Unknown interaction contract: {interaction_id}") from exc


def validate_interaction_contracts(*, workflow_catalog: Mapping[str, tuple[str, dict[str, Any]]] | None = None, strict_orphans: bool = True) -> dict[str, Any]:
    contracts = load_interaction_contracts()
    references: dict[str, list[str]] = {key: [] for key in contracts}
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, Any]] = []

    if workflow_catalog is None:
        # Local import avoids a module import cycle.
        from research_fellow.application.dsl.workflow import workflow_catalog_snapshot
        workflow_catalog = workflow_catalog_snapshot()

    for workflow_id, (path, raw) in workflow_catalog.items():
        for step in raw.get("steps") or []:
            if str(step.get("kind") or "") != "interaction":
                continue
            step_id = str(step.get("id") or "")
            contract_id = str(step.get("interaction") or "")
            contract = contracts.get(contract_id)
            if contract is None:
                missing.append({"workflow": workflow_id, "step": step_id, "interaction": contract_id})
                continue
            references[contract_id].append(workflow_id)
            uses = step.get("uses") or {}
            actual = (
                str(step.get("actor") or ""),
                [str(x) for x in (uses.get("required") or [])],
                [str(x) for x in (uses.get("optional") or [])],
                [str(x) for x in (step.get("produces") or [])],
            )
            expected = (contract.actor, contract.required_inputs, contract.optional_inputs, contract.outputs)
            if actual != expected:
                mismatches.append({"workflow": workflow_id, "path": path, "step": step_id, "interaction": contract_id, "workflow_contract": actual, "interaction_contract": expected})

    boundary_interactions = sorted(
        key for key, contract in contracts.items() if isinstance(contract.raw.get("boundary"), dict)
    )
    orphans = sorted(
        key for key, refs in references.items()
        if not refs and key not in boundary_interactions
    )
    if missing:
        raise ValueError(f"Workflow steps reference missing interaction contracts: {missing}")
    if mismatches:
        raise ValueError(f"Workflow/interaction contract mismatches: {mismatches}")
    if strict_orphans and orphans:
        raise ValueError(f"Interaction contracts have unreferenced entries: {orphans}")
    return {
        "version": INTERACTION_CONTRACT_VERSION,
        "interaction_count": len(contracts),
        "contracts": contracts,
        "references": references,
        "orphan_interactions": orphans,
        "boundary_interactions": boundary_interactions,
        "response_required": sorted(c.interaction_id for c in contracts.values() if c.requires_response),
    }
