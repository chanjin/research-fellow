"""Capability gap analysis for spec-driven agent development.

The analyzer sits between Workflow DSL and the reusable Capability Contract
catalog.  It answers the development question that validation alone cannot:
which requested capabilities can be reused as-is, which existing capability is
semantically compatible under a different id, which contract has drifted, and
which capability must be implemented.

It deliberately does not choose Python implementations.  Contract matching is
performed against the implementation-independent contract catalog; Binding is a
later execution concern.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Iterable, Mapping

import yaml

from research_fellow.application.dsl.capability import (
    CapabilityContract,
    CapabilityDataSpec,
    load_capability_contracts,
    semantic_data_compatible,
)
from research_fellow.application.dsl.workflow import WorkflowDefinition, load_workflow_definition


CAPABILITY_REQUIREMENT_VERSION = "ajd-capability-requirement/v0.1"
_GAP_STATUSES = {"exact", "compatible", "mismatch", "missing"}


@dataclass(frozen=True)
class CapabilityRequirement:
    """Implementation-independent capability needed by a workflow step."""

    capability_id: str
    kind: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    data: Mapping[str, CapabilityDataSpec]
    purpose: str = ""
    step_id: str | None = None

    @property
    def all_data_names(self) -> tuple[str, ...]:
        return (*self.required_inputs, *self.optional_inputs, *self.outputs)

    @property
    def unknown_data(self) -> tuple[str, ...]:
        return tuple(name for name in self.all_data_names if name not in self.data)


@dataclass(frozen=True)
class CapabilityGap:
    step_id: str
    requested_capability: str
    status: str
    reuse_capability: str | None
    compatible_candidates: tuple[str, ...]
    closest_candidates: tuple[str, ...]
    unknown_data: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step_id,
            "requested_capability": self.requested_capability,
            "status": self.status,
            "reuse_capability": self.reuse_capability,
            "compatible_candidates": list(self.compatible_candidates),
            "closest_candidates": list(self.closest_candidates),
            "unknown_data": list(self.unknown_data),
            "reasons": list(self.reasons),
        }


def _as_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


def _data_specs(raw: Mapping[str, Any] | None) -> dict[str, CapabilityDataSpec]:
    result: dict[str, CapabilityDataSpec] = {}
    for name, item in (raw or {}).items():
        if not isinstance(item, Mapping) or not item.get("type"):
            continue
        result[str(name)] = CapabilityDataSpec(
            name=str(name),
            semantic_type=str(item["type"]),
            cardinality=str(item.get("cardinality") or "one"),
        )
    return result


def capability_requirement_from_raw(raw: Mapping[str, Any]) -> CapabilityRequirement:
    """Build a typed requirement used to evaluate reusable capabilities.

    Requirement specs are analysis-time artifacts, not Workflow DSL syntax.  A
    caller may create them from a design tool or a future agent specification
    without adding implementation details to the workflow language.
    """
    if raw.get("requirement") not in {None, CAPABILITY_REQUIREMENT_VERSION}:
        raise ValueError(f"Unsupported capability requirement version: {raw.get('requirement')!r}")
    capability_id = str(raw.get("id") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    if not capability_id or kind not in {"action", "decision"}:
        raise ValueError("Capability requirement needs id and action/decision kind")
    uses = raw.get("uses") or {}
    if not isinstance(uses, Mapping):
        raise ValueError(f"Capability requirement uses must be a mapping: {capability_id}")
    required = _as_names(uses.get("required"))
    optional = _as_names(uses.get("optional"))
    outputs = _as_names(raw.get("produces"))
    data = _data_specs(raw.get("data") if isinstance(raw.get("data"), Mapping) else None)
    unknown = set((*required, *optional, *outputs)) - set(data)
    extra = set(data) - set((*required, *optional, *outputs))
    if extra:
        raise ValueError(f"Requirement data contains undeclared names {sorted(extra)}: {capability_id}")
    # Partial semantic data is accepted for gap analysis. Unknown data prevents a
    # 'compatible' claim but still allows a precise 'missing' implementation plan.
    return CapabilityRequirement(
        capability_id=capability_id,
        kind=kind,
        required_inputs=required,
        optional_inputs=optional,
        outputs=outputs,
        data=data,
        purpose=str(raw.get("purpose") or ""),
        step_id=str(raw.get("step_id") or capability_id),
    )


def _same_shape(requirement: CapabilityRequirement, contract: CapabilityContract) -> bool:
    return (
        requirement.kind == contract.kind
        and list(requirement.required_inputs) == contract.required_inputs
        and list(requirement.optional_inputs) == contract.optional_inputs
        and list(requirement.outputs) == contract.outputs
    )


def _exact_semantics(requirement: CapabilityRequirement, contract: CapabilityContract) -> bool:
    if requirement.unknown_data:
        return False
    for name in requirement.all_data_names:
        expected = requirement.data[name]
        actual = contract.data_spec(name)
        if (
            expected.semantic_type != actual.semantic_type
            or expected.cardinality != actual.cardinality
        ):
            return False
    return True


def _compatible_semantics(requirement: CapabilityRequirement, contract: CapabilityContract) -> bool:
    """Return whether an existing contract can safely satisfy the requirement.

    Inputs are contravariant from the workflow's point of view: data supplied by
    the workflow must be accepted by the candidate capability. Outputs are
    covariant: data produced by the candidate must satisfy what the workflow
    requires.
    """
    if requirement.unknown_data:
        return False
    for name in (*requirement.required_inputs, *requirement.optional_inputs):
        supplied = requirement.data[name]
        accepted = contract.data_spec(name)
        if not semantic_data_compatible(supplied, accepted):
            return False
    for name in requirement.outputs:
        produced = contract.data_spec(name)
        expected = requirement.data[name]
        if not semantic_data_compatible(produced, expected):
            return False
    return True


def _shape_distance(requirement: CapabilityRequirement, contract: CapabilityContract) -> tuple[int, str]:
    score = 0
    if requirement.kind != contract.kind:
        score += 100
    for left, right in (
        (set(requirement.required_inputs), set(contract.required_inputs)),
        (set(requirement.optional_inputs), set(contract.optional_inputs)),
        (set(requirement.outputs), set(contract.outputs)),
    ):
        score += len(left.symmetric_difference(right))
    return score, contract.capability_id


def analyze_capability_requirement(
    requirement: CapabilityRequirement,
    *,
    contracts: Mapping[str, CapabilityContract] | None = None,
) -> CapabilityGap:
    contract_map = dict(contracts or load_capability_contracts())
    same_id = contract_map.get(requirement.capability_id)

    if same_id is not None:
        if _same_shape(requirement, same_id) and _exact_semantics(requirement, same_id):
            return CapabilityGap(
                step_id=requirement.step_id or requirement.capability_id,
                requested_capability=requirement.capability_id,
                status="exact",
                reuse_capability=same_id.capability_id,
                compatible_candidates=(),
                closest_candidates=(),
                unknown_data=requirement.unknown_data,
                reasons=("same capability id and contract are identical",),
            )
        reasons: list[str] = []
        if not _same_shape(requirement, same_id):
            reasons.append("same capability id exists but uses/produces contract differs")
        elif requirement.unknown_data:
            reasons.append("same capability id exists but semantic requirement data is incomplete")
        else:
            reasons.append("same capability id exists but semantic type/cardinality differs")
        alternatives = [
            candidate.capability_id
            for candidate in contract_map.values()
            if candidate.capability_id != same_id.capability_id
            and _same_shape(requirement, candidate)
            and _compatible_semantics(requirement, candidate)
        ]
        return CapabilityGap(
            step_id=requirement.step_id or requirement.capability_id,
            requested_capability=requirement.capability_id,
            status="mismatch",
            reuse_capability=None,
            compatible_candidates=tuple(sorted(alternatives)),
            closest_candidates=tuple(
                item[1] for item in sorted(_shape_distance(requirement, c) for c in contract_map.values())[:3]
            ),
            unknown_data=requirement.unknown_data,
            reasons=tuple(reasons),
        )

    compatible = [
        contract.capability_id
        for contract in contract_map.values()
        if _same_shape(requirement, contract) and _compatible_semantics(requirement, contract)
    ]
    if compatible:
        selected = sorted(compatible)[0]
        return CapabilityGap(
            step_id=requirement.step_id or requirement.capability_id,
            requested_capability=requirement.capability_id,
            status="compatible",
            reuse_capability=selected,
            compatible_candidates=tuple(sorted(compatible)),
            closest_candidates=(),
            unknown_data=requirement.unknown_data,
            reasons=("different capability id has a semantically compatible contract",),
        )

    reasons = ["no reusable capability contract satisfies the requested interface"]
    if requirement.unknown_data:
        reasons.append("semantic types are incomplete; compatible reuse is not inferred")
    return CapabilityGap(
        step_id=requirement.step_id or requirement.capability_id,
        requested_capability=requirement.capability_id,
        status="missing",
        reuse_capability=None,
        compatible_candidates=(),
        closest_candidates=tuple(
            item[1] for item in sorted(_shape_distance(requirement, c) for c in contract_map.values())[:3]
        ),
        unknown_data=requirement.unknown_data,
        reasons=tuple(reasons),
    )


def _catalog_data_index(
    contracts: Mapping[str, CapabilityContract],
) -> dict[str, CapabilityDataSpec]:
    """Return only datum names whose semantic meaning is unambiguous in catalog."""
    candidates: dict[str, set[tuple[str, str]]] = {}
    for contract in contracts.values():
        for name in (*contract.required_inputs, *contract.optional_inputs, *contract.outputs):
            spec = contract.data_spec(name)
            candidates.setdefault(name, set()).add((spec.semantic_type, spec.cardinality))
    return {
        name: CapabilityDataSpec(name=name, semantic_type=next(iter(values))[0], cardinality=next(iter(values))[1])
        for name, values in candidates.items()
        if len(values) == 1
    }


def workflow_capability_requirements(
    definition: WorkflowDefinition,
    *,
    contracts: Mapping[str, CapabilityContract] | None = None,
) -> list[CapabilityRequirement]:
    """Extract capability requirements from one Workflow DSL definition.

    Existing capability ids borrow semantic data from their independent Contract.
    For a new/missing capability, semantic types are inferred only from upstream
    typed outputs and catalog-wide unambiguous datum names. Unknowns remain
    unknown so the analyzer never fabricates a compatible reuse decision.
    """
    contract_map = dict(contracts or load_capability_contracts())
    global_data = _catalog_data_index(contract_map)
    produced: dict[str, CapabilityDataSpec] = {}
    requirements: list[CapabilityRequirement] = []

    for step in definition.steps:
        kind = str(step.get("kind") or "")
        if kind == "workflow":
            for name in step.get("produces") or []:
                produced.pop(str(name), None)
            continue
        if kind == "interaction":
            for name in step.get("produces") or []:
                produced.pop(str(name), None)
            continue
        capability_id = str(step.get("capability") or step.get("id") or "")
        uses = step.get("uses") or {}
        required = tuple(str(item) for item in (uses.get("required") or [])) if isinstance(uses, Mapping) else ()
        optional = tuple(str(item) for item in (uses.get("optional") or [])) if isinstance(uses, Mapping) else ()
        outputs = tuple(str(item) for item in (step.get("produces") or []))
        existing = contract_map.get(capability_id)
        data: dict[str, CapabilityDataSpec] = {}
        for name in (*required, *optional, *outputs):
            if existing is not None and name in existing.input_specs | existing.output_specs:
                data[name] = existing.data_spec(name)
            elif name in produced:
                data[name] = produced[name]
            elif name in global_data:
                data[name] = global_data[name]
        requirement = CapabilityRequirement(
            capability_id=capability_id,
            kind=kind,
            required_inputs=required,
            optional_inputs=optional,
            outputs=outputs,
            data=data,
            purpose=str(step.get("purpose") or ""),
            step_id=str(step.get("id") or capability_id),
        )
        requirements.append(requirement)
        # Only existing contracts can provide authoritative output semantics here.
        # Missing capability outputs remain untyped until a requirement/contract is designed.
        if existing is not None:
            for name in outputs:
                if name in existing.output_specs:
                    produced[name] = existing.data_spec(name)
        else:
            for name in outputs:
                produced.pop(name, None)
    return requirements


def _implementation_plan(gaps: Iterable[CapabilityGap]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for gap in gaps:
        if gap.status == "exact":
            plan.append({
                "step": gap.step_id,
                "action": "reuse",
                "capability": gap.reuse_capability,
            })
        elif gap.status == "compatible":
            plan.append({
                "step": gap.step_id,
                "action": "reuse_compatible",
                "requested_capability": gap.requested_capability,
                "capability": gap.reuse_capability,
                "change": f"set step capability to {gap.reuse_capability}",
            })
        elif gap.status == "mismatch":
            plan.append({
                "step": gap.step_id,
                "action": "reconcile_contract",
                "capability": gap.requested_capability,
                "compatible_candidates": list(gap.compatible_candidates),
            })
        else:
            plan.append({
                "step": gap.step_id,
                "action": "implement",
                "capability": gap.requested_capability,
                "needs": ["capability contract", "binding", "python implementation"],
                "unknown_data": list(gap.unknown_data),
            })
    return plan


def analyze_capability_requirements(
    requirements: Iterable[CapabilityRequirement],
    *,
    contracts: Mapping[str, CapabilityContract] | None = None,
) -> dict[str, Any]:
    contract_map = dict(contracts or load_capability_contracts())
    gaps = [analyze_capability_requirement(req, contracts=contract_map) for req in requirements]
    counts = {status: sum(gap.status == status for gap in gaps) for status in sorted(_GAP_STATUSES)}
    return {
        "required_capabilities": len(gaps),
        "summary": counts,
        "reusable_capabilities": counts["exact"] + counts["compatible"],
        "implementation_needed": counts["mismatch"] + counts["missing"],
        "capabilities": [gap.as_dict() for gap in gaps],
        "implementation_plan": _implementation_plan(gaps),
    }


def analyze_workflow_capability_gaps(
    workflow: str | WorkflowDefinition,
    *,
    contracts: Mapping[str, CapabilityContract] | None = None,
) -> dict[str, Any]:
    definition = load_workflow_definition(workflow) if isinstance(workflow, str) else workflow
    contract_map = dict(contracts or load_capability_contracts())
    requirements = workflow_capability_requirements(definition, contracts=contract_map)
    report = analyze_capability_requirements(requirements, contracts=contract_map)
    report["workflow"] = definition.workflow_id
    return report


def _iter_workflow_resources(root: Any, prefix: str = "") -> Iterable[str]:
    for child in root.iterdir():
        rel = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            yield from _iter_workflow_resources(child, rel)
        elif child.name.endswith((".yaml", ".yml")):
            yield rel


def analyze_workflow_catalog_capability_gaps() -> dict[str, Any]:
    """Analyze every packaged workflow and return an aggregate development view."""
    root = resources.files("research_fellow").joinpath("workflows")
    workflows = [analyze_workflow_capability_gaps(path) for path in _iter_workflow_resources(root)]
    summary = {status: 0 for status in sorted(_GAP_STATUSES)}
    for report in workflows:
        for status, count in report["summary"].items():
            summary[status] += int(count)
    return {
        "workflow_count": len(workflows),
        "required_capabilities": sum(item["required_capabilities"] for item in workflows),
        "summary": summary,
        "reusable_capabilities": summary["exact"] + summary["compatible"],
        "implementation_needed": summary["mismatch"] + summary["missing"],
        "workflows": workflows,
    }
