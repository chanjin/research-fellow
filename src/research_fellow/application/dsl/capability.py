"""Capability contracts and implementation bindings for spec-driven agents.

Capability contracts belong to the job specification layer. They describe the
semantic capability, its job-facing inputs/outputs, and purpose without knowing
how it is implemented. Bindings belong to the execution environment. A binding
profile maps a contract id to a concrete Python implementation or an external
provider.

This separation lets a workflow remain stable while implementations are reused,
replaced, mocked, or moved to a different runtime profile.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module, resources
from typing import Any, Callable, Iterable, Mapping

import yaml


CAPABILITY_CONTRACT_VERSION = "ajd-capability-contract/v0.2"
SEMANTIC_TYPES_VERSION = "ajd-semantic-types/v0.1"
CAPABILITY_BINDING_VERSION = "ajd-capability-binding/v0.1"
# Compatibility alias for code that treated the old combined catalog version as
# the capability schema version.
CAPABILITY_CATALOG_VERSION = CAPABILITY_CONTRACT_VERSION
DEFAULT_BINDING_PROFILE = "default"

_CAPABILITY_KINDS = {"action", "decision", "interaction"}
_IMPLEMENTATION_TYPES = {"python", "external"}
CapabilityHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SemanticType:
    type_id: str
    extends: str | None = None


@dataclass(frozen=True)
class CapabilityDataSpec:
    name: str
    semantic_type: str
    cardinality: str = "one"


@dataclass(frozen=True)
class CapabilityContract:
    raw: dict[str, Any]

    @property
    def capability_id(self) -> str:
        return str(self.raw["id"])

    @property
    def kind(self) -> str:
        return str(self.raw["kind"])

    @property
    def purpose(self) -> str:
        return str(self.raw.get("purpose") or "")

    @property
    def required_inputs(self) -> list[str]:
        return [str(item) for item in ((self.raw.get("uses") or {}).get("required") or [])]

    @property
    def optional_inputs(self) -> list[str]:
        return [str(item) for item in ((self.raw.get("uses") or {}).get("optional") or [])]

    @property
    def outputs(self) -> list[str]:
        return [str(item) for item in (self.raw.get("produces") or [])]

    def data_spec(self, name: str) -> CapabilityDataSpec:
        raw = (self.raw.get("data") or {}).get(name)
        if not isinstance(raw, Mapping):
            raise KeyError(f"Unknown capability datum {name!r}: {self.capability_id}")
        return CapabilityDataSpec(
            name=name,
            semantic_type=str(raw["type"]),
            cardinality=str(raw.get("cardinality") or "one"),
        )

    @property
    def input_specs(self) -> dict[str, CapabilityDataSpec]:
        return {name: self.data_spec(name) for name in self.required_inputs + self.optional_inputs}

    @property
    def output_specs(self) -> dict[str, CapabilityDataSpec]:
        return {name: self.data_spec(name) for name in self.outputs}


@dataclass(frozen=True)
class CapabilityBinding:
    capability_id: str
    implementation: dict[str, Any]
    profile: str = DEFAULT_BINDING_PROFILE

    @property
    def implementation_type(self) -> str:
        return str(self.implementation.get("type") or "")


@dataclass(frozen=True)
class CapabilityDefinition:
    """Compatibility/composed view of one contract plus one profile binding."""

    contract: CapabilityContract
    binding: CapabilityBinding

    @property
    def raw(self) -> dict[str, Any]:
        return {
            "id": self.capability_id,
            "kind": self.kind,
            "purpose": self.contract.purpose,
            "contract": {
                "uses": {
                    "required": self.required_inputs,
                    "optional": self.optional_inputs,
                },
                "produces": self.outputs,
                "data": {
                    name: dict(self.contract.raw["data"][name])
                    for name in self.required_inputs + self.optional_inputs + self.outputs
                },
            },
            "implementation": dict(self.binding.implementation),
        }

    @property
    def capability_id(self) -> str:
        return self.contract.capability_id

    @property
    def kind(self) -> str:
        return self.contract.kind

    @property
    def required_inputs(self) -> list[str]:
        return self.contract.required_inputs

    @property
    def optional_inputs(self) -> list[str]:
        return self.contract.optional_inputs

    @property
    def outputs(self) -> list[str]:
        return self.contract.outputs

    @property
    def implementation(self) -> dict[str, Any]:
        return dict(self.binding.implementation)

    @property
    def implementation_type(self) -> str:
        return self.binding.implementation_type



def _name_list(value: Any, *, field: str, capability_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list for capability {capability_id}")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings for capability {capability_id}")
        if item in result:
            raise ValueError(f"{field} repeats {item!r} for capability {capability_id}")
        result.append(item)
    return result


def _iter_yaml(root: Any) -> Iterable[Any]:
    for child in root.iterdir():
        if child.is_dir():
            yield from _iter_yaml(child)
        elif child.name.endswith((".yaml", ".yml")):
            yield child


def _validate_semantic_type_catalog(raw: dict[str, Any]) -> dict[str, SemanticType]:
    if not isinstance(raw, dict) or set(raw) != {"semantic_types", "types"}:
        raise ValueError("Semantic type catalog must contain semantic_types/types")
    if raw.get("semantic_types") != SEMANTIC_TYPES_VERSION:
        raise ValueError(f"Unsupported semantic type version {raw.get('semantic_types')!r}")
    entries = raw.get("types")
    if not isinstance(entries, list):
        raise ValueError("Semantic type catalog types must be a list")
    result: dict[str, SemanticType] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not set(entry).issubset({"id", "extends"}) or "id" not in entry:
            raise ValueError(f"Invalid semantic type entry: {entry!r}")
        type_id = str(entry.get("id") or "").strip()
        parent = str(entry.get("extends") or "").strip() or None
        if not type_id:
            raise ValueError("Semantic type id is required")
        if type_id in result:
            raise ValueError(f"Duplicate semantic type {type_id}")
        result[type_id] = SemanticType(type_id, parent)
    for semantic_type in result.values():
        if semantic_type.extends and semantic_type.extends not in result:
            raise ValueError(f"Semantic type {semantic_type.type_id} extends unknown type {semantic_type.extends}")
        seen: set[str] = set()
        current = semantic_type
        while current.extends:
            if current.type_id in seen:
                raise ValueError(f"Semantic type inheritance cycle at {semantic_type.type_id}")
            seen.add(current.type_id)
            current = result[current.extends]
    return result


@lru_cache(maxsize=1)
def load_semantic_types() -> dict[str, SemanticType]:
    resource = resources.files("research_fellow").joinpath("capabilities", "semantic_types.yaml")
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _validate_semantic_type_catalog(raw)


def semantic_type_compatible(actual: str, expected: str) -> bool:
    """Return True when ``actual`` is the same as or a subtype of ``expected``."""
    types = load_semantic_types()
    if actual not in types or expected not in types:
        return False
    current = types[actual]
    while True:
        if current.type_id == expected:
            return True
        if not current.extends:
            return False
        current = types[current.extends]


def semantic_data_compatible(actual: CapabilityDataSpec, expected: CapabilityDataSpec) -> bool:
    """Check semantic type and cardinality compatibility for one datum."""
    return (
        actual.cardinality == expected.cardinality
        and semantic_type_compatible(actual.semantic_type, expected.semantic_type)
    )


def _validate_contract_entry(raw: dict[str, Any]) -> CapabilityContract:
    capability_id = str(raw.get("id") or "").strip()
    if not capability_id:
        raise ValueError("Capability contract needs id")
    allowed = {"capability", "id", "kind", "purpose", "uses", "produces", "data"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown capability contract keys {unknown}: {capability_id}")
    if raw.get("capability") != CAPABILITY_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported capability contract version {raw.get('capability')!r}: {capability_id}"
        )
    kind = str(raw.get("kind") or "")
    if kind not in _CAPABILITY_KINDS:
        raise ValueError(f"Unsupported capability kind {kind!r}: {capability_id}")
    if not str(raw.get("purpose") or "").strip():
        raise ValueError(f"Capability purpose is required: {capability_id}")
    uses = raw.get("uses")
    if not isinstance(uses, dict) or set(uses) != {"required", "optional"}:
        raise ValueError(f"Capability uses must contain required/optional: {capability_id}")
    required = _name_list(uses.get("required"), field="uses.required", capability_id=capability_id)
    optional = _name_list(uses.get("optional"), field="uses.optional", capability_id=capability_id)
    outputs = _name_list(raw.get("produces"), field="produces", capability_id=capability_id)
    overlap = sorted(set(required).intersection(optional))
    if overlap:
        raise ValueError(f"Capability repeats required/optional inputs {overlap}: {capability_id}")
    data = raw.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Capability data semantic types are required: {capability_id}")
    expected_names = set(required + optional + outputs)
    if set(data) != expected_names:
        missing = sorted(expected_names - set(data))
        extra = sorted(set(data) - expected_names)
        raise ValueError(f"Capability data coverage mismatch for {capability_id}: missing={missing}, extra={extra}")
    semantic_types = load_semantic_types()
    for name, spec in data.items():
        if not isinstance(spec, dict) or not set(spec).issubset({"type", "cardinality"}) or "type" not in spec:
            raise ValueError(f"Invalid semantic data spec {name!r}: {capability_id}")
        semantic_type = str(spec.get("type") or "").strip()
        if semantic_type not in semantic_types:
            raise ValueError(f"Unknown semantic type {semantic_type!r} for {capability_id}:{name}")
        cardinality = str(spec.get("cardinality") or "one")
        if cardinality not in {"one", "many"}:
            raise ValueError(f"Unsupported cardinality {cardinality!r} for {capability_id}:{name}")
    return CapabilityContract(raw)


def _validate_binding_entry(
    raw: dict[str, Any],
    *,
    profile: str,
    contracts: Mapping[str, CapabilityContract] | None,
    validate_imports: bool,
) -> CapabilityBinding:
    capability_id = str(raw.get("id") or "").strip()
    if not capability_id:
        raise ValueError("Capability binding needs id")
    if set(raw) != {"id", "implementation"}:
        raise ValueError(f"Capability binding needs only id/implementation: {capability_id}")
    implementation = raw.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError(f"Capability implementation must be a mapping: {capability_id}")
    implementation_type = str(implementation.get("type") or "")
    if implementation_type not in _IMPLEMENTATION_TYPES:
        raise ValueError(f"Unsupported implementation type {implementation_type!r}: {capability_id}")

    contract = contracts.get(capability_id) if contracts is not None else None
    if contracts is not None and contract is None:
        raise ValueError(f"Binding references unknown capability contract: {capability_id}")

    if implementation_type == "python":
        if set(implementation) != {"type", "module", "symbol"}:
            raise ValueError(f"Python implementation needs only type/module/symbol: {capability_id}")
        module_name = str(implementation.get("module") or "").strip()
        symbol = str(implementation.get("symbol") or "").strip()
        if not module_name or not symbol:
            raise ValueError(f"Python implementation needs module and symbol: {capability_id}")
        if contract is not None and contract.kind == "interaction":
            raise ValueError(f"Interaction capability must use an external implementation: {capability_id}")
        if validate_imports:
            module = import_module(module_name)
            handler = getattr(module, symbol, None)
            if not callable(handler):
                raise ValueError(
                    f"Capability implementation is not callable: {capability_id} -> {module_name}:{symbol}"
                )
    else:
        if set(implementation) != {"type", "provider"}:
            raise ValueError(f"External implementation needs only type/provider: {capability_id}")
        if contract is not None and contract.kind != "interaction":
            raise ValueError(f"Only interaction capabilities may be external in v0.1: {capability_id}")
        if not str(implementation.get("provider") or "").strip():
            raise ValueError(f"External capability needs provider: {capability_id}")

    return CapabilityBinding(capability_id, dict(implementation), profile)


@lru_cache(maxsize=1)
def load_capability_contracts() -> dict[str, CapabilityContract]:
    root = resources.files("research_fellow").joinpath("capabilities", "contracts")
    contracts: dict[str, CapabilityContract] = {}
    for resource in _iter_yaml(root):
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Capability contract must be a mapping: {resource.name}")
        contract = _validate_contract_entry(raw)
        if resource.name != f"{contract.capability_id}.yaml":
            raise ValueError(
                f"Capability contract filename must match id: {resource.name} != {contract.capability_id}.yaml"
            )
        if contract.capability_id in contracts:
            raise ValueError(f"Duplicate capability contract id {contract.capability_id}")
        contracts[contract.capability_id] = contract
    return contracts


@lru_cache(maxsize=None)
def load_capability_bindings(profile: str = DEFAULT_BINDING_PROFILE) -> dict[str, CapabilityBinding]:
    contracts = load_capability_contracts()
    resource = resources.files("research_fellow").joinpath("capabilities", "bindings", f"{profile}.yaml")
    if not resource.is_file():
        raise ValueError(f"Unknown capability binding profile: {profile}")
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"bindings", "profile", "capabilities"}:
        raise ValueError("Capability binding profile must contain bindings/profile/capabilities")
    if raw.get("bindings") != CAPABILITY_BINDING_VERSION:
        raise ValueError(f"Unsupported capability binding version {raw.get('bindings')!r}")
    if str(raw.get("profile") or "") != profile:
        raise ValueError(f"Capability binding profile mismatch: expected {profile!r}")
    entries = raw.get("capabilities")
    if not isinstance(entries, list):
        raise ValueError("Capability bindings capabilities must be a list")

    bindings: dict[str, CapabilityBinding] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Capability binding entries must be mappings")
        binding = _validate_binding_entry(
            item,
            profile=profile,
            contracts=contracts,
            validate_imports=True,
        )
        if binding.capability_id in bindings:
            raise ValueError(f"Duplicate capability binding id {binding.capability_id}")
        bindings[binding.capability_id] = binding
    return bindings


@lru_cache(maxsize=None)
def load_capability_catalog(profile: str = DEFAULT_BINDING_PROFILE) -> dict[str, CapabilityDefinition]:
    """Return a composed compatibility view of contracts plus one binding profile."""
    contracts = load_capability_contracts()
    bindings = load_capability_bindings(profile)
    return {
        capability_id: CapabilityDefinition(contract, bindings[capability_id])
        for capability_id, contract in contracts.items()
        if capability_id in bindings
    }


def _workflow_steps_from_resources() -> list[tuple[str, str, dict[str, Any]]]:
    root = resources.files("research_fellow").joinpath("workflows")
    result: list[tuple[str, str, dict[str, Any]]] = []
    for resource in _iter_yaml(root):
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        workflow_id = str(raw["id"])
        for step in raw.get("steps") or []:
            if isinstance(step, dict):
                result.append((workflow_id, resource.name, step))
    return result


def _step_contract(step: Mapping[str, Any]) -> tuple[str, list[str], list[str], list[str]]:
    uses = step.get("uses") or {}
    if not isinstance(uses, Mapping):
        uses = {}
    return (
        str(step.get("kind") or ""),
        [str(item) for item in (uses.get("required") or [])],
        [str(item) for item in (uses.get("optional") or [])],
        [str(item) for item in (step.get("produces") or [])],
    )


def validate_workflow_semantic_compatibility(
    contracts: Mapping[str, CapabilityContract] | None = None,
) -> dict[str, Any]:
    """Validate typed data passed directly between semantic capability steps.

    Workflow inputs and subworkflow outputs remain name-based in DSL v0.2. This
    validator therefore checks only edges for which both producer and consumer
    are capability contracts. That keeps semantic typing in the capability layer
    without expanding the workflow language.
    """
    contract_map = dict(contracts or load_capability_contracts())
    workflows: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for workflow_id, path, step in _workflow_steps_from_resources():
        workflows.setdefault(workflow_id, []).append((path, step))

    checks = 0
    mismatches: list[dict[str, Any]] = []
    for workflow_id, items in workflows.items():
        produced: dict[str, tuple[str, CapabilityDataSpec]] = {}
        for path, step in items:
            if str(step.get("kind") or "") in {"workflow", "interaction"}:
                # Subworkflow outputs do not yet carry semantic types in Workflow DSL.
                for name in step.get("produces") or []:
                    produced.pop(str(name), None)
                continue
            capability_id = str(step.get("capability") or step.get("id") or "")
            contract = contract_map.get(capability_id)
            if contract is None:
                continue
            for name, expected in contract.input_specs.items():
                if name not in produced:
                    continue
                producer_id, actual = produced[name]
                checks += 1
                if not semantic_data_compatible(actual, expected):
                    mismatches.append({
                        "workflow": workflow_id,
                        "path": path,
                        "datum": name,
                        "producer": producer_id,
                        "consumer": capability_id,
                        "actual_type": actual.semantic_type,
                        "actual_cardinality": actual.cardinality,
                        "expected_type": expected.semantic_type,
                        "expected_cardinality": expected.cardinality,
                    })
            for name, spec in contract.output_specs.items():
                produced[name] = (capability_id, spec)

    if mismatches:
        raise ValueError(f"Workflow semantic type mismatches: {mismatches}")
    return {"checks": checks, "mismatches": mismatches}


def validate_capability_contracts(
    *,
    raw_contracts: Iterable[dict[str, Any]] | None = None,
    validate_workflows: bool = True,
    strict_orphans: bool = True,
) -> dict[str, Any]:
    """Validate job-facing contracts independently of any implementation binding."""
    if raw_contracts is None:
        contracts = dict(load_capability_contracts())
    else:
        contracts: dict[str, CapabilityContract] = {}
        for raw in raw_contracts:
            contract = _validate_contract_entry(raw)
            if contract.capability_id in contracts:
                raise ValueError(f"Duplicate capability contract id {contract.capability_id}")
            contracts[contract.capability_id] = contract

    references: dict[str, list[str]] = {capability_id: [] for capability_id in contracts}
    missing: list[dict[str, str]] = []
    mismatches: list[dict[str, Any]] = []
    if validate_workflows:
        for workflow_id, path, step in _workflow_steps_from_resources():
            if str(step.get("kind") or "") in {"workflow", "interaction"}:
                continue
            capability_id = str(step.get("capability") or step.get("id") or "")
            contract = contracts.get(capability_id)
            if contract is None:
                missing.append({"workflow": workflow_id, "step": str(step.get("id") or ""), "capability": capability_id})
                continue
            references[capability_id].append(workflow_id)
            actual = _step_contract(step)
            expected = (contract.kind, contract.required_inputs, contract.optional_inputs, contract.outputs)
            if actual != expected:
                mismatches.append({
                    "workflow": workflow_id,
                    "step": str(step.get("id") or ""),
                    "capability": capability_id,
                    "workflow_contract": actual,
                    "capability_contract": expected,
                })

    orphans = sorted(capability_id for capability_id, refs in references.items() if not refs)
    if missing:
        raise ValueError(f"Workflow steps reference missing capability contracts: {missing}")
    if mismatches:
        raise ValueError(f"Workflow/capability contract mismatches: {mismatches}")
    if strict_orphans and validate_workflows and orphans:
        raise ValueError(f"Capability contracts have unreferenced entries: {orphans}")

    semantic_report = validate_workflow_semantic_compatibility(contracts) if validate_workflows else {"checks": 0, "mismatches": []}
    return {
        "version": CAPABILITY_CONTRACT_VERSION,
        "semantic_types_version": SEMANTIC_TYPES_VERSION,
        "semantic_type_count": len(load_semantic_types()),
        "semantic_flow_checks": semantic_report["checks"],
        "contracts": contracts,
        "capability_count": len(contracts),
        "references": references,
        "orphan_capabilities": orphans,
    }


def validate_capability_bindings(
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
    raw: dict[str, Any] | None = None,
    contracts: Mapping[str, CapabilityContract] | None = None,
    validate_imports: bool = True,
    strict_coverage: bool = True,
) -> dict[str, Any]:
    """Validate an execution profile independently from workflow definitions."""
    contract_map = dict(contracts or load_capability_contracts())
    if raw is None:
        resource = resources.files("research_fellow").joinpath("capabilities", "bindings", f"{profile}.yaml")
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"bindings", "profile", "capabilities"}:
        raise ValueError("Capability binding profile must contain bindings/profile/capabilities")
    if raw.get("bindings") != CAPABILITY_BINDING_VERSION:
        raise ValueError(f"Unsupported capability binding version {raw.get('bindings')!r}")
    if str(raw.get("profile") or "") != profile:
        raise ValueError(f"Capability binding profile mismatch: expected {profile!r}")
    entries = raw.get("capabilities")
    if not isinstance(entries, list):
        raise ValueError("Capability bindings capabilities must be a list")

    bindings: dict[str, CapabilityBinding] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Capability binding entries must be mappings")
        binding = _validate_binding_entry(
            item,
            profile=profile,
            contracts=contract_map,
            validate_imports=validate_imports,
        )
        if binding.capability_id in bindings:
            raise ValueError(f"Duplicate capability binding id {binding.capability_id}")
        bindings[binding.capability_id] = binding

    missing_bindings = sorted(set(contract_map) - set(bindings))
    unknown_bindings = sorted(set(bindings) - set(contract_map))
    if unknown_bindings:
        raise ValueError(f"Bindings reference unknown capability contracts: {unknown_bindings}")
    if strict_coverage and missing_bindings:
        raise ValueError(f"Capability contracts missing {profile!r} bindings: {missing_bindings}")

    return {
        "version": CAPABILITY_BINDING_VERSION,
        "profile": profile,
        "bindings": bindings,
        "binding_count": len(bindings),
        "python_capabilities": sum(1 for item in bindings.values() if item.implementation_type == "python"),
        "external_capabilities": sum(1 for item in bindings.values() if item.implementation_type == "external"),
        "missing_bindings": missing_bindings,
        "unknown_bindings": unknown_bindings,
    }


def validate_capability_catalog(
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
    raw: dict[str, Any] | None = None,
    validate_imports: bool = True,
    validate_workflows: bool = True,
    strict_orphans: bool = True,
) -> dict[str, Any]:
    """Compatibility validator for the composed Contract + Binding registry.

    ``raw`` is retained only for callers migrating from R26. New code should use
    ``validate_capability_contracts`` and ``validate_capability_bindings``.
    """
    if raw is not None:
        # R27 no longer accepts the combined catalog as source-of-truth.
        raise ValueError("Combined capability catalog is retired; validate contracts and bindings separately")

    contract_report = validate_capability_contracts(
        validate_workflows=validate_workflows,
        strict_orphans=strict_orphans,
    )
    binding_report = validate_capability_bindings(
        profile=profile,
        contracts=contract_report["contracts"],
        validate_imports=validate_imports,
        strict_coverage=True,
    )
    catalog = {
        capability_id: CapabilityDefinition(contract, binding_report["bindings"][capability_id])
        for capability_id, contract in contract_report["contracts"].items()
    }
    return {
        "version": CAPABILITY_CONTRACT_VERSION,
        "binding_version": CAPABILITY_BINDING_VERSION,
        "binding_profile": profile,
        "catalog": catalog,
        "capability_count": len(catalog),
        "semantic_types_version": contract_report["semantic_types_version"],
        "semantic_type_count": contract_report["semantic_type_count"],
        "semantic_flow_checks": contract_report["semantic_flow_checks"],
        "python_capabilities": binding_report["python_capabilities"],
        "external_capabilities": binding_report["external_capabilities"],
        "references": contract_report["references"],
        "orphan_capabilities": contract_report["orphan_capabilities"],
        "missing_bindings": binding_report["missing_bindings"],
    }


def capability_contract(capability_id: str) -> CapabilityContract:
    try:
        return load_capability_contracts()[capability_id]
    except KeyError as exc:
        raise KeyError(f"Unknown capability contract: {capability_id}") from exc


def capability_binding(
    capability_id: str,
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
) -> CapabilityBinding:
    try:
        return load_capability_bindings(profile)[capability_id]
    except KeyError as exc:
        raise KeyError(f"Unknown capability binding in profile {profile!r}: {capability_id}") from exc


def capability_definition(
    capability_id: str,
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
) -> CapabilityDefinition:
    return CapabilityDefinition(
        capability_contract(capability_id),
        capability_binding(capability_id, profile=profile),
    )


def resolve_capability_handler(
    capability_id: str,
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
) -> CapabilityHandler | None:
    contract = capability_contract(capability_id)
    binding = capability_binding(capability_id, profile=profile)
    implementation = binding.implementation
    if binding.implementation_type == "external":
        return None
    if contract.kind == "interaction":
        raise ValueError(f"Interaction capability cannot resolve to Python: {capability_id}")
    module = import_module(str(implementation["module"]))
    handler = getattr(module, str(implementation["symbol"]), None)
    if not callable(handler):
        raise ValueError(
            f"Capability implementation is not callable: {capability_id} -> "
            f"{implementation.get('module')}:{implementation.get('symbol')}"
        )
    return handler


def capability_bindings_from_catalog(
    steps: Iterable[Mapping[str, Any]],
    *,
    profile: str = DEFAULT_BINDING_PROFILE,
) -> dict[str, CapabilityHandler]:
    """Resolve action/decision workflow steps through Contract + Binding."""
    bindings: dict[str, CapabilityHandler] = {}
    for step in steps:
        kind = str(step.get("kind") or "")
        if kind in {"workflow", "interaction"}:
            continue
        capability_id = str(step.get("capability") or step.get("id") or "")
        binding = capability_binding(capability_id, profile=profile)
        if binding.implementation_type == "external":
            continue
        handler = resolve_capability_handler(capability_id, profile=profile)
        if handler is not None:
            bindings[capability_id] = handler
    return bindings
