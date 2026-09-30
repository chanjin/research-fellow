"""Executable support for AJD Workflow DSL v0.2.

The DSL owns job-level orchestration: semantic capability ordering, conditions,
foreach iteration, shared-phenomenon metadata, and completion/invariant intent.
Python bindings implement capabilities and runtime concerns such as persistence,
LLM calls, retry, parsing, and execution observation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from research_fellow.application.dsl.ajd import validate_ajd_traceability, validate_workflow_against_ajd
from research_fellow.application.dsl.capability import (
    capability_bindings_from_catalog,
    validate_capability_catalog,
)
from research_fellow.application.dsl.interaction import interaction_contract, validate_interaction_contracts
from research_fellow.application.dsl.autonomy import (
    AUTONOMY_AUTO, AUTONOMY_AUTO_NOTIFY, evaluate_interaction_autonomy,
)


WorkflowHandler = Callable[[dict[str, Any]], None]
DSL_VERSION = "ajd-workflow/v0.2"
_STEP_KINDS = {"action", "decision", "interaction", "workflow"}
_FORBIDDEN_V02_KEYS = {
    "type",
    "handler",
    "performer",
    "reads",
    "writes",
    "semantics",
    "description",
    "optional_uses",
}
_TOP_LEVEL_FORBIDDEN_V02_KEYS = {
    "version",
    "responsibility",
    "evaluation",
    "optional_outputs",
}



@dataclass(frozen=True)
class WorkflowDefinition:
    raw: dict[str, Any]

    @property
    def workflow_id(self) -> str:
        return str(self.raw["id"])

    @property
    def dsl_version(self) -> str:
        return str(self.raw.get("dsl") or "")

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self.raw.get("steps") or [])

    @property
    def input_names(self) -> list[str]:
        return [str(item) for item in (self.raw.get("inputs") or [])]

    @property
    def output_names(self) -> list[str]:
        outputs = self.raw.get("outputs") or {}
        if isinstance(outputs, list):
            return [str(item) for item in outputs]
        return [
            *[str(item) for item in (outputs.get("required") or [])],
            *[str(item) for item in (outputs.get("optional") or [])],
        ]

    @property
    def required_output_names(self) -> list[str]:
        outputs = self.raw.get("outputs") or {}
        if isinstance(outputs, list):
            return [str(item) for item in outputs]
        return [str(item) for item in (outputs.get("required") or [])]

    @property
    def optional_output_names(self) -> list[str]:
        outputs = self.raw.get("outputs") or {}
        if isinstance(outputs, list):
            return []
        return [str(item) for item in (outputs.get("optional") or [])]

    @property
    def produced_names(self) -> list[str]:
        names: list[str] = []
        for step in self.steps:
            for item in step.get("produces") or []:
                name = str(item)
                if name not in names:
                    names.append(name)
        return names



def _validate_v02_phenomena(raw: dict[str, Any], relative_path: str) -> None:
    trigger = raw.get("trigger")
    if not isinstance(trigger, dict) or not str(trigger.get("phenomenon") or "").strip():
        raise ValueError(f"trigger needs phenomenon: {relative_path}")
    unknown_trigger = sorted(set(trigger) - {"phenomenon", "from", "condition"})
    if unknown_trigger:
        raise ValueError(f"Unknown trigger keys {unknown_trigger}: {relative_path}")
    if "from" in trigger and not str(trigger.get("from") or "").strip():
        raise ValueError(f"trigger from must be non-empty when present: {relative_path}")
    if "condition" in trigger and not str(trigger.get("condition") or "").strip():
        raise ValueError(f"trigger condition must be non-empty when present: {relative_path}")

    for step in raw.get("steps") or []:
        step_id = str(step.get("id") or "")
        emits = step.get("emits") or []
        for event in emits:
            if not isinstance(event, dict):
                raise ValueError(f"emits entries must be mappings in {step_id}: {relative_path}")
            unknown_event = sorted(set(event) - {"phenomenon", "to"})
            if unknown_event:
                raise ValueError(f"Unknown emits keys {unknown_event} in {step_id}: {relative_path}")
            if not str(event.get("phenomenon") or "").strip():
                raise ValueError(f"emits entry needs phenomenon in {step_id}: {relative_path}")
            targets = event.get("to")
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"emits entry needs non-empty to list in {step_id}: {relative_path}")
            if any(not isinstance(item, str) or not item.strip() for item in targets):
                raise ValueError(f"emits to entries must be non-empty strings in {step_id}: {relative_path}")


def _iter_workflow_resources(root: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    for child in root.iterdir():
        rel = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            yield from _iter_workflow_resources(child, rel)
        elif child.name.endswith((".yaml", ".yml")):
            yield rel, child


@lru_cache(maxsize=1)
def _workflow_catalog() -> dict[str, tuple[str, dict[str, Any]]]:
    root = resources.files("research_fellow").joinpath("workflows")
    catalog: dict[str, tuple[str, dict[str, Any]]] = {}
    for relative_path, resource in _iter_workflow_resources(root):
        raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        workflow_id = str(raw["id"])
        if workflow_id in catalog:
            other_path = catalog[workflow_id][0]
            raise ValueError(f"Duplicate workflow id {workflow_id}: {other_path}, {relative_path}")
        catalog[workflow_id] = (relative_path, raw)
    return catalog


def workflow_catalog_snapshot() -> dict[str, tuple[str, dict[str, Any]]]:
    """Return a shallow copy of the loaded workflow catalog for validators/tooling."""
    return dict(_workflow_catalog())


def _step_uses(step: dict[str, Any]) -> tuple[list[str], list[str]]:
    spec = step.get("uses") or {}
    if not isinstance(spec, dict):
        return [], []
    return (
        [str(item) for item in (spec.get("required") or [])],
        [str(item) for item in (spec.get("optional") or [])],
    )


def _raw_output_names(raw: dict[str, Any]) -> list[str]:
    spec = raw.get("outputs") or {}
    if not isinstance(spec, dict):
        return []
    return [
        *[str(item) for item in (spec.get("required") or [])],
        *[str(item) for item in (spec.get("optional") or [])],
    ]


def _validate_v02_subworkflow_contracts(raw: dict[str, Any], relative_path: str) -> None:
    catalog = _workflow_catalog()
    for step in raw.get("steps") or []:
        if str(step.get("kind") or "") != "workflow":
            continue
        step_id = str(step["id"])
        child_id = str(step.get("workflow") or "")
        child_entry = catalog.get(child_id)
        if child_entry is None:
            raise ValueError(f"Unknown subworkflow {child_id!r} in {step_id}: {relative_path}")
        child_path, child_raw = child_entry
        child_inputs = [str(item) for item in (child_raw.get("inputs") or [])]
        child_outputs = _raw_output_names(child_raw)
        parent_required, parent_optional = _step_uses(step)
        parent_uses = [*parent_required, *parent_optional]
        parent_produces = [str(item) for item in (step.get("produces") or [])]

        foreach = step.get("foreach")
        if foreach:
            if len(child_inputs) != 1:
                raise ValueError(
                    f"Foreach subworkflow {child_id} must declare exactly one input; "
                    f"found {child_inputs}: {relative_path}"
                )
        else:
            missing_inputs = [name for name in child_inputs if name not in parent_uses]
            if missing_inputs:
                raise ValueError(
                    f"Subworkflow {child_id} inputs {missing_inputs} are not supplied by step "
                    f"{step_id} uses: {relative_path}"
                )

        if child_outputs and not parent_produces:
            raise ValueError(
                f"Subworkflow step {step_id} must declare produces for child outputs "
                f"{child_outputs}: {relative_path}"
            )
        if not child_outputs and parent_produces:
            raise ValueError(
                f"Subworkflow step {step_id} declares produces {parent_produces}, but child "
                f"{child_id} has no outputs: {child_path}"
            )



def validate_shared_phenomenon_compatibility(
    catalog: dict[str, tuple[str, dict[str, Any]]] | None = None,
    *,
    strict_orphans: bool = False,
) -> dict[str, Any]:
    """Validate cross-workflow shared-phenomenon compatibility.

    A matching workflow-to-workflow interaction requires four things to agree:
    phenomenon name, producer agent, consumer agent, and trigger.from/to routing.
    Endpoints with no workflow counterpart are treated as system-boundary
    interactions by default; callers may make them errors with ``strict_orphans``.
    """
    entries = catalog or _workflow_catalog()
    triggers: list[dict[str, Any]] = []
    emits: list[dict[str, Any]] = []

    for workflow_id, (path, raw) in entries.items():
        if raw.get("dsl") != DSL_VERSION:
            continue
        agent = str(raw.get("agent") or "")
        trigger = raw.get("trigger") or {}
        phenomenon = str(trigger.get("phenomenon") or "").strip()
        if phenomenon:
            triggers.append({
                "workflow": workflow_id,
                "path": path,
                "agent": agent,
                "phenomenon": phenomenon,
                "from": str(trigger.get("from") or "").strip() or None,
            })
        for step in raw.get("steps") or []:
            for event in step.get("emits") or []:
                emits.append({
                    "workflow": workflow_id,
                    "path": path,
                    "step": str(step.get("id") or ""),
                    "agent": agent,
                    "phenomenon": str(event.get("phenomenon") or "").strip(),
                    "to": [str(item) for item in (event.get("to") or [])],
                })

    links: list[dict[str, Any]] = []
    boundary_triggers: list[dict[str, Any]] = []
    boundary_emits: list[dict[str, Any]] = []
    errors: list[str] = []

    seen_links: set[tuple[str, str, str, str, str]] = set()

    for event in emits:
        for target in event["to"]:
            candidates = [
                t for t in triggers
                if t["phenomenon"] == event["phenomenon"] and t["agent"] == target
            ]
            if not candidates:
                boundary_emits.append({**event, "target": target})
                continue
            compatible = [
                t for t in candidates
                if t["from"] is None or t["from"] == event["agent"]
            ]
            if not compatible:
                expected = sorted({t["from"] or "*" for t in candidates})
                errors.append(
                    f"Emit {event['workflow']}.{event['step']} sends {event['phenomenon']!r} "
                    f"from {event['agent']} to {target}, but matching trigger(s) expect from {expected}"
                )
                continue
            for trigger in compatible:
                key = (event["workflow"], event["step"], trigger["workflow"], event["phenomenon"], target)
                if key in seen_links:
                    continue
                seen_links.add(key)
                links.append({
                    "phenomenon": event["phenomenon"],
                    "producer_workflow": event["workflow"],
                    "producer_step": event["step"],
                    "producer_agent": event["agent"],
                    "consumer_workflow": trigger["workflow"],
                    "consumer_agent": trigger["agent"],
                })

    for trigger in triggers:
        source = trigger["from"]
        if not source:
            continue
        candidates = [
            e for e in emits
            if e["phenomenon"] == trigger["phenomenon"] and e["agent"] == source
        ]
        if not candidates:
            boundary_triggers.append(trigger)
            continue
        if not any(trigger["agent"] in e["to"] for e in candidates):
            destinations = sorted({dest for e in candidates for dest in e["to"]})
            errors.append(
                f"Trigger {trigger['workflow']} expects {trigger['phenomenon']!r} from {source} "
                f"to {trigger['agent']}, but producer emit(s) target {destinations}"
            )

    if strict_orphans:
        for trigger in boundary_triggers:
            errors.append(
                f"Producerless trigger {trigger['workflow']} expects {trigger['phenomenon']!r} "
                f"from {trigger['from']}"
            )
        for event in boundary_emits:
            errors.append(
                f"Consumerless emit {event['workflow']}.{event['step']} sends "
                f"{event['phenomenon']!r} to {event['target']}"
            )

    if errors:
        raise ValueError("Shared phenomenon compatibility errors: " + "; ".join(errors))

    return {
        "links": links,
        "boundary_triggers": boundary_triggers,
        "boundary_emits": boundary_emits,
    }


def workflow_composition_graph(
    catalog: dict[str, tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return the DSL-declared workflow hierarchy and cross-workflow phenomena.

    This is a specification view, not an execution registry: parent/child edges
    come only from ``kind: workflow`` steps, while interaction edges come from
    trigger/emits compatibility. It lets tooling render the job architecture
    without reverse-engineering Python calls.
    """
    entries = catalog or _workflow_catalog()
    nodes: list[dict[str, Any]] = []
    subworkflows: list[dict[str, Any]] = []
    for workflow_id, (path, raw) in entries.items():
        if raw.get("dsl") != DSL_VERSION:
            continue
        nodes.append({
            "workflow": workflow_id,
            "agent": str(raw.get("agent") or ""),
            "path": path,
            "realizes": [str(item) for item in (raw.get("realizes") or [])],
        })
        for step in raw.get("steps") or []:
            if str(step.get("kind") or "") != "workflow":
                continue
            subworkflows.append({
                "parent": workflow_id,
                "step": str(step.get("id") or ""),
                "child": str(step.get("workflow") or ""),
                "actor": str(step.get("actor") or raw.get("agent") or ""),
                "when": step.get("when"),
                "foreach": step.get("foreach"),
            })
    phenomena = validate_shared_phenomenon_compatibility(entries)
    return {
        "workflows": nodes,
        "subworkflows": subworkflows,
        "phenomenon_links": phenomena["links"],
        "boundary_triggers": phenomena["boundary_triggers"],
        "boundary_emits": phenomena["boundary_emits"],
    }


def validate_workflow_catalog(
    *,
    strict_orphans: bool = False,
    strict_ajd_coverage: bool = False,
) -> dict[str, Any]:
    """Validate workflow syntax, composition, phenomena, and AJD traceability."""
    catalog = _workflow_catalog()
    for path, raw in catalog.values():
        declared = raw.get("dsl")
        if declared != DSL_VERSION:
            raise ValueError(f"Unsupported workflow DSL version {declared!r}: {path}")
        _validate_v02(raw, path)
        _validate_v02_subworkflow_contracts(raw, path)
        validate_workflow_against_ajd(raw, path, workflow_catalog=catalog)
    phenomena = validate_shared_phenomenon_compatibility(
        catalog, strict_orphans=strict_orphans
    )
    traceability = validate_ajd_traceability(
        catalog, strict_coverage=strict_ajd_coverage
    )
    capabilities = validate_capability_catalog(
        validate_imports=True, validate_workflows=True, strict_orphans=True
    )
    interactions = validate_interaction_contracts(workflow_catalog=catalog, strict_orphans=True)
    return {**phenomena, "ajd_traceability": traceability, "capability_catalog": capabilities, "interaction_contracts": interactions}

def validate_workflow_bindings(
    definition: WorkflowDefinition,
    handlers: dict[str, WorkflowHandler],
) -> None:
    """Validate executable bindings before execution.

    Interaction steps are resolved from Interaction Contracts by the runtime and
    therefore deliberately do not require Python handlers. Action/decision
    capabilities and subworkflow adapters still require executable bindings.
    """
    missing: list[str] = []
    for step in definition.steps:
        if str(step.get("kind") or "") == "interaction":
            continue
        name = _binding_name(step)
        if name not in handlers and name not in missing:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Workflow {definition.workflow_id} has unbound capabilities/subworkflows: {missing}"
        )

def _validate_v02(raw: dict[str, Any], relative_path: str) -> None:
    required = ("id", "agent", "purpose", "trigger", "steps", "completion")
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"AJD Workflow DSL v0.2 missing {missing}: {relative_path}")

    forbidden_top = sorted(_TOP_LEVEL_FORBIDDEN_V02_KEYS.intersection(raw))
    if forbidden_top:
        raise ValueError(f"Legacy/redundant workflow keys {forbidden_top} are not allowed in v0.2: {relative_path}")

    revision = raw.get("revision")
    if revision is not None and (not isinstance(revision, int) or revision < 1):
        raise ValueError(f"revision must be a positive integer: {relative_path}")

    realizes = raw.get("realizes")
    if realizes is not None and (
        not isinstance(realizes, list)
        or any(not isinstance(item, str) or not item.strip() for item in realizes)
    ):
        raise ValueError(f"realizes must be a list of non-empty responsibility ids: {relative_path}")

    steps = raw.get("steps") or []
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Workflow has no steps: {relative_path}")

    workflow_agent = str(raw.get("agent") or "")
    seen: set[str] = set()
    for step in steps:
        if not isinstance(step, dict) or not step.get("id"):
            raise ValueError(f"Every workflow step needs id: {relative_path}")
        step_id = str(step["id"])
        if step_id in seen:
            raise ValueError(f"Duplicate workflow step id {step_id}: {relative_path}")
        seen.add(step_id)

        forbidden = sorted(_FORBIDDEN_V02_KEYS.intersection(step))
        if forbidden:
            raise ValueError(f"Legacy/redundant step keys {forbidden} in {step_id}: {relative_path}")

        kind = str(step.get("kind") or "")
        if kind not in _STEP_KINDS:
            raise ValueError(f"Invalid step kind {kind!r} in {step_id}: {relative_path}")
        if kind == "workflow":
            if not step.get("workflow"):
                raise ValueError(f"Workflow step {step_id} needs workflow: {relative_path}")
        elif kind == "interaction":
            if not str(step.get("interaction") or "").strip():
                raise ValueError(f"Interaction step {step_id} needs interaction contract id: {relative_path}")
            if not str(step.get("actor") or "").strip():
                raise ValueError(f"Interaction step {step_id} needs actor: {relative_path}")
        else:
            capability = step.get("capability")
            if capability is not None and (not isinstance(capability, str) or not capability.strip()):
                raise ValueError(f"Step {step_id} capability must be non-empty text: {relative_path}")
        if kind == "decision" and not step.get("authority"):
            raise ValueError(f"Decision step {step_id} needs authority: {relative_path}")

        emits = step.get("emits")
        if emits is not None and not isinstance(emits, list):
            raise ValueError(f"emits must be a list in {step_id}: {relative_path}")

        step_purpose = step.get("purpose")
        if step_purpose is not None and (not isinstance(step_purpose, str) or not step_purpose.strip()):
            raise ValueError(f"step purpose must be non-empty text in {step_id}: {relative_path}")

    _validate_v02_dataflow(raw, relative_path)
    _validate_v02_phenomena(raw, relative_path)


def _as_name_list(value: Any, *, field: str, step_id: str, relative_path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list in {step_id}: {relative_path}")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings in {step_id}: {relative_path}")
        names.append(item)
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicates in {step_id}: {relative_path}")
    return names


def _parse_uses(step: dict[str, Any], *, step_id: str, relative_path: str) -> tuple[list[str], list[str]]:
    spec = step.get("uses") or {}
    if not isinstance(spec, dict):
        raise ValueError(f"uses must be a mapping with required/optional in {step_id}: {relative_path}")
    unknown = sorted(set(spec) - {"required", "optional"})
    if unknown:
        raise ValueError(f"Unknown uses keys {unknown} in {step_id}: {relative_path}")
    required = _as_name_list(spec.get("required"), field="uses.required", step_id=step_id, relative_path=relative_path)
    optional = _as_name_list(spec.get("optional"), field="uses.optional", step_id=step_id, relative_path=relative_path)
    overlap = sorted(set(required).intersection(optional))
    if overlap:
        raise ValueError(f"Step {step_id} repeats values in required/optional uses {overlap}: {relative_path}")
    return required, optional


def _parse_outputs(raw: dict[str, Any], relative_path: str) -> tuple[list[str], list[str]]:
    spec = raw.get("outputs") or {}
    if not isinstance(spec, dict):
        raise ValueError(f"outputs must be a mapping with required/optional: {relative_path}")
    unknown = sorted(set(spec) - {"required", "optional"})
    if unknown:
        raise ValueError(f"Unknown outputs keys {unknown}: {relative_path}")
    required = _as_name_list(spec.get("required"), field="outputs.required", step_id="workflow", relative_path=relative_path)
    optional = _as_name_list(spec.get("optional"), field="outputs.optional", step_id="workflow", relative_path=relative_path)
    overlap = sorted(set(required).intersection(optional))
    if overlap:
        raise ValueError(f"Workflow repeats values in required/optional outputs {overlap}: {relative_path}")
    return required, optional


def _validate_v02_outcomes(raw: dict[str, Any], relative_path: str) -> dict[str, list[str]]:
    value = raw.get("outcomes") or {}
    if not isinstance(value, dict):
        raise ValueError(f"outcomes must be a mapping: {relative_path}")
    groups: dict[str, list[str]] = {}
    seen_conditions: set[str] = set()
    for name, spec in value.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            raise ValueError(f"outcome entries must be named mappings: {relative_path}")
        if set(spec) != {"one_of"}:
            raise ValueError(f"outcome {name} supports only one_of in v0.2: {relative_path}")
        alternatives = spec.get("one_of")
        if not isinstance(alternatives, list) or len(alternatives) < 2:
            raise ValueError(f"outcome {name} needs one_of with at least two conditions: {relative_path}")
        cleaned: list[str] = []
        for item in alternatives:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"outcome {name} one_of entries must be non-empty strings: {relative_path}")
            if item in cleaned:
                raise ValueError(f"outcome {name} repeats condition {item!r}: {relative_path}")
            if item in seen_conditions:
                raise ValueError(f"outcome condition {item!r} belongs to multiple groups: {relative_path}")
            cleaned.append(item)
            seen_conditions.add(item)
        groups[name] = cleaned
    return groups


def _guard_implies(consumer_guard: frozenset[str], producer_guard: frozenset[str]) -> bool:
    return producer_guard.issubset(consumer_guard)


def _is_guaranteed_across_outcomes(
    producer_guards: set[frozenset[str]],
    outcome_groups: dict[str, list[str]],
) -> bool:
    if frozenset() in producer_guards:
        return True
    for alternatives in outcome_groups.values():
        if all(frozenset({condition}) in producer_guards for condition in alternatives):
            return True
    return False


def _validate_v02_dataflow(raw: dict[str, Any], relative_path: str) -> None:
    """Validate v0.2 data contracts and branch-sensitive path safety."""
    inputs = raw.get("inputs") or []
    if not isinstance(inputs, list):
        raise ValueError(f"inputs must be a list: {relative_path}")

    outcome_groups = _validate_v02_outcomes(raw, relative_path)
    producers: dict[str, set[frozenset[str]]] = {}
    available_names: set[str] = set()
    for item in inputs:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"inputs entries must be non-empty strings: {relative_path}")
        if item in available_names:
            raise ValueError(f"inputs contains duplicate {item!r}: {relative_path}")
        available_names.add(item)
        producers.setdefault(item, set()).add(frozenset())

    for step in raw.get("steps") or []:
        step_id = str(step["id"])
        when = step.get("when")
        if when is not None and not isinstance(when, (bool, str)):
            raise ValueError(f"Unsupported when condition in {step_id}: {relative_path}")
        if isinstance(when, str) and when not in available_names:
            raise ValueError(f"Step {step_id} when references unavailable data {when!r}: {relative_path}")

        if when is False:
            consumer_guard = frozenset({"__never__"})
        elif isinstance(when, str):
            consumer_guard = frozenset({when})
        else:
            consumer_guard = frozenset()

        required_uses, optional_uses = _parse_uses(step, step_id=step_id, relative_path=relative_path)
        for name in [*required_uses, *optional_uses]:
            if name not in available_names:
                raise ValueError(f"Step {step_id} uses unavailable data {[name]}: {relative_path}")

        for name in required_uses:
            guards = producers.get(name, set())
            guarded_by_self = isinstance(when, str) and when == name
            if not guarded_by_self and not any(_guard_implies(consumer_guard, guard) for guard in guards):
                raise ValueError(
                    f"Step {step_id} unsafely uses conditional data {name!r}; "
                    f"add a compatible when guard or move it to uses.optional: {relative_path}"
                )

        foreach = step.get("foreach")
        if foreach is not None:
            if not isinstance(foreach, str) or not foreach.strip():
                raise ValueError(f"foreach must reference a data name in {step_id}: {relative_path}")
            if foreach not in available_names:
                raise ValueError(f"Step {step_id} foreach references unavailable data {foreach!r}: {relative_path}")
            guards = producers.get(foreach, set())
            guarded_by_self = isinstance(when, str) and when == foreach
            if not guarded_by_self and not any(_guard_implies(consumer_guard, guard) for guard in guards):
                raise ValueError(f"Step {step_id} foreach unsafely references conditional data {foreach!r}: {relative_path}")

        produces = _as_name_list(step.get("produces"), field="produces", step_id=step_id, relative_path=relative_path)
        producer_guard = (
            frozenset({"__never__"}) if when is False
            else frozenset({when}) if isinstance(when, str)
            else frozenset()
        )
        for name in produces:
            available_names.add(name)
            producers.setdefault(name, set()).add(producer_guard)

    for group_name, alternatives in outcome_groups.items():
        for condition in alternatives:
            if condition not in producers:
                raise ValueError(f"Outcome {group_name} references unknown condition {condition!r}: {relative_path}")
            if frozenset() not in producers[condition]:
                raise ValueError(f"Outcome condition {condition!r} must be produced unconditionally: {relative_path}")

    required_outputs, optional_outputs = _parse_outputs(raw, relative_path)
    output_names = [*required_outputs, *optional_outputs]
    invalid_outputs = [name for name in output_names if name not in available_names]
    if invalid_outputs:
        raise ValueError(f"Workflow outputs are never declared as input/produced data {invalid_outputs}: {relative_path}")

    unsafe_outputs = [
        name for name in required_outputs
        if not _is_guaranteed_across_outcomes(producers.get(name, set()), outcome_groups)
    ]
    if unsafe_outputs:
        raise ValueError(
            f"Workflow required outputs are not guaranteed on every normal outcome path {unsafe_outputs}; "
            f"move them to outputs.optional or complete all outcome branches: {relative_path}"
        )

    completion = raw.get("completion")
    if not isinstance(completion, dict) or set(completion) != {"criteria"}:
        raise ValueError(f"completion must contain only criteria in v0.2: {relative_path}")
    criteria = completion.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError(f"completion.criteria must be a non-empty list: {relative_path}")
    if any(not isinstance(item, str) or not item.strip() for item in criteria):
        raise ValueError(f"completion.criteria entries must be non-empty strings: {relative_path}")

    invariants = raw.get("invariants") or []
    if not isinstance(invariants, list) or any(not isinstance(item, str) or not item.strip() for item in invariants):
        raise ValueError(f"invariants must be a list of non-empty strings: {relative_path}")


def load_workflow_definition_by_id(workflow_id: str) -> WorkflowDefinition:
    entry = _workflow_catalog().get(str(workflow_id))
    if entry is None:
        raise KeyError(f"Unknown workflow id: {workflow_id}")
    relative_path, _raw = entry
    return load_workflow_definition(relative_path)


def load_workflow_definition(relative_path: str) -> WorkflowDefinition:
    root = resources.files("research_fellow")
    text = root.joinpath("workflows", relative_path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError(f"Invalid workflow definition: {relative_path}")

    declared = raw.get("dsl")
    if declared:
        if declared != DSL_VERSION:
            raise ValueError(f"Unsupported workflow DSL version {declared!r}: {relative_path}")
        _validate_v02(raw, relative_path)
        _validate_v02_subworkflow_contracts(raw, relative_path)
        validate_workflow_against_ajd(
            raw, relative_path, workflow_catalog=_workflow_catalog()
        )
    else:
        # Minimal unversioned definitions remain useful for runtime unit tests.
        steps = raw.get("steps") or []
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Workflow has no steps: {relative_path}")
        seen: set[str] = set()
        for step in steps:
            if not isinstance(step, dict) or not step.get("id"):
                raise ValueError(f"Every workflow step needs id: {relative_path}")
            step_id = str(step["id"])
            if step_id in seen:
                raise ValueError(f"Duplicate workflow step id {step_id}: {relative_path}")
            seen.add(step_id)
    return WorkflowDefinition(raw)


def _condition_is_true(condition: Any, context: dict[str, Any]) -> bool:
    if condition is None:
        return True
    if isinstance(condition, bool):
        return condition
    if isinstance(condition, str):
        return bool(context.get(condition))
    raise ValueError(f"Unsupported workflow condition: {condition!r}")


def _binding_name(step: dict[str, Any]) -> str:
    kind = str(step.get("kind") or "")
    if kind == "workflow" and step.get("workflow"):
        return str(step["workflow"])
    # v0.2 convention: a job capability defaults to the semantic step id.
    if step.get("capability"):
        return str(step["capability"])
    return str(step.get("handler") or step["id"])


def _execute_bound_step(
    step: dict[str, Any],
    context: dict[str, Any],
    handlers: dict[str, WorkflowHandler],
) -> None:
    binding_name = _binding_name(step)
    handler = handlers.get(binding_name)
    if handler is None:
        raise KeyError(f"No capability binding registered for workflow step: {binding_name}")
    handler(context)



def prepare_workflow_context(
    definition: WorkflowDefinition,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Create runtime context from the DSL data contract.

    v0.2 inputs must be supplied explicitly. Declared step products and workflow
    outputs are predeclared as ``None`` so application entrypoints do not need
    to repeat orchestration-only empty values.
    """
    context = dict(values)
    if definition.dsl_version == DSL_VERSION:
        missing = [name for name in definition.input_names if name not in context]
        if missing:
            raise ValueError(f"Missing workflow inputs {missing}: {definition.workflow_id}")
        for name in [*definition.produced_names, *definition.output_names]:
            context.setdefault(name, None)
    return context


RecoveryHandler = Callable[[dict[str, Any], Exception, WorkflowDefinition], dict[str, Any]]


WORKFLOW_STATUS_READY = "ready"
WORKFLOW_STATUS_RUNNING = "running"
WORKFLOW_STATUS_WAITING = "waiting_for_interaction"
WORKFLOW_STATUS_COMPLETED = "completed"


@dataclass(frozen=True)
class InteractionRequest:
    workflow_id: str
    step_id: str
    interaction_id: str
    actor: str
    mode: str
    inputs: dict[str, Any]
    outputs: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "step_id": self.step_id,
            "interaction_id": self.interaction_id,
            "actor": self.actor,
            "mode": self.mode,
            "inputs": dict(self.inputs),
            "outputs": list(self.outputs),
        }


def _interaction_request(
    definition: WorkflowDefinition,
    step: dict[str, Any],
    context: dict[str, Any],
) -> InteractionRequest:
    contract = interaction_contract(str(step.get("interaction") or ""))
    names = [*contract.required_inputs, *contract.optional_inputs]
    values = {name: context.get(name) for name in names if name in context}
    missing = [name for name in contract.required_inputs if name not in context]
    if missing:
        raise ValueError(
            f"Missing interaction inputs {missing} for {contract.interaction_id}: "
            f"{definition.workflow_id}"
        )
    return InteractionRequest(
        workflow_id=definition.workflow_id,
        step_id=str(step["id"]),
        interaction_id=contract.interaction_id,
        actor=contract.actor,
        mode=contract.mode,
        inputs=values,
        outputs=contract.outputs,
    )


def _validate_interaction_response(request: InteractionRequest, values: Mapping[str, Any]) -> dict[str, Any]:
    supplied = {str(key): value for key, value in values.items()}
    expected = list(request.outputs)
    missing = [name for name in expected if name not in supplied]
    unknown = sorted(set(supplied) - set(expected))
    if missing or unknown:
        raise ValueError(
            f"Interaction response contract mismatch for {request.interaction_id}; "
            f"missing={missing}, unknown={unknown}"
        )
    contract = interaction_contract(request.interaction_id)
    constraints = contract.raw.get("constraints") or {}
    for name in expected:
        spec = contract.data_spec(name)
        value = supplied[name]
        if spec.cardinality == "many" and not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Interaction response {name} must be a collection: {request.interaction_id}"
            )
    min_selection = constraints.get("min_selection")
    if min_selection is not None and expected:
        first = supplied[expected[0]]
        if isinstance(first, (list, tuple)) and len(first) < int(min_selection):
            raise ValueError(
                f"Interaction response needs at least {min_selection} selection(s): "
                f"{request.interaction_id}"
            )
    return supplied


@dataclass
class WorkflowRun:
    """Prepared execution of one workflow definition with interaction suspension.

    The run keeps its in-memory context and current step index. Response-required
    interaction steps suspend execution without requiring a Python handler. A
    caller can render the returned interaction request, then call ``resume`` with
    the contract outputs to continue from the next step. Existing workflows with
    no interactions still complete in one ``execute`` call.
    """

    definition: WorkflowDefinition
    context: dict[str, Any]
    handlers: dict[str, WorkflowHandler]
    next_step_index: int = 0
    status: str = WORKFLOW_STATUS_READY
    pending_interaction: InteractionRequest | None = None

    def _public_waiting_result(self) -> dict[str, Any]:
        if self.pending_interaction is None:
            raise RuntimeError("Workflow is not waiting for an interaction")
        return {
            "status": WORKFLOW_STATUS_WAITING,
            "workflow_id": self.definition.workflow_id,
            "workflow_trace": list(self.context.get("workflow_trace") or []),
            "interaction": self.pending_interaction.as_dict(),
            "checkpoint": {
                "workflow_id": self.definition.workflow_id,
                "next_step_index": self.next_step_index,
                "interaction_step_id": self.pending_interaction.step_id,
            },
        }

    def _run(
        self,
        *,
        recover: Sequence[tuple[type[BaseException], RecoveryHandler]] = (),
    ) -> dict[str, Any]:
        try:
            self.status = WORKFLOW_STATUS_RUNNING
            outcome = _execute_workflow_segment(
                self.definition,
                self.context,
                self.handlers,
                start_index=self.next_step_index,
            )
            self.next_step_index = int(outcome["next_step_index"])
            self.pending_interaction = outcome.get("pending_interaction")
            if self.pending_interaction is not None:
                self.status = WORKFLOW_STATUS_WAITING
                return self._public_waiting_result()
        except Exception as error:
            for error_type, handler in recover:
                if isinstance(error, error_type):
                    return handler(self.context, error, self.definition)
            raise
        self.status = WORKFLOW_STATUS_COMPLETED
        return project_workflow_outputs(self.definition, self.context)

    def execute(
        self,
        *,
        recover: Sequence[tuple[type[BaseException], RecoveryHandler]] = (),
    ) -> dict[str, Any]:
        if self.status == WORKFLOW_STATUS_WAITING:
            return self._public_waiting_result()
        if self.status == WORKFLOW_STATUS_COMPLETED:
            return project_workflow_outputs(self.definition, self.context)
        return self._run(recover=recover)

    def checkpoint_payload(self, *, checkpoint_id: str | None = None) -> dict[str, Any]:
        from research_fellow.application.dsl.checkpoint import checkpoint_payload
        return checkpoint_payload(self, checkpoint_id=checkpoint_id)

    def save_checkpoint(self, store: Any, *, checkpoint_id: str | None = None) -> str:
        from research_fellow.application.dsl.checkpoint import save_workflow_checkpoint
        return save_workflow_checkpoint(self, store, checkpoint_id=checkpoint_id)

    def resume(
        self,
        response: Mapping[str, Any],
        *,
        recover: Sequence[tuple[type[BaseException], RecoveryHandler]] = (),
    ) -> dict[str, Any]:
        if self.status != WORKFLOW_STATUS_WAITING or self.pending_interaction is None:
            raise ValueError(f"Workflow {self.definition.workflow_id} is not waiting for interaction")
        request = self.pending_interaction
        values = _validate_interaction_response(request, response)
        self.context.update(values)
        self.context.setdefault("workflow_trace", []).append({
            "step": request.step_id,
            "status": "completed",
            "interaction": request.interaction_id,
        })
        self.pending_interaction = None
        self.status = WORKFLOW_STATUS_READY
        return self._run(recover=recover)



def workflow_result(
    definition: WorkflowDefinition,
    context: dict[str, Any],
    *,
    status: str | None = None,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a public workflow result from the declared output contract.

    Recovery paths often need to return before normal capability execution has
    populated every optional output. This helper keeps those result envelopes
    consistent with normal output projection and avoids repeating workflow id /
    trace metadata in application modules.
    """
    payload = project_workflow_outputs(definition, context)
    if status is not None:
        payload["status"] = status
    if values:
        payload.update(values)
    return payload

def capability_bindings_from_namespace(
    definition: WorkflowDefinition,
    namespace: Mapping[str, Any],
) -> dict[str, WorkflowHandler]:
    """Resolve DSL semantic capabilities from a module/class namespace.

    A capability ``analyze_paper`` is implemented by ``_analyze_paper`` (preferred)
    or ``analyze_paper``. Workflow steps use the child workflow id as their binding
    name, following the existing v0.2 runtime convention.
    """
    bindings: dict[str, WorkflowHandler] = {}
    for step in definition.steps:
        binding_name = _binding_name(step)
        if binding_name in bindings:
            continue
        candidates = (f"_{binding_name}", binding_name)
        handler = next((namespace.get(name) for name in candidates if callable(namespace.get(name))), None)
        if handler is not None:
            bindings[binding_name] = handler
    return bindings


def prepare_workflow_run_by_id(
    workflow_id: str,
    values: dict[str, Any],
    namespace: Mapping[str, Any],
) -> WorkflowRun:
    definition = load_workflow_definition_by_id(workflow_id)
    context = prepare_workflow_context(definition, values)
    handlers = capability_bindings_from_catalog(definition.steps)
    namespace_handlers = capability_bindings_from_namespace(definition, namespace)
    for name, handler in namespace_handlers.items():
        handlers.setdefault(name, handler)
    validate_workflow_bindings(definition, handlers)
    return WorkflowRun(definition=definition, context=context, handlers=handlers)


def prepare_workflow_run(
    relative_path: str,
    values: dict[str, Any],
    namespace: Mapping[str, Any],
) -> WorkflowRun:
    """Load, prepare, and bind a workflow without executing it yet.

    Keeping preparation separate from execution lets application-specific recovery
    code inspect the same runtime context when a capability raises.
    """
    definition = load_workflow_definition(relative_path)
    context = prepare_workflow_context(definition, values)

    # Action/decision implementations are resolved through the Capability Catalog.
    # Namespace binding remains only for composition adapters (subworkflows) and
    # explicit external interactions, so workflow logic no longer depends on
    # implicit Python function-name lookup for reusable capabilities.
    handlers = capability_bindings_from_catalog(definition.steps)
    namespace_handlers = capability_bindings_from_namespace(definition, namespace)
    for name, handler in namespace_handlers.items():
        handlers.setdefault(name, handler)

    validate_workflow_bindings(definition, handlers)
    return WorkflowRun(definition=definition, context=context, handlers=handlers)


def capability_bindings(*handlers: WorkflowHandler) -> dict[str, WorkflowHandler]:
    """Bind semantic capability ids to Python implementations by naming convention.

    A function named ``_analyze_paper`` implements DSL capability
    ``analyze_paper``. This keeps the semantic id in one place: the DSL.
    """
    bindings: dict[str, WorkflowHandler] = {}
    for handler in handlers:
        name = handler.__name__.lstrip("_")
        if not name:
            raise ValueError("Workflow capability handler must have a semantic name")
        if name in bindings:
            raise ValueError(f"Duplicate workflow capability binding: {name}")
        bindings[name] = handler
    return bindings


def project_workflow_outputs(
    definition: WorkflowDefinition,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Return the workflow's declared public outputs plus execution metadata."""
    result = {name: context.get(name) for name in definition.output_names}
    result["workflow_id"] = definition.workflow_id
    result["workflow_trace"] = list(context.get("workflow_trace") or [])
    return result


def _execute_workflow_segment(
    definition: WorkflowDefinition,
    context: dict[str, Any],
    handlers: dict[str, WorkflowHandler],
    *,
    start_index: int = 0,
) -> dict[str, Any]:
    """Execute a workflow segment until completion or a response-required interaction."""
    validate_workflow_bindings(definition, handlers)
    context.setdefault("workflow_id", definition.workflow_id)
    context.setdefault("workflow_trace", [])
    context.setdefault("interaction_events", [])
    steps = definition.steps

    for index in range(start_index, len(steps)):
        step = steps[index]
        if not _condition_is_true(step.get("when"), context):
            context["workflow_trace"].append({"step": step["id"], "status": "skipped"})
            continue

        context["current_step"] = step
        try:
            if str(step.get("kind") or "") == "interaction":
                request = _interaction_request(definition, step, context)
                contract = interaction_contract(request.interaction_id)
                autonomy_map = context.get("autonomy_signals") or {}
                if isinstance(autonomy_map, Mapping):
                    signal_values = autonomy_map.get(request.interaction_id, {})
                    if not isinstance(signal_values, Mapping):
                        signal_values = {}
                else:
                    signal_values = {}
                autonomy = evaluate_interaction_autonomy(request.interaction_id, signal_values, inputs=request.inputs)
                context.setdefault("autonomy_events", []).append({
                    "step": str(step["id"]),
                    **autonomy.as_dict(),
                })

                if autonomy.action == AUTONOMY_AUTO:
                    values = _validate_interaction_response(request, autonomy.resolution) if contract.requires_response else {}
                    context.update(values)
                    context["interaction_events"].append({
                        **request.as_dict(),
                        "autonomy": autonomy.action,
                        "resolution": dict(values),
                    })
                    context["workflow_trace"].append({
                        "step": step["id"],
                        "status": "auto_resolved",
                        "interaction": request.interaction_id,
                    })
                    continue

                if autonomy.action == AUTONOMY_AUTO_NOTIFY and not contract.requires_response:
                    context["interaction_events"].append({
                        **request.as_dict(),
                        "autonomy": autonomy.action,
                    })
                elif contract.requires_response:
                    context["workflow_trace"].append({
                        "step": step["id"],
                        "status": WORKFLOW_STATUS_WAITING,
                        "interaction": request.interaction_id,
                        "autonomy": autonomy.action,
                    })
                    context.pop("current_step", None)
                    return {
                        "next_step_index": index + 1,
                        "pending_interaction": request,
                    }
                else:
                    context["interaction_events"].append(request.as_dict())
            else:
                foreach_key = step.get("foreach")
                if foreach_key:
                    items = list(context.get(str(foreach_key)) or [])
                    for item_index, item in enumerate(items):
                        context["current_item"] = item
                        context["current_index"] = item_index
                        _execute_bound_step(step, context, handlers)
                    context.pop("current_item", None)
                    context.pop("current_index", None)
                else:
                    _execute_bound_step(step, context, handlers)
        except Exception as error:
            context["workflow_trace"].append({
                "step": step["id"],
                "status": "failed",
                "error_type": type(error).__name__,
            })
            context.pop("current_item", None)
            context.pop("current_index", None)
            context.pop("current_step", None)
            raise
        context["workflow_trace"].append({"step": step["id"], "status": "completed"})

    context.pop("current_step", None)
    return {"next_step_index": len(steps), "pending_interaction": None}


def execute_workflow(
    definition: WorkflowDefinition,
    context: dict[str, Any],
    handlers: dict[str, WorkflowHandler],
) -> dict[str, Any]:
    """Execute from the first step and return the mutable workflow context.

    This preserves the public v0.2 helper contract for direct callers. Response-
    required interactions stop execution at the interaction boundary; callers
    that need resume semantics should use ``WorkflowRun``.
    """
    _execute_workflow_segment(definition, context, handlers, start_index=0)
    return context
