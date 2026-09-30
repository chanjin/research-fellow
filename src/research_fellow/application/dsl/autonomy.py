"""Autonomy policy evaluation for human interaction boundaries.

The policy decides *whether* an Interaction Contract needs a human at runtime.
It does not redefine the interaction itself and it does not describe UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

import yaml

from research_fellow.application.dsl.interaction import load_interaction_contracts

AUTONOMY_POLICY_VERSION = "ajd-autonomy/v0.1"
DEFAULT_AUTONOMY_PROFILE = "default"
AUTONOMY_AUTO = "auto"
AUTONOMY_AUTO_NOTIFY = "auto_notify"
AUTONOMY_ESCALATE = "escalate"
AUTONOMY_ACTIONS = {AUTONOMY_AUTO, AUTONOMY_AUTO_NOTIFY, AUTONOMY_ESCALATE}


@dataclass(frozen=True)
class AutonomyDecision:
    interaction_id: str
    action: str
    reasons: tuple[str, ...]
    resolution: dict[str, Any]
    audit_record: bool
    audit_notify: bool
    signals: dict[str, Any]
    policy_id: str
    policy_profile: str
    autonomy_level: str
    evaluated_at: str

    @property
    def requires_human(self) -> bool:
        return self.action == AUTONOMY_ESCALATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "action": self.action,
            "requires_human": self.requires_human,
            "reasons": list(self.reasons),
            "resolution": dict(self.resolution),
            "audit": {"record": self.audit_record, "notify": self.audit_notify},
            "signals": dict(self.signals),
            "policy_id": self.policy_id,
            "policy_profile": self.policy_profile,
            "autonomy_level": self.autonomy_level,
            "evaluated_at": self.evaluated_at,
        }


@lru_cache(maxsize=None)
def load_autonomy_policies(profile: str = DEFAULT_AUTONOMY_PROFILE) -> dict[str, dict[str, Any]]:
    path = resources.files("research_fellow").joinpath("autonomy", "policies", f"{profile}.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Autonomy policy profile must be a mapping: {profile}")
    if raw.get("policy") != AUTONOMY_POLICY_VERSION:
        raise ValueError(f"Unsupported autonomy policy version: {raw.get('policy')}")
    if str(raw.get("profile") or "") != profile:
        raise ValueError(f"Autonomy policy profile mismatch: {profile}")
    result: dict[str, dict[str, Any]] = {}
    for item in raw.get("interactions") or []:
        if not isinstance(item, Mapping) or not str(item.get("id") or "").strip():
            raise ValueError(f"Invalid autonomy policy entry in profile {profile}")
        interaction_id = str(item["id"])
        if interaction_id in result:
            raise ValueError(f"Duplicate autonomy policy for {interaction_id}: {profile}")
        result[interaction_id] = dict(item)
    return result


def autonomy_policy(interaction_id: str, profile: str = DEFAULT_AUTONOMY_PROFILE) -> dict[str, Any] | None:
    return load_autonomy_policies(profile).get(interaction_id)


def _matches_condition(signals: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    if len(condition) != 1:
        raise ValueError(f"Autonomy condition must contain exactly one signal: {condition}")
    key, expected = next(iter(condition.items()))
    actual = signals.get(str(key))
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _resolve_auto_resolution(
    interaction_id: str,
    policy: Mapping[str, Any],
    signals: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    on_auto = policy.get("on_auto") or {}
    resolver = str(on_auto.get("resolver") or "").strip()
    if not resolver:
        return dict(on_auto.get("resolution") or {})

    args = on_auto.get("args") or {}
    if resolver == "select_by_ids":
        input_name = str(args.get("input") or "")
        output_name = str(args.get("output") or "")
        item_id = str(args.get("item_id") or "")
        signal_ids = str(args.get("signal_ids") or "")
        requested_ids = {str(value) for value in (signals.get(signal_ids) or [])}
        selected = []
        for item in list(inputs.get(input_name) or []):
            if isinstance(item, Mapping) and str(item.get(item_id) or "") in requested_ids:
                selected.append(item)
        return {output_name: selected}

    if resolver == "approve_all":
        input_name = str(args.get("input") or "")
        output_name = str(args.get("output") or "")
        item_id = str(args.get("item_id") or "")
        output_id = str(args.get("output_id") or item_id)
        decision = str(args.get("decision") or "approved")
        note = str(args.get("note") or "autonomy_policy")
        resolutions = []
        for item in list(inputs.get(input_name) or []):
            if not isinstance(item, Mapping):
                continue
            identifier = str(item.get(item_id) or "").strip()
            if identifier:
                resolutions.append({output_id: identifier, "decision": decision, "note": note})
        return {output_name: resolutions}

    raise ValueError(f"Unknown autonomy auto resolver {resolver!r}: {interaction_id}")


def evaluate_interaction_autonomy(
    interaction_id: str,
    signals: Mapping[str, Any] | None = None,
    *,
    inputs: Mapping[str, Any] | None = None,
    profile: str = DEFAULT_AUTONOMY_PROFILE,
) -> AutonomyDecision:
    policy = autonomy_policy(interaction_id, profile)
    normalized = {str(key): value for key, value in (signals or {}).items()}
    from research_fellow.application.dsl.autonomy_classification import load_autonomy_classification
    classification = load_autonomy_classification(profile).get(interaction_id)
    autonomy_level = classification.current_level if classification is not None else ""
    evaluated_at = datetime.now(UTC).isoformat(timespec="seconds")
    if policy is None:
        return AutonomyDecision(
            interaction_id=interaction_id,
            action=AUTONOMY_ESCALATE,
            reasons=("no_policy",),
            resolution={},
            audit_record=True,
            audit_notify=False,
            signals=normalized,
            policy_id="no_policy",
            policy_profile=profile,
            autonomy_level=autonomy_level,
            evaluated_at=evaluated_at,
        )

    default = str(policy.get("default") or AUTONOMY_ESCALATE)
    if default not in AUTONOMY_ACTIONS:
        raise ValueError(f"Invalid autonomy default {default!r}: {interaction_id}")

    required = [str(item) for item in (policy.get("required_signals") or [])]
    missing = [name for name in required if name not in normalized or normalized[name] in (None, "")]
    if missing and str(policy.get("missing_signals") or AUTONOMY_ESCALATE) == AUTONOMY_ESCALATE:
        action = AUTONOMY_ESCALATE
        reasons = tuple(f"missing:{name}" for name in missing)
    else:
        action = default
        reasons_list: list[str] = [f"default:{default}"]
        clauses = (policy.get("escalate_when") or {}).get("any") or []
        matched: list[str] = []
        for clause in clauses:
            if isinstance(clause, Mapping) and _matches_condition(normalized, clause):
                key, _value = next(iter(clause.items()))
                matched.append(f"{key}={normalized.get(str(key))}")
        if matched:
            action = AUTONOMY_ESCALATE
            reasons_list = matched
        reasons = tuple(reasons_list)

    resolution = _resolve_auto_resolution(interaction_id, policy, normalized, dict(inputs or {})) if action in {AUTONOMY_AUTO, AUTONOMY_AUTO_NOTIFY} else {}
    audit = policy.get("audit") or {}
    return AutonomyDecision(
        interaction_id=interaction_id,
        action=action,
        reasons=reasons,
        resolution=resolution,
        audit_record=bool(audit.get("record", True)),
        audit_notify=bool(audit.get("notify", False)),
        signals=normalized,
        policy_id=str(policy.get("policy_id") or f"{profile}:{interaction_id}"),
        policy_profile=profile,
        autonomy_level=autonomy_level,
        evaluated_at=evaluated_at,
    )


def validate_autonomy_policies(profile: str = DEFAULT_AUTONOMY_PROFILE) -> dict[str, Any]:
    policies = load_autonomy_policies(profile)
    contracts = load_interaction_contracts()
    unknown = sorted(set(policies) - set(contracts))
    if unknown:
        raise ValueError(f"Autonomy policies reference unknown interactions: {unknown}")

    auto = auto_notify = escalate_defaults = 0
    for interaction_id, policy in policies.items():
        default = str(policy.get("default") or AUTONOMY_ESCALATE)
        if default not in AUTONOMY_ACTIONS:
            raise ValueError(f"Invalid autonomy default {default!r}: {interaction_id}")
        if default == AUTONOMY_AUTO:
            auto += 1
            contract = contracts[interaction_id]
            on_auto = policy.get("on_auto") or {}
            resolver = str(on_auto.get("resolver") or "").strip()
            if resolver:
                if resolver not in {"select_by_ids", "approve_all"}:
                    raise ValueError(f"Unknown autonomy auto resolver {resolver!r}: {interaction_id}")
                output_name = str((on_auto.get("args") or {}).get("output") or "")
                if contract.requires_response and output_name not in contract.outputs:
                    raise ValueError(f"Auto resolver output mismatch for {interaction_id}: {output_name!r}")
            else:
                resolution = dict(on_auto.get("resolution") or {})
                missing_outputs = sorted(set(contract.outputs) - set(resolution))
                unknown_outputs = sorted(set(resolution) - set(contract.outputs))
                if contract.requires_response and (missing_outputs or unknown_outputs):
                    raise ValueError(
                        f"Auto resolution mismatch for {interaction_id}: "
                        f"missing={missing_outputs}, unknown={unknown_outputs}"
                    )
        elif default == AUTONOMY_AUTO_NOTIFY:
            auto_notify += 1
            contract = contracts[interaction_id]
            if contract.requires_response:
                on_auto = policy.get("on_auto") or {}
                resolver = str(on_auto.get("resolver") or "").strip()
                if resolver:
                    if resolver not in {"select_by_ids", "approve_all"}:
                        raise ValueError(f"Unknown autonomy auto resolver {resolver!r}: {interaction_id}")
                    output_name = str((on_auto.get("args") or {}).get("output") or "")
                    if output_name not in contract.outputs:
                        raise ValueError(f"AUTO_NOTIFY resolver output mismatch for {interaction_id}: {output_name!r}")
                else:
                    resolution = dict(on_auto.get("resolution") or {})
                    missing_outputs = sorted(set(contract.outputs) - set(resolution))
                    unknown_outputs = sorted(set(resolution) - set(contract.outputs))
                    if missing_outputs or unknown_outputs:
                        raise ValueError(
                            f"AUTO_NOTIFY resolution mismatch for {interaction_id}: "
                            f"missing={missing_outputs}, unknown={unknown_outputs}"
                        )
        else:
            escalate_defaults += 1

    return {
        "version": AUTONOMY_POLICY_VERSION,
        "profile": profile,
        "policy_count": len(policies),
        "default_auto": auto,
        "default_auto_notify": auto_notify,
        "default_escalate": escalate_defaults,
        "uncovered_interactions": sorted(set(contracts) - set(policies)),
    }
