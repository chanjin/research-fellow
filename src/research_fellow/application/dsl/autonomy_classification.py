"""Autonomy level classification for human interactions.

The classification is a design/read-model layer.  It explains the current and
intended human involvement for each Interaction Contract without changing the
runtime policy semantics.  H0-H3 are therefore not executable policy actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

import yaml

from research_fellow.application.dsl.interaction import load_interaction_contracts

AUTONOMY_CLASSIFICATION_VERSION = "ajd-autonomy-classification/v0.1"
AUTONOMY_LEVELS = {"H0", "H1", "H2", "H3"}
AUTONOMY_CATEGORIES = {"task_entry", "oversight", "notification"}
ESCALATION_DIMENSIONS = {
    "impact", "reversibility", "evidence_sufficiency", "novelty", "conflict", "external_commitment"
}


@dataclass(frozen=True)
class InteractionAutonomyClass:
    interaction_id: str
    category: str
    current_level: str
    target_level: str
    rationale: str
    escalation_dimensions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "category": self.category,
            "current_level": self.current_level,
            "target_level": self.target_level,
            "rationale": self.rationale,
            "escalation_dimensions": list(self.escalation_dimensions),
        }


@lru_cache(maxsize=None)
def load_autonomy_classification(profile: str = "default") -> dict[str, InteractionAutonomyClass]:
    path = resources.files("research_fellow").joinpath("autonomy", "classification.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Autonomy classification must be a mapping")
    if raw.get("classification") != AUTONOMY_CLASSIFICATION_VERSION:
        raise ValueError(f"Unsupported autonomy classification version: {raw.get('classification')}")
    if str(raw.get("profile") or "") != profile:
        raise ValueError(f"Autonomy classification profile mismatch: {profile}")

    result: dict[str, InteractionAutonomyClass] = {}
    for item in raw.get("interactions") or []:
        if not isinstance(item, Mapping):
            raise ValueError("Autonomy classification entry must be a mapping")
        interaction_id = str(item.get("id") or "").strip()
        category = str(item.get("category") or "").strip()
        current = str(item.get("current_level") or "").strip()
        target = str(item.get("target_level") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        dimensions = tuple(str(x) for x in (item.get("escalation_dimensions") or []))
        if not interaction_id or interaction_id in result:
            raise ValueError(f"Invalid or duplicate autonomy classification id: {interaction_id!r}")
        if category not in AUTONOMY_CATEGORIES:
            raise ValueError(f"Invalid autonomy category {category!r}: {interaction_id}")
        if current not in AUTONOMY_LEVELS or target not in AUTONOMY_LEVELS:
            raise ValueError(f"Invalid autonomy level for {interaction_id}: {current}/{target}")
        unknown = sorted(set(dimensions) - ESCALATION_DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown escalation dimensions for {interaction_id}: {unknown}")
        if not rationale:
            raise ValueError(f"Autonomy classification rationale is required: {interaction_id}")
        result[interaction_id] = InteractionAutonomyClass(
            interaction_id, category, current, target, rationale, dimensions
        )
    return result


def validate_autonomy_classification(profile: str = "default") -> dict[str, Any]:
    classes = load_autonomy_classification(profile)
    contracts = load_interaction_contracts()
    missing = sorted(set(contracts) - set(classes))
    unknown = sorted(set(classes) - set(contracts))
    if missing or unknown:
        raise ValueError(f"Autonomy classification coverage mismatch: missing={missing}, unknown={unknown}")

    by_level = {level: 0 for level in sorted(AUTONOMY_LEVELS)}
    by_target = {level: 0 for level in sorted(AUTONOMY_LEVELS)}
    by_category = {category: 0 for category in sorted(AUTONOMY_CATEGORIES)}
    transition_candidates: list[str] = []
    for item in classes.values():
        by_level[item.current_level] += 1
        by_target[item.target_level] += 1
        by_category[item.category] += 1
        if item.current_level != item.target_level:
            transition_candidates.append(item.interaction_id)
    return {
        "version": AUTONOMY_CLASSIFICATION_VERSION,
        "profile": profile,
        "interaction_count": len(classes),
        "current_levels": by_level,
        "target_levels": by_target,
        "categories": by_category,
        "transition_candidates": sorted(transition_candidates),
    }
