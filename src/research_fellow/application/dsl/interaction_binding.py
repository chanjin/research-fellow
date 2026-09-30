"""UI bindings for interaction contracts."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping
import yaml
from research_fellow.application.dsl.interaction import InteractionContract, load_interaction_contracts
INTERACTION_BINDING_VERSION = "ajd-interaction-binding/v0.1"
DEFAULT_INTERACTION_BINDING_PROFILE = "default"
_RENDERERS = {"message", "text_input", "multi_select", "review_form", "approval", "confirm"}
_MODE_RENDERERS = {
    "inform": {"message"}, "request_input": {"text_input"}, "select": {"multi_select"},
    "review": {"review_form"}, "decide": {"approval"}, "confirm": {"confirm"},
}
@dataclass(frozen=True)
class InteractionBinding:
    interaction_id: str
    renderer: dict[str, Any]
    profile: str
    @property
    def renderer_type(self) -> str:
        return str(self.renderer.get("type") or "")
def _validate_renderer(contract: InteractionContract, renderer: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(renderer, Mapping):
        raise ValueError(f"renderer must be a mapping: {contract.interaction_id}")
    renderer_type = str(renderer.get("type") or "")
    if renderer_type not in _RENDERERS:
        raise ValueError(f"Unsupported interaction renderer {renderer_type!r}: {contract.interaction_id}")
    if renderer_type not in _MODE_RENDERERS.get(contract.mode, set()):
        raise ValueError(f"Renderer {renderer_type!r} is incompatible with interaction mode {contract.mode!r}: {contract.interaction_id}")
    common = {"type", "title", "help"}
    specific = {
        "message": {"item_label", "item_detail", "empty_label"}, "text_input": {"fields", "submit_label"},
        "multi_select": {"item_label", "item_detail", "submit_label", "empty_label"},
        "review_form": {"fields", "submit_label", "variant", "regenerate_label"},
        "approval": {"item_id", "item_label", "output_id_field", "approve_label", "defer_label", "reject_label", "comment_label", "select_all_label", "empty_label", "default_all"},
        "confirm": {"confirm_label", "cancel_label"},
    }
    unknown = set(renderer) - common - specific[renderer_type]
    if unknown:
        raise ValueError(f"Unknown renderer keys {sorted(unknown)}: {contract.interaction_id}")
    if not str(renderer.get("title") or "").strip():
        raise ValueError(f"Renderer title is required: {contract.interaction_id}")

    if renderer_type == "text_input":
        if contract.required_inputs or contract.optional_inputs or len(contract.outputs) != 1:
            raise ValueError(f"text_input v0.1 expects no contract inputs and one semantic output: {contract.interaction_id}")
        fields = renderer.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"text_input fields are required: {contract.interaction_id}")
        seen = set()
        allowed = {"name", "label", "widget", "placeholder", "help", "required", "height"}
        for field in fields:
            if not isinstance(field, Mapping) or set(field) - allowed:
                raise ValueError(f"Invalid text_input field: {contract.interaction_id}")
            name = str(field.get("name") or "").strip()
            label = str(field.get("label") or "").strip()
            widget = str(field.get("widget") or "text_input")
            if not name or not label or name in seen or widget not in {"text_input", "text_area"}:
                raise ValueError(f"Invalid text_input field definition: {contract.interaction_id}")
            seen.add(name)
    if renderer_type == "message":
        if len(contract.required_inputs) != 1 or contract.optional_inputs or contract.outputs:
            raise ValueError(f"message v0.1 expects one required input and no outputs: {contract.interaction_id}")
        if not str(renderer.get("item_label") or "").strip():
            raise ValueError(f"message item_label is required: {contract.interaction_id}")
        details = renderer.get("item_detail") or []
        if not isinstance(details, list) or any(not isinstance(x, str) or not x.strip() for x in details):
            raise ValueError(f"message item_detail must be a string list: {contract.interaction_id}")
    if renderer_type == "multi_select":
        if len(contract.required_inputs) != 1 or len(contract.outputs) != 1:
            raise ValueError(f"multi_select v0.1 expects one required input and one output: {contract.interaction_id}")
        if not str(renderer.get("item_label") or "").strip():
            raise ValueError(f"multi_select item_label is required: {contract.interaction_id}")
        details = renderer.get("item_detail") or []
        if not isinstance(details, list) or any(not isinstance(x, str) or not x.strip() for x in details):
            raise ValueError(f"multi_select item_detail must be a string list: {contract.interaction_id}")
    if renderer_type == "review_form":
        if not str(renderer.get("variant") or "").strip():
            raise ValueError(f"review_form variant is required: {contract.interaction_id}")
        if not str(renderer.get("submit_label") or "").strip():
            raise ValueError(f"review_form submit_label is required: {contract.interaction_id}")
    if renderer_type == "confirm":
        if len(contract.required_inputs) != 1 or len(contract.outputs) != 1:
            raise ValueError(f"confirm v0.1 expects one required input and one output: {contract.interaction_id}")
        if not str(renderer.get("confirm_label") or "").strip():
            raise ValueError(f"confirm confirm_label is required: {contract.interaction_id}")
        if not str(renderer.get("cancel_label") or "").strip():
            raise ValueError(f"confirm cancel_label is required: {contract.interaction_id}")
    if renderer_type == "approval":
        if len(contract.required_inputs) != 1 or len(contract.outputs) != 1:
            raise ValueError(f"approval v0.1 expects one required input and one output: {contract.interaction_id}")
        if contract.data_spec(contract.required_inputs[0]).cardinality != "many" or contract.data_spec(contract.outputs[0]).cardinality != "many":
            raise ValueError(f"approval v0.1 expects many input and many output: {contract.interaction_id}")
        for key in ("item_id", "item_label", "approve_label", "defer_label", "reject_label", "comment_label"):
            if not str(renderer.get(key) or "").strip():
                raise ValueError(f"approval {key} is required: {contract.interaction_id}")
    return dict(renderer)
@lru_cache(maxsize=8)
def load_interaction_bindings(profile: str = DEFAULT_INTERACTION_BINDING_PROFILE) -> dict[str, InteractionBinding]:
    resource = resources.files("research_fellow").joinpath("interactions", "bindings", f"{profile}.yaml")
    if not resource.is_file():
        raise ValueError(f"Unknown interaction binding profile: {profile}")
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"bindings", "profile", "interactions"}:
        raise ValueError(f"Interaction binding keys invalid: {profile}")
    if raw.get("bindings") != INTERACTION_BINDING_VERSION:
        raise ValueError(f"Unsupported interaction binding version {raw.get('bindings')!r}")
    if str(raw.get("profile") or "") != profile:
        raise ValueError(f"Interaction binding profile mismatch: expected {profile}")
    contracts = load_interaction_contracts()
    items = raw.get("interactions")
    if not isinstance(items, list):
        raise ValueError(f"Interaction bindings must be a list: {profile}")
    result: dict[str, InteractionBinding] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "renderer"}:
            raise ValueError(f"Interaction binding entry must contain id/renderer only: {profile}")
        interaction_id = str(item.get("id") or "").strip()
        if not interaction_id or interaction_id in result:
            raise ValueError(f"Invalid or duplicate interaction binding: {interaction_id!r}")
        contract = contracts.get(interaction_id)
        if contract is None:
            raise ValueError(f"Interaction binding references unknown contract: {interaction_id}")
        result[interaction_id] = InteractionBinding(interaction_id, _validate_renderer(contract, item.get("renderer") or {}), profile)
    return result
def interaction_binding(interaction_id: str, profile: str = DEFAULT_INTERACTION_BINDING_PROFILE) -> InteractionBinding:
    try:
        return load_interaction_bindings(profile)[interaction_id]
    except KeyError as exc:
        raise ValueError(f"No interaction binding for {interaction_id!r} in profile {profile!r}") from exc
def validate_interaction_bindings(profile: str = DEFAULT_INTERACTION_BINDING_PROFILE, *, strict_orphans: bool = True) -> dict[str, Any]:
    contracts = load_interaction_contracts(); bindings = load_interaction_bindings(profile)
    missing = sorted(set(contracts) - set(bindings)); orphan = sorted(set(bindings) - set(contracts))
    if missing: raise ValueError(f"Interaction contracts have no UI binding in profile {profile}: {missing}")
    if strict_orphans and orphan: raise ValueError(f"Interaction bindings have no contract in profile {profile}: {orphan}")
    return {"version": INTERACTION_BINDING_VERSION, "profile": profile, "binding_count": len(bindings), "bindings": bindings,
            "missing_bindings": missing, "orphan_bindings": orphan,
            "renderer_counts": {r: sum(1 for b in bindings.values() if b.renderer_type == r) for r in sorted(_RENDERERS) if any(b.renderer_type == r for b in bindings.values())}}
