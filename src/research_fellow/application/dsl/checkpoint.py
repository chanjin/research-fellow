"""Durable workflow checkpoints for interaction-suspended AJD workflows.

Only semantic workflow state is persisted. Runtime dependencies such as database
connections, LLM clients, callables, and service objects are deliberately
excluded and must be reinjected when restoring a run.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from importlib import import_module
from pathlib import Path
import json
from typing import Any, Mapping, Protocol, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from research_fellow.application.dsl.workflow import WorkflowRun


CHECKPOINT_VERSION = "ajd-workflow-checkpoint/v0.1"


class CheckpointStore(Protocol):
    def save(self, checkpoint_id: str, payload: Mapping[str, Any]) -> None: ...
    def load(self, checkpoint_id: str) -> dict[str, Any]: ...
    def delete(self, checkpoint_id: str) -> None: ...


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def _load_symbol(qualified: str) -> Any:
    module_name, _, qualname = qualified.partition(":")
    if not module_name or not qualname:
        raise ValueError(f"Invalid checkpoint type reference: {qualified!r}")
    value: Any = import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def encode_checkpoint_value(value: Any) -> Any:
    """Encode semantic values to JSON-compatible data with minimal type tags."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [encode_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return {"__checkpoint_type__": "tuple", "items": [encode_checkpoint_value(item) for item in value]}
    if isinstance(value, set):
        return {"__checkpoint_type__": "set", "items": [encode_checkpoint_value(item) for item in value]}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Checkpoint mappings must use string keys")
        return {str(key): encode_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, Path):
        return {"__checkpoint_type__": "path", "value": str(value)}
    if isinstance(value, datetime):
        return {"__checkpoint_type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__checkpoint_type__": "date", "value": value.isoformat()}
    if is_dataclass(value):
        return {
            "__checkpoint_type__": "dataclass",
            "class": _qualified_name(value),
            "fields": {
                field.name: encode_checkpoint_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return {
            "__checkpoint_type__": "pydantic",
            "class": _qualified_name(value),
            "fields": encode_checkpoint_value(model_dump(mode="python")),
        }
    raise TypeError(
        f"Workflow semantic state contains non-checkpointable value {type(value).__name__}; "
        "keep runtime dependencies outside declared workflow data or provide a semantic value type"
    )


def decode_checkpoint_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_checkpoint_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__checkpoint_type__")
    if not tag:
        return {str(key): decode_checkpoint_value(item) for key, item in value.items()}
    if tag == "tuple":
        return tuple(decode_checkpoint_value(item) for item in value.get("items") or [])
    if tag == "set":
        return set(decode_checkpoint_value(item) for item in value.get("items") or [])
    if tag == "path":
        return Path(str(value.get("value") or ""))
    if tag == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if tag == "date":
        return date.fromisoformat(str(value["value"]))
    if tag in {"dataclass", "pydantic"}:
        cls = _load_symbol(str(value.get("class") or ""))
        decoded = decode_checkpoint_value(value.get("fields") or {})
        if tag == "pydantic" and hasattr(cls, "model_validate"):
            return cls.model_validate(decoded)
        return cls(**decoded)
    raise ValueError(f"Unknown checkpoint type tag: {tag}")


def semantic_context_snapshot(run: "WorkflowRun") -> dict[str, Any]:
    """Return only workflow-declared semantic data plus runtime-neutral trace metadata."""
    definition = run.definition
    names = {
        *definition.input_names,
        *definition.produced_names,
        *definition.output_names,
        "workflow_trace",
        "interaction_events",
        "autonomy_events",
    }
    snapshot: dict[str, Any] = {}
    for name in sorted(names):
        if name in run.context:
            snapshot[name] = encode_checkpoint_value(run.context[name])
    return snapshot


def checkpoint_payload(run: "WorkflowRun", *, checkpoint_id: str | None = None) -> dict[str, Any]:
    from research_fellow.application.dsl.workflow import WORKFLOW_STATUS_WAITING

    if run.status != WORKFLOW_STATUS_WAITING or run.pending_interaction is None:
        raise ValueError("Durable checkpoints are created only while waiting for interaction")
    definition = run.definition
    cid = checkpoint_id or uuid4().hex
    return {
        "checkpoint": CHECKPOINT_VERSION,
        "checkpoint_id": cid,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "workflow_id": definition.workflow_id,
        "workflow_revision": definition.raw.get("revision"),
        "status": run.status,
        "next_step_index": int(run.next_step_index),
        "pending_interaction": {
            **run.pending_interaction.as_dict(),
            "inputs": encode_checkpoint_value(run.pending_interaction.inputs),
        },
        "semantic_context": semantic_context_snapshot(run),
    }


def save_workflow_checkpoint(
    run: "WorkflowRun",
    store: CheckpointStore,
    *,
    checkpoint_id: str | None = None,
) -> str:
    payload = checkpoint_payload(run, checkpoint_id=checkpoint_id)
    cid = str(payload["checkpoint_id"])
    store.save(cid, payload)
    return cid


def _validate_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("checkpoint") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported workflow checkpoint version: {payload.get('checkpoint')!r}")
    if payload.get("status") != "waiting_for_interaction":
        raise ValueError("Only waiting workflow checkpoints can be restored")
    if not str(payload.get("workflow_id") or "").strip():
        raise ValueError("Checkpoint is missing workflow_id")
    if not isinstance(payload.get("semantic_context"), dict):
        raise ValueError("Checkpoint semantic_context must be a mapping")
    if not isinstance(payload.get("pending_interaction"), dict):
        raise ValueError("Checkpoint pending_interaction must be a mapping")


def restore_workflow_run(
    payload: Mapping[str, Any],
    *,
    runtime_values: Mapping[str, Any] | None = None,
    namespace: Mapping[str, Any] | None = None,
) -> "WorkflowRun":
    """Restore a suspended run, reinjecting non-persisted runtime dependencies."""
    _validate_checkpoint_payload(payload)
    from research_fellow.application.dsl.workflow import (
        InteractionRequest,
        WORKFLOW_STATUS_WAITING,
        prepare_workflow_run_by_id,
    )

    semantic = decode_checkpoint_value(dict(payload["semantic_context"]))
    if not isinstance(semantic, dict):
        raise ValueError("Decoded checkpoint semantic_context must be a mapping")
    runtime = dict(runtime_values or {})
    collisions = sorted(set(runtime) & set(semantic))
    if collisions:
        raise ValueError(
            f"Runtime dependency reinjection may not overwrite semantic checkpoint state: {collisions}"
        )
    values = dict(semantic)
    values.update(runtime)
    run = prepare_workflow_run_by_id(str(payload["workflow_id"]), values, namespace or {})

    expected_revision = run.definition.raw.get("revision")
    stored_revision = payload.get("workflow_revision")
    if stored_revision != expected_revision:
        raise ValueError(
            f"Workflow revision changed since checkpoint: stored={stored_revision!r}, current={expected_revision!r}"
        )

    request_raw = dict(payload["pending_interaction"])
    run.next_step_index = int(payload.get("next_step_index") or 0)
    run.pending_interaction = InteractionRequest(
        workflow_id=str(request_raw.get("workflow_id") or run.definition.workflow_id),
        step_id=str(request_raw.get("step_id") or ""),
        interaction_id=str(request_raw.get("interaction_id") or ""),
        actor=str(request_raw.get("actor") or ""),
        mode=str(request_raw.get("mode") or ""),
        inputs=dict(decode_checkpoint_value(request_raw.get("inputs") or {})),
        outputs=[str(item) for item in (request_raw.get("outputs") or [])],
    )
    run.status = WORKFLOW_STATUS_WAITING
    return run


def load_workflow_checkpoint(
    store: CheckpointStore,
    checkpoint_id: str,
    *,
    runtime_values: Mapping[str, Any] | None = None,
    namespace: Mapping[str, Any] | None = None,
) -> "WorkflowRun":
    return restore_workflow_run(
        store.load(checkpoint_id),
        runtime_values=runtime_values,
        namespace=namespace,
    )
