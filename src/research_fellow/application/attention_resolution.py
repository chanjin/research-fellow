"""Application service for resolving Attention items through Interaction Contracts.

Attention remains a read model.  This module reconstructs the declared
Interaction inputs from each projected item and delegates submitted responses to
existing durable application services.  Suspended workflows are resumed from
checkpoints rather than from UI session state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from research_fellow.application.decision_interaction import apply_researcher_decision_resolutions
from research_fellow.application.dsl.checkpoint import load_workflow_checkpoint
from research_fellow.application.ontology_decision_interaction import apply_ontology_change_resolutions
from research_fellow.application.ontology_review_interaction import apply_ontology_review_with_regeneration
from research_fellow.infrastructure.workflow_checkpoint import JsonFileCheckpointStore
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class WorkflowRuntimeBinding:
    runtime_values: Mapping[str, Any]
    namespace: Mapping[str, Any]
    on_completed: Callable[[dict[str, Any], Mapping[str, Any]], None] | None = None


@dataclass(frozen=True)
class AttentionResolutionResult:
    attention_id: str
    interaction_id: str
    status: str
    outcome: dict[str, Any]


def interaction_inputs_for_attention_item(item: Mapping[str, Any], ledger: Ledger) -> dict[str, Any] | None:
    """Rebuild renderer inputs from the durable source represented by an Attention item."""
    interaction_id = str(item.get("interaction_id") or "")
    source_type = str(item.get("source_type") or "")
    payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}

    if source_type == "decision_request" and interaction_id == "resolve_pending_decision_requests":
        return {"decision_requests": [dict(payload)]}

    if source_type == "ontology_change_review":
        review = dict(payload)
        if interaction_id == "review_ontology_change":
            return {
                "ontology_change_review": review,
                "ontology_types": ledger.ontology_types(),
                "ontology_relations": ledger.ontology_type_relations(),
            }
        if interaction_id == "resolve_ontology_change_reviews":
            return {"ontology_change_reviews": [review]}

    if source_type == "workflow_interaction":
        request = payload.get("interaction") if isinstance(payload, Mapping) else None
        if isinstance(request, Mapping) and isinstance(request.get("inputs"), Mapping):
            return dict(request["inputs"])

    return None


def _resume_workflow_attention(
    item: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    checkpoint_dir: Path,
    runtime_binding_for: Callable[[str], WorkflowRuntimeBinding] | None,
) -> AttentionResolutionResult:
    checkpoint_id = str(item.get("source_id") or "").strip()
    payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    workflow_id = str(payload.get("workflow_id") or "").strip()
    if not checkpoint_id or not workflow_id:
        raise ValueError("Waiting workflow Attention item is missing checkpoint/workflow identity")

    binding = runtime_binding_for(workflow_id) if runtime_binding_for is not None else WorkflowRuntimeBinding({}, {})
    store = JsonFileCheckpointStore(checkpoint_dir)
    checkpoint_payload = store.load(checkpoint_id)
    run = load_workflow_checkpoint(
        store,
        checkpoint_id,
        runtime_values=dict(binding.runtime_values),
        namespace=dict(binding.namespace),
    )
    result = run.resume(values)
    if result.get("status") == "waiting_for_interaction":
        # Reuse the stable checkpoint identity and preserve application resume metadata
        # as the workflow reaches another human boundary.
        next_payload = run.checkpoint_payload(checkpoint_id=checkpoint_id)
        if isinstance(checkpoint_payload.get("resume_metadata"), Mapping):
            next_payload["resume_metadata"] = dict(checkpoint_payload["resume_metadata"])
        store.save(checkpoint_id, next_payload)
        status = "waiting"
    else:
        store.delete(checkpoint_id)
        status = "completed"
        if binding.on_completed is not None:
            binding.on_completed(dict(result), checkpoint_payload)
    return AttentionResolutionResult(
        attention_id=str(item.get("attention_id") or ""),
        interaction_id=str(item.get("interaction_id") or ""),
        status=status,
        outcome=dict(result),
    )


def apply_attention_response(
    item: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    ledger: Ledger,
    memory: KnowledgeMemory,
    relations: RelationMemory,
    checkpoint_dir: Path | None = None,
    runtime_binding_for: Callable[[str], WorkflowRuntimeBinding] | None = None,
    ontology_draft_fn: Callable[[str], str | None] | None = None,
) -> AttentionResolutionResult:
    """Apply one Interaction response to the item's existing durable source."""
    interaction_id = str(item.get("interaction_id") or "")
    source_type = str(item.get("source_type") or "")

    if source_type == "decision_request" and interaction_id == "resolve_pending_decision_requests":
        outcome = apply_researcher_decision_resolutions(
            ledger, memory, relations, values.get("decision_resolutions") or []
        )
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", dict(outcome))

    if source_type == "ontology_change_review" and interaction_id == "review_ontology_change":
        outcome = apply_ontology_review_with_regeneration(
            ledger, memory, values.get("ontology_review_feedback") or {}, draft_fn=ontology_draft_fn
        )
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", dict(outcome))

    if source_type == "ontology_change_review" and interaction_id == "resolve_ontology_change_reviews":
        outcome = apply_ontology_change_resolutions(
            ledger, values.get("ontology_change_resolutions") or []
        )
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", dict(outcome))

    if source_type == "workflow_interaction":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required to resume a waiting workflow")
        return _resume_workflow_attention(
            item, values, checkpoint_dir=checkpoint_dir, runtime_binding_for=runtime_binding_for
        )

    raise ValueError(f"Attention item is not actionable through an Interaction Contract: {source_type}/{interaction_id}")
