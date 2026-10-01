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
from research_fellow.application.manual_recovery import external_recovery_prompt, validate_external_response
from research_fellow.application.search_profile_strategy import auto_search_strategy_prompt
from research_fellow.application.dsl.checkpoint import load_workflow_checkpoint
from research_fellow.application.ontology_decision_interaction import apply_ontology_change_resolutions
from research_fellow.application.ontology_review_interaction import apply_ontology_review_with_regeneration
from research_fellow.infrastructure.workflow_checkpoint import JsonFileCheckpointStore
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger
from research_fellow.application.paper_review_tasks import apply_paper_review_response
from research_fellow.application.literature_candidate_review import candidate_library_context


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

    if source_type == "research_task" and interaction_id == "provide_external_llm_result":
        task = dict(payload)
        task_payload = dict(task.get("payload") or {})
        if str(task.get("subject_type") or "") != "paper_first_review":
            return None
        return {"external_llm_task": {
            "task_id": str(task.get("phenomenon_id") or ""),
            "stage": "single_paper_first_review",
            "item_key": str(task_payload.get("paper_id") or task.get("subject_id") or ""),
            "prompt": str(task_payload.get("prompt") or ""),
            "prompt_source": "durable_paper_review_task",
            "expected_output": str(task_payload.get("expected_output") or "JSON papers[] with paper_summary and claims"),
        }}

    if source_type == "workflow_interaction":
        request = payload.get("interaction") if isinstance(payload, Mapping) else None
        if isinstance(request, Mapping) and isinstance(request.get("inputs"), Mapping):
            values = dict(request["inputs"])
            if interaction_id == "review_literature_candidates":
                candidates = []
                for candidate in list(values.get("discovered_candidates") or []):
                    if not isinstance(candidate, Mapping):
                        continue
                    row = dict(candidate)
                    row.update(candidate_library_context(ledger, row))
                    candidates.append(row)
                values["discovered_candidates"] = candidates
            return values

    if source_type == "auto_research_failure" and interaction_id == "provide_external_llm_result":
        failure = dict(payload)
        context = dict(failure.get("context") or {})
        prompt = external_recovery_prompt(failure)
        prompt_source = "durable_failure"

        # search_strategy changed from an arXiv Boolean-query planner to direct
        # external-LLM paper discovery. Historical workspaces can still contain
        # unresolved failures whose persisted recovery_prompt uses the old format.
        # Re-render this stage from the current durable SearchProfile so old DB
        # records remain actionable without mutating or silently deleting history.
        if str(failure.get("stage") or "") == "search_strategy":
            intent_id = str(failure.get("intent_id") or "").strip()
            profile = next((
                item for item in ledger.search_profiles(include_deleted=True)
                if str(item.get("intent_id") or "") == intent_id
            ), None)
            if profile is not None:
                prompt = auto_search_strategy_prompt(dict(profile))
                prompt_source = "current_search_profile"

        return {"external_llm_task": {
            "failure_id": str(failure.get("failure_id") or failure.get("id") or ""),
            "run_id": str(failure.get("run_id") or ""),
            "stage": str(failure.get("stage") or ""),
            "item_key": str(failure.get("item_key") or ""),
            "prompt": prompt,
            "prompt_source": prompt_source,
            "expected_output": str(context.get("expected_output") or "현재 stage가 요구하는 형식"),
        }}

    if source_type == "auto_research_run_attention" and interaction_id == "provide_external_llm_result":
        run = dict(payload)
        intent_id = str(run.get("intent_id") or "").strip()
        stage = str(run.get("current_stage") or "").strip()
        profile = next((
            row for row in ledger.search_profiles(include_deleted=True)
            if str(row.get("intent_id") or "") == intent_id
        ), None)
        if stage != "search_strategy" or profile is None:
            return None
        prompt = auto_search_strategy_prompt(dict(profile))
        return {"external_llm_task": {
            "failure_id": "",
            "run_id": str(run.get("run_id") or ""),
            "stage": stage,
            "item_key": "",
            "prompt": prompt,
            "prompt_source": "recovered_from_run_and_search_profile",
            "expected_output": "현재 stage가 요구하는 형식",
        }}

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

    if source_type == "auto_research_failure" and interaction_id == "provide_external_llm_result":
        failure = dict(item.get("payload") or {})
        response = str(values.get("external_llm_response") or "")
        stage = str(failure.get("stage") or "")
        context = dict(failure.get("context") or {})
        ok, message = validate_external_response(stage, response, context)
        failure_id = str(failure.get("failure_id") or failure.get("id") or "")
        run_id = str(failure.get("run_id") or "")
        item_key = str(failure.get("item_key") or "")
        ledger.record_manual_recovery_attempt(
            failure_id, run_id=run_id, stage=stage, item_key=item_key, response_text=response,
            validation_status="valid" if ok else "invalid", validation_message=message, applied=ok,
        )
        if not ok:
            raise ValueError(f"외부 LLM 응답 검증 실패: {message}")
        ledger.set_manual_recovery_override(run_id, stage=stage, item_key=item_key, response=response)
        ledger.resolve_auto_research_failure(failure_id, status="manual_resolved")
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", {
            "failure_id": failure_id, "run_id": run_id, "stage": stage, "validation": message,
        })

    if source_type == "auto_research_run_attention" and interaction_id == "provide_external_llm_result":
        run = dict(item.get("payload") or {})
        response = str(values.get("external_llm_response") or "")
        run_id = str(run.get("run_id") or "")
        stage = str(run.get("current_stage") or "")
        ok, message = validate_external_response(stage, response, {})
        if not ok:
            raise ValueError(f"외부 LLM 응답 검증 실패: {message}")
        ledger.set_manual_recovery_override(run_id, stage=stage, item_key="", response=response)
        ledger.update_auto_research_run(run_id, status="running", stage=stage)
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", {
            "run_id": run_id, "stage": stage, "validation": message, "recovered_attention": True,
        })

    if source_type == "research_task" and interaction_id == "provide_external_llm_result":
        task = dict(item.get("payload") or {})
        response = str(values.get("external_llm_response") or "").strip()
        if not response:
            raise ValueError("외부 LLM 응답이 비어 있습니다.")
        try:
            outcome = apply_paper_review_response(ledger, task, response)
        except Exception as error:
            raise ValueError(f"논문 원문 리뷰 응답을 반영하지 못했습니다: {error}") from error
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", dict(outcome))

    if source_type == "workflow_interaction":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required to resume a waiting workflow")
        return _resume_workflow_attention(
            item, values, checkpoint_dir=checkpoint_dir, runtime_binding_for=runtime_binding_for
        )

    raise ValueError(f"Attention item is not actionable through an Interaction Contract: {source_type}/{interaction_id}")
