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
from research_fellow.application.ontology_workflow import apply_ontology_task_response
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
        subject_type = str(task.get("subject_type") or "")
        if subject_type == "paper_first_review":
            stage = "single_paper_first_review"
            prompt_source = "durable_paper_review_task"
            item_key = str(task_payload.get("paper_id") or task.get("subject_id") or "")
            expected = str(task_payload.get("expected_output") or "JSON papers[] with executive_summary, source-grounded claims, and review_note")
        elif subject_type in {
            "ontology_type_suggestion",
            "ontology_relation_suggestion",
            "ontology_facet_suggestion",
            "knowledge_relation_suggestion",
        }:
            stage = subject_type
            prompt_source = "durable_ontology_task"
            item_key = str(task.get("subject_id") or task.get("phenomenon_id") or "")
            expected = str(task_payload.get("expected_output") or "JSON ontology proposal")
        else:
            return None
        return {"external_llm_task": {
            "task_id": str(task.get("phenomenon_id") or ""),
            "stage": stage,
            "item_key": item_key,
            "prompt": str(task_payload.get("prompt") or ""),
            "prompt_source": prompt_source,
            "expected_output": expected,
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
        subject_type = str(task.get("subject_type") or "")
        if subject_type == "paper_first_review":
            try:
                outcome = apply_paper_review_response(ledger, task, response)
            except Exception as error:
                raise ValueError(f"논문 원문 리뷰 응답을 반영하지 못했습니다: {error}") from error
        elif subject_type in {
            "ontology_type_suggestion",
            "ontology_relation_suggestion",
            "ontology_facet_suggestion",
            "knowledge_relation_suggestion",
        }:
            try:
                outcome = apply_ontology_task_response(ledger, memory, task, response)
            except Exception as error:
                raise ValueError(f"Ontology 제안 응답을 반영하지 못했습니다: {error}") from error
        else:
            raise ValueError(f"지원하지 않는 research_task입니다: {subject_type}")
        return AttentionResolutionResult(str(item.get("attention_id") or ""), interaction_id, "completed", dict(outcome))

    if source_type == "workflow_interaction":
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required to resume a waiting workflow")
        return _resume_workflow_attention(
            item, values, checkpoint_dir=checkpoint_dir, runtime_binding_for=runtime_binding_for
        )

    raise ValueError(f"Attention item is not actionable through an Interaction Contract: {source_type}/{interaction_id}")

def dismiss_attention_item(
    item: Mapping[str, Any],
    *,
    ledger: Ledger,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Dismiss one Attention item by closing its durable source.

    Attention is only a projection, so deletion must update the underlying durable
    source. Independent tasks, decisions, ontology reviews and execution failures
    are closed individually. A suspended workflow interaction represents the
    current boundary of its literature round and therefore cancels that round.
    """
    source_type = str(item.get("source_type") or "")
    source_id = str(item.get("source_id") or "").strip()
    round_id = str(item.get("round_id") or "").strip()
    attention_id = str(item.get("attention_id") or "")
    interaction_id = str(item.get("interaction_id") or "").strip()
    payload = dict(item.get("payload") or {}) if isinstance(item.get("payload"), Mapping) else {}

    # A literature-discovery external-LLM Input is only the visible boundary of
    # the whole literature round.  Closing just the recovered run/failure leaves
    # the ready curation intent/search profile behind, so the same Input can be
    # projected again on the next advance.  Treat deletion as cancellation of
    # the round, matching workflow_interaction behavior.
    stage = str(payload.get("stage") or payload.get("current_stage") or "").strip()
    if (
        round_id
        and interaction_id == "provide_external_llm_result"
        and stage == "search_strategy"
        and source_type in {"auto_research_failure", "auto_research_run_attention"}
    ):
        return dismiss_attention_round(item, ledger=ledger, checkpoint_dir=checkpoint_dir)

    if source_type == "research_task":
        if not source_id:
            raise ValueError("삭제할 연구 작업 ID를 찾을 수 없습니다.")
        ledger.set_status(source_id, "cancelled")
        return {"attention_id": attention_id, "status": "cancelled", "source_type": source_type}

    if source_type == "decision_request":
        if not source_id:
            raise ValueError("삭제할 Decision ID를 찾을 수 없습니다.")
        ledger.set_status(source_id, "rejected")
        return {"attention_id": attention_id, "status": "rejected", "source_type": source_type}

    if source_type == "ontology_change_review":
        if not source_id:
            raise ValueError("삭제할 Ontology review ID를 찾을 수 없습니다.")
        ledger.resolve_ontology_change_review(
            source_id, "rejected", note="Attention에서 연구자가 삭제"
        )
        return {"attention_id": attention_id, "status": "rejected", "source_type": source_type}

    if source_type == "auto_research_failure":
        failure_id = source_id or str((item.get("payload") or {}).get("failure_id") or "").strip()
        if not failure_id:
            raise ValueError("삭제할 실행 실패 ID를 찾을 수 없습니다.")
        ledger.resolve_auto_research_failure(failure_id, status="dismissed")
        return {"attention_id": attention_id, "status": "dismissed", "source_type": source_type}

    if source_type == "auto_research_run_attention":
        if not source_id:
            raise ValueError("삭제할 실행 ID를 찾을 수 없습니다.")
        ledger.update_auto_research_run(
            source_id, status="cancelled", error_type="cancelled_by_researcher", error_message=""
        )
        return {"attention_id": attention_id, "status": "cancelled", "source_type": source_type}

    if source_type == "workflow_interaction":
        if round_id:
            return dismiss_attention_round(item, ledger=ledger, checkpoint_dir=checkpoint_dir)
        if checkpoint_dir is None or not source_id:
            raise ValueError("삭제할 Workflow checkpoint를 찾을 수 없습니다.")
        JsonFileCheckpointStore(checkpoint_dir).delete(source_id)
        return {"attention_id": attention_id, "status": "cancelled", "source_type": source_type}

    # Generic execution exceptions tied to a literature round are only safe to
    # dismiss by cancelling the round; otherwise their source is diagnostic-only.
    if round_id:
        return dismiss_attention_round(item, ledger=ledger, checkpoint_dir=checkpoint_dir)

    raise ValueError(f"이 Attention 항목은 안전하게 삭제할 수 없습니다: {source_type or 'unknown'}")


def dismiss_attention_round(
    item: Mapping[str, Any],
    *,
    ledger: Ledger,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Cancel one unfinished literature round so its projected Attention does not reappear."""
    import json

    intent_id = str(item.get("round_id") or "").strip()
    if not intent_id:
        raise ValueError("연구질문에 연결된 진행 중 문헌조사만 삭제할 수 있습니다.")

    counts = {"runs": 0, "failures": 0, "tasks": 0, "decisions": 0, "checkpoints": 0, "profiles": 0}

    for run in ledger.auto_research_runs(statuses=("running", "needs_attention"), limit=200):
        if str(run.get("intent_id") or "") != intent_id:
            continue
        ledger.update_auto_research_run(str(run.get("run_id") or ""), status="cancelled", error_type="cancelled_by_researcher", error_message="")
        counts["runs"] += 1

    for failure in ledger.auto_research_failures(status="needs_attention", limit=200):
        if str(failure.get("intent_id") or "") != intent_id:
            continue
        ledger.resolve_auto_research_failure(str(failure.get("failure_id") or ""), status="dismissed")
        counts["failures"] += 1

    for phenomenon in ledger.phenomena():
        payload = dict(phenomenon.get("payload") or {})
        if str(payload.get("intent_id") or "") != intent_id:
            continue
        status = str(phenomenon.get("status") or "")
        ptype = str(phenomenon.get("phenomenon_type") or "")
        if ptype == "research_task" and status == "ready":
            ledger.set_status(str(phenomenon.get("phenomenon_id") or ""), "cancelled")
            counts["tasks"] += 1
        elif ptype == "decision_request" and status == "proposed":
            ledger.set_status(str(phenomenon.get("phenomenon_id") or ""), "rejected")
            counts["decisions"] += 1

    for phenomenon in ledger.phenomena(type_="curation_intent"):
        if str(phenomenon.get("subject_id") or "") == intent_id and str(phenomenon.get("status") or "") in {"ready", "failed"}:
            ledger.set_status(str(phenomenon.get("phenomenon_id") or ""), "cancelled")

    for profile in ledger.search_profiles(include_deleted=False):
        if str(profile.get("intent_id") or "") == intent_id:
            if ledger.delete_search_profile(str(profile.get("profile_id") or ""), note="연구자가 진행 중 Attention을 삭제하여 문헌조사를 취소"):
                counts["profiles"] += 1

    if checkpoint_dir is not None and checkpoint_dir.exists():
        for path in checkpoint_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            metadata = payload.get("resume_metadata") if isinstance(payload, dict) else {}
            if isinstance(metadata, Mapping) and str(metadata.get("intent_id") or "") == intent_id:
                try:
                    path.unlink()
                    counts["checkpoints"] += 1
                except OSError:
                    pass

    rq_rows = ledger.research_questions_for_intent(intent_id)
    rq_id = str((rq_rows[0] if rq_rows else {}).get("rq_id") or "")
    if rq_id:
        ledger.update_research_question_status(rq_id, "hold")

    return {"intent_id": intent_id, "rq_id": rq_id, "status": "cancelled", **counts}
