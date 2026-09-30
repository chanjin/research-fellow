"""Human-gated M2 advisory session composed from existing advisory workflows."""
from __future__ import annotations

from typing import Any, Callable

from research_fellow.application.advisory_workflow import (
    execute_advisory_planning,
    execute_advisory_response,
)
from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.infrastructure.retrieval import KnowledgeRetriever
from research_fellow.storage import Ledger

DraftFunction = Callable[[str], str | None]


def _m2_advisory_planning(context: dict[str, Any]) -> None:
    result = execute_advisory_planning(
        request_type=str(context["request_type"]),
        question=str(context["question"]),
        context=str(context.get("context") or ""),
        recipient=str(context["recipient"]),
        recalled_context=str(context.get("recalled_context") or ""),
        draft_fn=context["draft_fn"],
    )
    context.update({name: result.get(name) for name in ("plan", "plan_draft")})


def _m2_advisory_response(context: dict[str, Any]) -> None:
    result = execute_advisory_response(
        plan=context["plan"],
        context=str(context.get("context") or ""),
        recipient=str(context["recipient"]),
        cards=list(context.get("cards") or []),
        relations=list(context.get("relations") or []),
        retriever=context["retriever"],
        draft_fn=context["draft_fn"],
        semantic=bool(context.get("semantic", False)),
        embedding_model=str(context.get("embedding_model") or "nomic-embed-text"),
    )
    for name in (
        "evidence_clusters", "judgments", "answer", "evidence_card_ids",
        "evidence_relation_ids", "unresolved_items",
    ):
        context[name] = result.get(name)


def prepare_advisory_session(
    *,
    request_type: str,
    question: str,
    context: str,
    recipient: str,
    recalled_context: str,
    cards: list[dict[str, object]],
    relations: list[dict[str, object]],
    retriever: KnowledgeRetriever,
    draft_fn: DraftFunction,
    semantic: bool = False,
    embedding_model: str = "nomic-embed-text",
    autonomy_signals: dict[str, object] | None = None,
    ledger: Ledger | None = None,
):
    """Prepare the advisory workflow that pauses at researcher plan confirmation."""
    return prepare_workflow_run(
        "m2/advisory_session.yaml",
        {
            "request_type": request_type,
            "question": question,
            "context": context,
            "recipient": recipient,
            "recalled_context": recalled_context,
            "cards": cards,
            "relations": relations,
            # Runtime-only dependencies are intentionally not workflow inputs.
            "retriever": retriever,
            "draft_fn": draft_fn,
            "semantic": semantic,
            "embedding_model": embedding_model,
            "autonomy_signals": {
                "confirm_advisory_plan": dict(autonomy_signals or {}),
            },
            "autonomy_audit_recorder": ledger.record_autonomy_decision if ledger is not None else None,
        },
        globals(),
    )


def save_advisory_session_checkpoint(
    run: Any,
    checkpoint_dir: Any,
    *,
    case_id: str,
    question: str,
    context: str,
) -> str:
    """Persist a waiting advisory session for handling from the Attention workspace."""
    from research_fellow.infrastructure.workflow_checkpoint import JsonFileCheckpointStore

    checkpoint_id = f"advisory-{case_id}"
    payload = run.checkpoint_payload(checkpoint_id=checkpoint_id)
    payload["resume_metadata"] = {
        "kind": "research_advisory",
        "case_id": case_id,
        "question": question,
        "context": context,
    }
    JsonFileCheckpointStore(checkpoint_dir).save(checkpoint_id, payload)
    return checkpoint_id
