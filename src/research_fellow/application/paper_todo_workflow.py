"""DSL-backed M2 revision To-do lifecycle workflows.

The workflow layer owns planning, grouping, evidence-grounded resolution,
verification, and post-revision backlog reconciliation. Prompt builders and
parsers in ``paper_coauthor.py`` remain capability implementations.
"""
from __future__ import annotations

from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.paper_coauthor_parsers import (
    parse_group_resolution, parse_revision_resolution_plan, parse_resolution_proposal,
    parse_todo_group_plan, parse_todo_reconciliation, parse_todo_verification,
)
from research_fellow.application.paper_coauthor_prompts import (
    group_resolution_prompt, resolution_proposal_prompt, revision_resolution_plan_prompt,
    todo_grouping_prompt, todo_reconciliation_prompt, todo_verification_prompt,
)

DraftFunction = Callable[[str], str | None]


def _propose_todo_resolution_plan(context: dict[str, Any]) -> None:
    todo = dict(context["todo"])
    prompt = revision_resolution_plan_prompt(
        dict(context["project"]),
        dict(context["manuscript"]),
        todo,
        list(context.get("linked_cards") or []),
        list(context.get("linked_papers") or []),
    )
    raw = context["draft_fn"](prompt) or ""
    context["plan"] = parse_revision_resolution_plan(raw, todo)
    context["raw_response"] = raw


def execute_todo_resolution_plan(
    *, project: dict[str, Any], manuscript: dict[str, Any], todo: dict[str, Any],
    linked_cards: list[dict[str, Any]], linked_papers: list[dict[str, Any]],
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_plan.yaml",
        {
            "project": project, "manuscript": manuscript, "todo": todo,
            "linked_cards": linked_cards, "linked_papers": linked_papers,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _group_revision_todos(context: dict[str, Any]) -> None:
    todos = list(context.get("todos") or [])
    prompt = todo_grouping_prompt(dict(context["project"]), dict(context["manuscript"]), todos)
    raw = context["draft_fn"](prompt) or ""
    context["group_plan"] = parse_todo_group_plan(raw, todos)
    context["raw_response"] = raw


def execute_todo_grouping(
    *, project: dict[str, Any], manuscript: dict[str, Any], todos: list[dict[str, Any]],
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_grouping.yaml",
        {"project": project, "manuscript": manuscript, "todos": todos, "draft_fn": draft_fn},
        globals(),
    )
    return workflow.execute()


def _propose_todo_resolution(context: dict[str, Any]) -> None:
    todo = dict(context["todo"])
    prompt = resolution_proposal_prompt(
        dict(context["project"]),
        dict(context["manuscript"]),
        todo,
        list(context.get("papers") or []),
        list(context.get("cards") or []),
        str(context.get("connection_note") or ""),
        include_full_manuscript=bool(context.get("include_full_manuscript")),
        research_artifacts=list(context.get("research_artifacts") or []),
    )
    raw = context["draft_fn"](prompt) or ""
    context["proposal"] = parse_resolution_proposal(
        raw,
        todo,
        valid_card_ids=set(str(value) for value in context.get("valid_card_ids") or []),
        valid_paper_ids=set(str(value) for value in context.get("valid_paper_ids") or []),
    )
    context["raw_response"] = raw


def execute_todo_resolution(
    *, project: dict[str, Any], manuscript: dict[str, Any], todo: dict[str, Any],
    papers: list[dict[str, Any]], cards: list[dict[str, Any]], connection_note: str,
    include_full_manuscript: bool, research_artifacts: list[dict[str, Any]],
    valid_card_ids: set[str], valid_paper_ids: set[str], draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_resolution.yaml",
        {
            "project": project, "manuscript": manuscript, "todo": todo,
            "papers": papers, "cards": cards, "connection_note": connection_note,
            "include_full_manuscript": include_full_manuscript,
            "research_artifacts": research_artifacts,
            "valid_card_ids": sorted(valid_card_ids), "valid_paper_ids": sorted(valid_paper_ids),
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _propose_group_todo_resolution(context: dict[str, Any]) -> None:
    group = dict(context["group"])
    todos = list(context.get("todos") or [])
    prompt = group_resolution_prompt(
        dict(context["project"]),
        dict(context["manuscript"]),
        group,
        todos,
        list(context.get("papers") or []),
        list(context.get("cards") or []),
        str(context.get("connection_note") or ""),
        include_full_manuscript=bool(context.get("include_full_manuscript")),
    )
    raw = context["draft_fn"](prompt) or ""
    context["group_resolution"] = parse_group_resolution(
        raw,
        group,
        todos,
        valid_card_ids=set(str(value) for value in context.get("valid_card_ids") or []),
        valid_paper_ids=set(str(value) for value in context.get("valid_paper_ids") or []),
    )
    context["raw_response"] = raw


def execute_group_todo_resolution(
    *, project: dict[str, Any], manuscript: dict[str, Any], group: dict[str, Any],
    todos: list[dict[str, Any]], papers: list[dict[str, Any]], cards: list[dict[str, Any]],
    connection_note: str, include_full_manuscript: bool, valid_card_ids: set[str],
    valid_paper_ids: set[str], draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_group_resolution.yaml",
        {
            "project": project, "manuscript": manuscript, "group": group, "todos": todos,
            "papers": papers, "cards": cards, "connection_note": connection_note,
            "include_full_manuscript": include_full_manuscript,
            "valid_card_ids": sorted(valid_card_ids), "valid_paper_ids": sorted(valid_paper_ids),
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _verify_todo_resolution(context: dict[str, Any]) -> None:
    todo = dict(context["todo"])
    prompt = todo_verification_prompt(
        dict(context["project"]),
        todo,
        dict(context["current_sentence"]),
        list(context.get("evidence_cards") or []),
    )
    raw = context["draft_fn"](prompt) or ""
    context["verification"] = parse_todo_verification(raw, todo)
    context["raw_response"] = raw


def execute_todo_verification(
    *, project: dict[str, Any], todo: dict[str, Any], current_sentence: dict[str, Any],
    evidence_cards: list[dict[str, Any]], draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_verification.yaml",
        {
            "project": project, "todo": todo, "current_sentence": current_sentence,
            "evidence_cards": evidence_cards, "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _reconcile_revision_todos(context: dict[str, Any]) -> None:
    before_todos = list(context.get("before_todos") or [])
    after_manuscript = dict(context["after_manuscript"])
    prompt = todo_reconciliation_prompt(
        dict(context["project"]),
        dict(context["before_manuscript"]),
        after_manuscript,
        before_todos,
        list(context.get("revision_diff") or []),
    )
    raw = context["draft_fn"](prompt) or ""
    context["reconciliation"] = parse_todo_reconciliation(
        raw, existing_todos=before_todos, manuscript=after_manuscript,
    )
    context["raw_response"] = raw


def execute_todo_reconciliation(
    *, project: dict[str, Any], before_manuscript: dict[str, Any], after_manuscript: dict[str, Any],
    before_todos: list[dict[str, Any]], revision_diff: list[dict[str, Any]], draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/revision_todo_reconciliation.yaml",
        {
            "project": project, "before_manuscript": before_manuscript,
            "after_manuscript": after_manuscript, "before_todos": before_todos,
            "revision_diff": revision_diff, "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()
