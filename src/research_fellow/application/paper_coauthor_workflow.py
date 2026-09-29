"""DSL-backed M2 paper coauthor workflows.

The workflow layer owns the paper-coauthor orchestration (proposal, draft,
review, and revision). Existing prompt builders and parsers in
``paper_coauthor.py`` remain capability implementations.
"""
from __future__ import annotations

from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.paper_coauthor import (
    apply_review,
    apply_revisions,
    draft_prompt,
    full_revision_prompt,
    paper_proposal_prompt,
    parse_manuscript,
    parse_paper_proposal,
    parse_review,
    revision_prompt,
    review_prompt,
)

DraftFunction = Callable[[str], str | None]


def _review_paper_proposal(context: dict[str, Any]) -> None:
    prompt = paper_proposal_prompt(
        title=str(context["title"]),
        research_question=str(context["research_question"]),
        research_context=str(context.get("research_context") or ""),
        cards=list(context.get("cards") or []),
        papers=list(context.get("papers") or []),
    )
    raw = context["draft_fn"](prompt) or ""
    context["proposal_review"] = parse_paper_proposal(raw)
    context["raw_response"] = raw


def execute_paper_proposal_review(
    *,
    title: str,
    research_question: str,
    research_context: str,
    cards: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/paper_proposal.yaml",
        {
            "title": title,
            "research_question": research_question,
            "research_context": research_context,
            "cards": cards,
            "papers": papers,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _generate_paper_draft(context: dict[str, Any]) -> None:
    project = dict(context["project"])
    cards = list(context.get("cards") or [])
    raw = context["draft_fn"](draft_prompt(project, cards)) or ""
    context["raw_response"] = raw


def _normalize_paper_draft(context: dict[str, Any]) -> None:
    context["manuscript"] = parse_manuscript(
        str(context.get("raw_response") or ""),
        valid_card_ids=set(str(value) for value in context.get("valid_card_ids") or []),
        version=int(context["version"]),
    )


def execute_paper_draft(
    *,
    project: dict[str, Any],
    cards: list[dict[str, Any]],
    valid_card_ids: set[str],
    version: int,
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/paper_draft.yaml",
        {
            "project": project,
            "cards": cards,
            "valid_card_ids": sorted(valid_card_ids),
            "version": version,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _review_manuscript(context: dict[str, Any]) -> None:
    project = dict(context["project"])
    manuscript = dict(context["manuscript"])
    cards = list(context.get("cards") or [])
    raw = context["draft_fn"](review_prompt(project, manuscript, cards)) or ""
    annotations = parse_review(raw, manuscript)
    reviewed = apply_review(manuscript, annotations)
    reviewed["version"] = int(context["version"])
    context["annotations"] = annotations
    context["reviewed_manuscript"] = reviewed
    context["raw_response"] = raw


def execute_paper_review(
    *,
    project: dict[str, Any],
    manuscript: dict[str, Any],
    cards: list[dict[str, Any]],
    version: int,
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/paper_review.yaml",
        {
            "project": project,
            "manuscript": manuscript,
            "cards": cards,
            "version": version,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _revise_target_sentence(context: dict[str, Any]) -> None:
    project = dict(context["project"])
    manuscript = dict(context["manuscript"])
    comments = list(context.get("comments") or [])
    added_cards = list(context.get("added_cards") or [])
    raw = context["draft_fn"](revision_prompt(project, manuscript, comments, added_cards)) or ""
    revised, diff = apply_revisions(
        raw,
        manuscript,
        valid_card_ids=set(str(value) for value in context.get("valid_card_ids") or []),
        version=int(context["version"]),
        allowed_sentence_ids={str(context["sentence_id"])},
    )
    context["revised_manuscript"] = revised
    context["diff"] = diff
    context["raw_response"] = raw


def execute_targeted_paper_revision(
    *,
    project: dict[str, Any],
    manuscript: dict[str, Any],
    comments: list[dict[str, Any]],
    added_cards: list[dict[str, Any]],
    valid_card_ids: set[str],
    version: int,
    sentence_id: str,
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/paper_targeted_revision.yaml",
        {
            "project": project,
            "manuscript": manuscript,
            "comments": comments,
            "added_cards": added_cards,
            "valid_card_ids": sorted(valid_card_ids),
            "version": version,
            "sentence_id": sentence_id,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _revise_full_manuscript(context: dict[str, Any]) -> None:
    project = dict(context["project"])
    manuscript = dict(context["manuscript"])
    todo = dict(context["todo"])
    added_cards = list(context.get("added_cards") or [])
    references = list(context.get("references") or [])
    raw = context["draft_fn"](full_revision_prompt(project, manuscript, todo, added_cards, references)) or ""
    revised, diff = apply_revisions(
        raw,
        manuscript,
        valid_card_ids=set(str(value) for value in context.get("valid_card_ids") or []),
        version=int(context["version"]),
    )
    context["revised_manuscript"] = revised
    context["diff"] = diff
    context["raw_response"] = raw


def execute_full_paper_revision(
    *,
    project: dict[str, Any],
    manuscript: dict[str, Any],
    todo: dict[str, Any],
    added_cards: list[dict[str, Any]],
    references: list[dict[str, Any]],
    valid_card_ids: set[str],
    version: int,
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    workflow = prepare_workflow_run(
        "m2/paper_full_revision.yaml",
        {
            "project": project,
            "manuscript": manuscript,
            "todo": todo,
            "added_cards": added_cards,
            "references": references,
            "valid_card_ids": sorted(valid_card_ids),
            "version": version,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()
