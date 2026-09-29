"""Prompt builders for the paper coauthor capabilities.

Workflow ordering lives in AJD Workflow DSL; this module only renders prompts.
"""
from __future__ import annotations
from typing import Any
from research_fellow.application.paper_coauthor_manuscript import (
    revision_focus_context, sentences, targeted_revision_context,
)

def draft_prompt(project: dict[str,Any], cards: list[dict[str,Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt("m2_short_paper_draft.j2",project=project,cards=cards)

def paper_proposal_prompt(
    *, title: str, research_question: str, research_context: str,
    cards: list[dict[str,Any]], papers: list[dict[str,Any]],
) -> str:
    """Build a bounded pre-draft framing and provisional novelty review task."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_paper_proposal_review.j2",title=title,research_question=research_question,
        research_context=research_context,cards=cards,papers=papers,
    )

def writing_spec_guidance_prompt(
    *, project: dict[str,Any], proposal: dict[str,Any], current_spec: dict[str,Any],
    cards: list[dict[str,Any]], papers: list[dict[str,Any]],
) -> str:
    """Guide the researcher from confirmed framing to an editable writing contract."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_writing_spec_guidance.j2",project=project,proposal=proposal,
        current_spec=current_spec,cards=cards,papers=papers,
    )

def review_prompt(project: dict[str,Any], manuscript: dict[str,Any], cards: list[dict[str,Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt("m2_short_paper_review.j2",project=project,manuscript=manuscript,cards=cards)

def appendix_prompt(project: dict[str,Any], manuscript: dict[str,Any], cards: list[dict[str,Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_short_paper_appendix.j2",project=project,manuscript=manuscript,
        sentences=sentences(manuscript),cards=cards,
    )

def revision_prompt(project: dict[str,Any], manuscript: dict[str,Any], comments: list[dict[str,Any]], added_cards: list[dict[str,Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    targets = targeted_revision_context(manuscript, comments)
    return render_prompt(
        "m2_short_paper_revision.j2", project=project, manuscript_title=manuscript.get("title", ""),
        targets=targets, comments=comments, added_cards=added_cards,
    )

def full_revision_prompt(
    project: dict[str, Any], manuscript: dict[str, Any], todo: dict[str, Any],
    added_cards: list[dict[str, Any]], references: list[dict[str, Any]],
) -> str:
    """Build a whole-manuscript impact pass after one To-do gains new evidence."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_short_paper_full_revision.j2", project=project, manuscript=manuscript,
        sentences=sentences(manuscript), todo=todo, added_cards=added_cards,
        references=references, appendix_claims=list(manuscript.get("appendix_claims") or []),
    )

def todo_verification_prompt(
    project: dict[str,Any], todo: dict[str,Any], current_sentence: dict[str,Any],
    evidence_cards: list[dict[str,Any]],
) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_short_paper_todo_verification.j2", project=project, todo=todo,
        current_sentence=current_sentence, evidence_cards=evidence_cards,
    )

def resolution_proposal_prompt(
    project: dict[str,Any], manuscript: dict[str,Any], todo: dict[str,Any],
    papers: list[dict[str,Any]], cards: list[dict[str,Any]], connection_note: str,
    *, include_full_manuscript: bool=False, research_artifacts: list[dict[str,Any]] | None=None,
) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    focus=revision_focus_context(manuscript,todo)
    return render_prompt(
        "m2_revision_resolution_proposal.j2",project=project,manuscript=manuscript,
        todo=todo,papers=papers,cards=cards,focus=focus,
        research_artifacts=list(research_artifacts or []),
        full_manuscript_sentences=sentences(manuscript) if include_full_manuscript else [],
        include_full_manuscript=include_full_manuscript,connection_note=connection_note,
    )

def todo_grouping_prompt(
    project: dict[str,Any], manuscript: dict[str,Any], todos: list[dict[str,Any]],
) -> str:
    """Ask an LLM to propose bounded groups without sending any paper full text."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_revision_todo_grouping.j2",project=project,manuscript=manuscript,
        manuscript_sentences=sentences(manuscript),todos=todos,
    )

def group_resolution_prompt(
    project: dict[str,Any], manuscript: dict[str,Any], group: dict[str,Any],
    todos: list[dict[str,Any]], papers: list[dict[str,Any]], cards: list[dict[str,Any]],
    connection_note: str, *, include_full_manuscript: bool=False,
) -> str:
    """Build one evidence-heavy task for several researcher-confirmed, related To-dos."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    contexts=[];seen_paragraphs:set[tuple[str,str]]=set()
    for todo in todos:
        focus=revision_focus_context(manuscript,todo)
        key=(focus.get("section_id",""),focus.get("paragraph_id",""))
        if key not in seen_paragraphs:
            seen_paragraphs.add(key);contexts.append(focus)
    return render_prompt(
        "m2_revision_group_resolution.j2",project=project,group=group,todos=todos,
        contexts=contexts,papers=papers,cards=cards,connection_note=connection_note,
        include_full_manuscript=include_full_manuscript,
        full_manuscript_sentences=sentences(manuscript) if include_full_manuscript else [],
    )

def revision_resolution_plan_prompt(
    project: dict[str,Any], manuscript: dict[str,Any], todo: dict[str,Any],
    linked_cards: list[dict[str,Any]], linked_papers: list[dict[str,Any]],
) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_revision_resolution_plan.j2",project=project,manuscript=manuscript,todo=todo,
        linked_cards=linked_cards,linked_papers=linked_papers,
    )

def todo_reconciliation_prompt(
    project: dict[str,Any], before_manuscript: dict[str,Any], after_manuscript: dict[str,Any],
    before_todos: list[dict[str,Any]], revision_diff: list[dict[str,Any]],
) -> str:
    """Reassess the work backlog after one manuscript revision."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m2_revision_todo_reconciliation.j2",project=project,
        before_manuscript=before_manuscript,after_manuscript=after_manuscript,
        before_todos=before_todos,revision_diff=revision_diff,
    )
