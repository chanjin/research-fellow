"""Paper-level reading prompts; no draft here is approved knowledge."""

from __future__ import annotations

from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.paper_reading_prompts import reading_prompt, second_pass_prompt
from research_fellow.application.paper_reading_parsers import (
    parse_reading_questions, parse_reading_summary, parse_second_pass_reviews, unconsumed_reading_sections,
)
from research_fellow.application.paper_reading_assets import independent_card_context, promote_question, promote_ontology_candidate
from research_fellow.infrastructure.document_reader import ExtractedDocument
from research_fellow.storage import Ledger




Draft = Callable[[str], Any]
WORKFLOW_PATH = "m1/paper_reading_summary.yaml"


def execute_paper_reading_summary(
    ledger: Ledger,
    paper: dict[str, Any],
    analysis: dict[str, Any],
    document: ExtractedDocument,
    research_context: str,
    *,
    drafter: Draft,
    source_label: str = "LLM API",
) -> dict[str, Any]:
    """Execute the first-pass M1 paper-reading workflow.

    The YAML owns the business sequence. Python exposes two capabilities:
    analyze the paper, then commit the resulting reading assets. Prompt building,
    parsing, and persistence are implementation details of those capabilities.
    """
    workflow = prepare_workflow_run(WORKFLOW_PATH, {
        "ledger": ledger,
        "paper": paper,
        "analysis": analysis,
        "document": document,
        "research_context": research_context,
        "drafter": drafter,
        "source_label": source_label,
    }, globals())
    return workflow.execute()


def _analyze_paper(context: dict[str, Any]) -> None:
    prompt = reading_prompt(context["document"], context["paper"], context.get("research_context", ""))
    result = context["drafter"](prompt)
    context["draft_result"] = result
    if isinstance(result, str):
        text = result
        ok = bool(text.strip())
    else:
        ok = bool(getattr(result, "ok", False))
        text = str(getattr(result, "text", "") or "")
    context["raw_output"] = text
    context["generation_succeeded"] = ok
    context["generation_failed"] = not ok
    if not ok:
        return
    context["summary"] = parse_reading_summary(text)
    context["reading_questions"] = parse_reading_questions(text)
    context["unconsumed_sections"] = unconsumed_reading_sections(text)


def _commit_reading_assets(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    paper = context["paper"]
    analysis = context.get("analysis") or {}
    output = context.get("raw_output", "")
    source_label = context.get("source_label", "LLM API")
    raw_output = output if source_label == "LLM API" else f"[생성 경로: {source_label}]\n\n{output}"
    ledger.save_paper_analysis(
        paper["paper_id"],
        research_question=context.get("research_context", ""),
        summary=context.get("summary", ""),
        reading_raw_output=raw_output,
        researcher_note=analysis.get("researcher_note", ""),
        generated=True,
    )
    context["paper_analysis"] = ledger.paper_analysis(paper["paper_id"]) or {}
    questions = context.get("reading_questions") or []
    context["saved_question_ids"] = ledger.add_paper_reading_questions(paper["paper_id"], questions) if questions else []
    ledger.update_shelf_paper(paper["paper_id"], shelf_status=paper["shelf_status"], reading_status="read")
    context["reading_status"] = "read"






















EVIDENCE_WORKFLOW_PATH = "m1/evidence_investigation.yaml"


def execute_evidence_investigation(
    ledger: Ledger,
    paper: dict[str, Any],
    document: ExtractedDocument,
    selected_questions: list[dict[str, Any]],
    *,
    drafter: Draft,
) -> dict[str, Any]:
    """Re-read one paper for researcher-selected reading questions.

    This is intentionally separate from the default first-pass reading workflow.
    It is invoked only when the researcher wants deeper evidence verification.
    """
    if not selected_questions:
        raise ValueError("Evidence investigation requires at least one selected reading question")
    workflow = prepare_workflow_run(EVIDENCE_WORKFLOW_PATH, {
        "ledger": ledger,
        "paper": paper,
        "document": document,
        "selected_questions": selected_questions,
        "drafter": drafter,
    }, globals())
    return workflow.execute()


def _investigate_evidence(context: dict[str, Any]) -> None:
    selected = list(context.get("selected_questions") or [])
    prompt = second_pass_prompt(context["document"], context["paper"], selected)
    result = context["drafter"](prompt)
    context["draft_result"] = result
    if isinstance(result, str):
        text = result
        ok = bool(text.strip())
    else:
        ok = bool(getattr(result, "ok", False))
        text = str(getattr(result, "text", "") or "")
    context["raw_output"] = text
    if not ok:
        context["reviews"] = []
        context["validation_error"] = "generation_failed"
        context["generation_succeeded"] = False
        context["generation_failed"] = True
        return

    reviews = parse_second_pass_reviews(text)
    expected_ids = {str(item.get("question_id") or "") for item in selected if item.get("question_id")}
    actual_ids = {str(item.get("question_id") or "") for item in reviews if item.get("question_id")}
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    valid = bool(expected_ids) and not missing and not unexpected and len(reviews) == len(expected_ids)

    context["reviews"] = reviews
    context["validation_error"] = ""
    if not valid:
        parts: list[str] = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if unexpected:
            parts.append("unexpected=" + ",".join(unexpected))
        if len(reviews) != len(expected_ids):
            parts.append(f"count={len(reviews)}/{len(expected_ids)}")
        context["validation_error"] = "; ".join(parts) or "invalid_review_set"
    context["generation_succeeded"] = valid
    context["generation_failed"] = not valid


def _commit_evidence_reviews(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    paper = context["paper"]
    reviews = list(context.get("reviews") or [])
    context["saved_review_count"] = ledger.upsert_paper_reading_reviews(reviews)
    context["reading_reviews"] = ledger.paper_reading_reviews(paper["paper_id"])
