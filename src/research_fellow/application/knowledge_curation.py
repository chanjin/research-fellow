"""M1 knowledge curation from paper-reading evidence.

The workflow turns researcher-selected reading questions and optional second-pass
reviews into non-authoritative knowledge candidates. M1 may recommend a new
card, reinforcement of an approved card, a contested card, or a hold. Semantic
memory changes remain gated by researcher approval.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable

from research_fellow.application.curation import _compact_title
from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.duplicate_review import similar_approved_cards
from research_fellow.domain.knowledge import KnowledgeCard
from research_fellow.storage import Ledger


Draft = Callable[[str], Any]
WORKFLOW_PATH = "m1/knowledge_curation.yaml"


def execute_knowledge_curation(
    ledger: Ledger,
    paper: dict[str, Any],
    selected_questions: list[dict[str, Any]],
    reading_reviews: dict[str, dict[str, Any]],
    approved_cards: list[dict[str, Any]],
    *,
    drafter: Draft,
) -> dict[str, Any]:
    """Curate reading evidence into researcher-reviewable knowledge actions."""
    if not selected_questions:
        raise ValueError("Knowledge curation requires at least one selected reading question.")
    workflow = prepare_workflow_run(WORKFLOW_PATH, {
        "ledger": ledger,
        "paper": paper,
        "selected_questions": selected_questions,
        "reading_reviews": reading_reviews or {},
        "approved_cards": approved_cards or [],
        "drafter": drafter,
    }, globals())
    return workflow.execute()


def _prepare_knowledge_candidates(context: dict[str, Any]) -> None:
    paper = context["paper"]
    reviews = context.get("reading_reviews") or {}
    candidates: list[dict[str, Any]] = []
    for question in context["selected_questions"]:
        question_id = str(question.get("question_id", "")).strip()
        if not question_id:
            continue
        review = reviews.get(question_id) or {}
        claim = str(review.get("refined_answer") or question.get("tentative_answer") or "").strip()
        if len(claim) < 8:
            continue
        evidence = [str(item).strip() for item in (question.get("evidence") or []) if str(item).strip()]
        evidence.extend(
            str(item).strip() for item in (review.get("additional_evidence") or []) if str(item).strip()
        )
        evidence = list(dict.fromkeys(evidence))[:5]
        labels = _split_csv(question.get("suggested_labels"))
        concepts = _split_csv(question.get("suggested_concepts"))
        applies_to = _split_csv(question.get("suggested_applies_to"))
        conditions = str(question.get("suggested_conditions") or "").strip()
        limits = str(review.get("remaining_uncertainty") or question.get("suggested_limits") or question.get("uncertainty") or "").strip()
        source_excerpt = str(question.get("suggested_source_excerpt") or "").strip()
        card_id = f"kc-candidate-{uuid.uuid4().hex[:12]}"
        card = KnowledgeCard(
            card_id=card_id,
            title=str(question.get("suggested_title") or _compact_title(claim, limit=64)).strip(),
            source_kind="external_paper",
            claim=claim,
            context=str(question.get("suggested_context") or "").strip(),
            implication=str(question.get("suggested_implication") or "").strip(),
            source_excerpt=source_excerpt[:3200],
            labels=labels,
            concepts=concepts,
            applies_to=applies_to,
            evidence_excerpt="; ".join(evidence)[:1600],
            evidence_pages=[],
            citation_markers=[],
            conditions=conditions,
            limits=limits,
            provenance={
                "source_name": str(paper.get("title") or paper.get("paper_id") or "paper"),
                "paper_id": str(paper.get("paper_id") or ""),
                "question_id": question_id,
            },
        ).model_dump(mode="json")
        candidates.append({
            "candidate_id": card_id,
            "question_id": question_id,
            "card": card,
        })
    context["candidates"] = candidates
    context["similar_cards"] = similar_approved_cards(
        [item["card"] for item in candidates], context.get("approved_cards") or []
    )


def _assess_knowledge_candidates(context: dict[str, Any]) -> None:
    candidates = context.get("candidates") or []
    similar_cards = context.get("similar_cards") or {}
    automatic = [
        {
            "candidate_id": item["candidate_id"],
            "decision": "new",
            "target_card_id": "",
            "reason": "No sufficiently similar approved card was found.",
        }
        for item in candidates
        if item["candidate_id"] not in similar_cards
    ]
    ambiguous = [item for item in candidates if item["candidate_id"] in similar_cards]
    if not ambiguous:
        context["assessments"] = automatic
        context["assessment_succeeded"] = True
        context["assessment_failed"] = False
        context["validation_error"] = ""
        context["draft_result"] = None
        return

    prompt = _curation_assessment_prompt(ambiguous, similar_cards)
    result = context["drafter"](prompt)
    context["draft_result"] = result
    if isinstance(result, str):
        ok, text = bool(result.strip()), result
    else:
        ok = bool(getattr(result, "ok", False))
        text = str(getattr(result, "text", "") or "")
    if not ok:
        context["assessments"] = automatic
        context["assessment_succeeded"] = False
        context["assessment_failed"] = True
        context["validation_error"] = "Knowledge-fit assessment did not return a usable result."
        return

    parsed = _parse_curation_assessments(text, ambiguous, similar_cards)
    expected = {item["candidate_id"] for item in ambiguous}
    observed = {item["candidate_id"] for item in parsed}
    missing = sorted(expected - observed)
    if missing:
        context["assessments"] = automatic + parsed
        context["assessment_succeeded"] = False
        context["assessment_failed"] = True
        context["validation_error"] = f"Missing curation assessments: {', '.join(missing)}"
        return
    context["assessments"] = automatic + parsed
    context["assessment_succeeded"] = True
    context["assessment_failed"] = False
    context["validation_error"] = ""


def _submit_curation_decisions(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    paper = context["paper"]
    candidate_by_id = {item["candidate_id"]: item for item in context.get("candidates") or []}
    case_id = ledger.create_case("research", f"Knowledge curation: {paper.get('title') or paper.get('paper_id')}")
    request_ids: list[str] = []
    held: list[dict[str, Any]] = []
    action_counts = {"new": 0, "reinforce": 0, "conflict": 0, "hold": 0}

    for assessment in context.get("assessments") or []:
        candidate = candidate_by_id.get(str(assessment.get("candidate_id", "")))
        if candidate is None:
            continue
        decision = str(assessment.get("decision", "hold"))
        action_counts[decision] = action_counts.get(decision, 0) + 1
        if decision == "hold":
            held.append(assessment)
            continue

        card = dict(candidate["card"])
        target_card_id = str(assessment.get("target_card_id", "")).strip()
        if decision == "conflict":
            card["status"] = "contested"

        if decision == "reinforce":
            evidence_update = {
                "target_card_id": target_card_id,
                "source_name": str(paper.get("title") or paper.get("paper_id") or "paper"),
                "evidence_excerpt": str(card.get("evidence_excerpt", "")),
                "source_excerpt": str(card.get("source_excerpt", "")),
                "conditions": str(card.get("conditions", "")),
                "limits": str(card.get("limits", "")),
                "origin_links": list(card.get("origin_links", [])),
                "candidate_claim": str(card.get("claim", "")),
                "question_id": candidate["question_id"],
            }
            subject_id = f"kevidence-{uuid.uuid4().hex[:12]}"
            request_id = ledger.record(
                case_id, "decision_request", "m1", ["researcher"], "knowledge_evidence",
                {
                    "title": f"Knowledge evidence approval: {card['title']}",
                    "evidence_update": evidence_update,
                    "next_action": "승인 시 기존 지식카드에 supporting evidence를 추가하고 M2에 knowledge update를 통지합니다.",
                    "curation_action": decision,
                    "curation_reason": assessment.get("reason", ""),
                },
                subject_id=subject_id,
            )
        else:
            request_id = ledger.record(
                case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
                {
                    "title": f"Knowledge card approval: {card['title']}",
                    "card": card,
                    "next_action": "승인 시 semantic memory에 저장하고 M2에 knowledge update를 통지합니다.",
                    "curation_action": decision,
                    "curation_reason": assessment.get("reason", ""),
                    "related_card_ids": [target_card_id] if target_card_id else [],
                },
                subject_id=str(card["card_id"]),
            )
        request_ids.append(request_id)
        ledger.update_paper_reading_question(
            candidate["question_id"], researcher_comment="", status="promoted", promotion_request_id=request_id,
        )

    context["request_ids"] = request_ids
    context["held_assessments"] = held
    context["action_counts"] = action_counts


def _curation_assessment_prompt(candidates: list[dict[str, Any]], similar_cards: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "You are M1, curating paper-derived knowledge against already approved knowledge cards.",
        "For every candidate below, choose exactly one decision: new, reinforce, conflict, hold.",
        "reinforce means the candidate adds supporting evidence without materially changing the approved claim.",
        "conflict means the paper-derived claim materially contradicts or contests an approved claim.",
        "new means it is materially distinct. hold means evidence or comparison is insufficient.",
        "Use only the candidate and the listed similar approved cards. Do not invent a target.",
        "Return one block per candidate separated by --- using exactly these fields:",
        "Candidate: <candidate id>",
        "Decision: new|reinforce|conflict|hold",
        "Target: <approved card id or NONE>",
        "Reason: <brief reason>",
        "",
    ]
    for item in candidates:
        card = item["card"]
        lines.extend([
            f"Candidate {item['candidate_id']}",
            f"Claim: {card.get('claim', '')}",
            f"Evidence: {card.get('evidence_excerpt', '')}",
            "Similar approved cards:",
        ])
        for match in similar_cards.get(item["candidate_id"], []):
            lines.append(
                f"- {match['card_id']} | {match.get('title', '')} | {match.get('claim', '')} | similarity={match.get('score', '')}"
            )
        lines.append("")
    return "\n".join(lines)


def _parse_curation_assessments(
    text: str,
    candidates: list[dict[str, Any]],
    similar_cards: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    candidate_ids = {item["candidate_id"] for item in candidates}
    allowed_targets = {
        candidate_id: {str(match.get("card_id", "")) for match in matches}
        for candidate_id, matches in similar_cards.items()
    }
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in re.split(r"(?m)^---+\s*$", text):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().casefold()] = value.strip()
        candidate_id = fields.get("candidate", "")
        decision = fields.get("decision", "").casefold()
        target = fields.get("target", "")
        reason = fields.get("reason", "")
        if candidate_id not in candidate_ids or candidate_id in seen:
            continue
        if decision not in {"new", "reinforce", "conflict", "hold"} or not reason:
            continue
        if target.upper() == "NONE":
            target = ""
        if decision in {"reinforce", "conflict"} and target not in allowed_targets.get(candidate_id, set()):
            continue
        if decision in {"new", "hold"}:
            target = ""
        result.append({
            "candidate_id": candidate_id,
            "decision": decision,
            "target_card_id": target,
            "reason": reason,
        })
        seen.add(candidate_id)
    return result


def _split_csv(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value or "").split(",")
    return [str(item).strip() for item in raw if str(item).strip()][:10]
