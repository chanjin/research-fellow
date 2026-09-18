"""Jinja prompt entry points for future M1/M2 review tasks.

The functions in this module deliberately create small, reference-addressable
contexts.  They do not create IDs, write memory, or transition phenomena:
those remain deterministic application-service responsibilities.
"""

from __future__ import annotations

from typing import Any

from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.origin_lineage import origin_labels


def _card_view(card: dict[str, Any], reference: str) -> dict[str, Any]:
    provenance = card.get("provenance", {})
    return {
        "reference": reference,
        "card_id": str(card.get("card_id", "")),
        "title": str(card.get("title", "")),
        "claim": str(card.get("claim", "")),
        "evidence_excerpt": str(card.get("evidence_excerpt", "")),
        "labels": list(card.get("labels", [])),
        "conditions": str(card.get("conditions", "")),
        "limits": str(card.get("limits", "")),
        "source_name": str(provenance.get("source_name", "미상")),
        "origin_labels": origin_labels(card.get("origin_links", [])),
    }


def evidence_views(cards: list[dict[str, Any]], prefix: str = "E", limit: int = 12) -> list[dict[str, Any]]:
    """Provide only approved-card facts, each with an LLM-safe local reference."""
    return [_card_view(card, f"{prefix}{index}") for index, card in enumerate(cards[:limit], start=1)]


def claim_verification_prompt(assertion: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_claim_verification.j2", assertion=assertion, evidence=evidence_views(cards))


def gap_search_plan_prompt(gap: str, focus: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_gap_search_plan.j2", knowledge_gap=gap, focus=focus, existing_evidence=evidence_views(cards))


def source_triage_prompt(gap: str, sources: list[dict[str, Any]]) -> str:
    """Render only sources that Python has retrieved from an approved tool."""
    candidates = [
        {
            "reference": f"S{index}", "source_id": str(source.get("source_id", "")),
            "title": str(source.get("title", "")), "authors": list(source.get("authors", [])),
            "published": str(source.get("published", "")), "summary": str(source.get("summary", "")),
            "url": str(source.get("url", "")),
        }
        for index, source in enumerate(sources[:15], start=1)
    ]
    return render_prompt("m1_source_triage.j2", knowledge_gap=gap, source_candidates=candidates)


def lineage_review_prompt(topic: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_lineage_review.j2", topic=topic, knowledge=evidence_views(cards, prefix="K", limit=10))


def revalidation_review_prompt(reason: str, cards: list[dict[str, Any]]) -> str:
    return render_prompt("m1_revalidation_review.j2", reason=reason, evidence=evidence_views(cards))


def knowledge_update_report_prompt(cards: list[dict[str, Any]], research_question: str = "") -> str:
    return render_prompt(
        "m2_knowledge_update_report.j2", research_question=research_question, evidence=evidence_views(cards)
    )


def research_question_suggestions_prompt(
    updates: list[dict[str, Any]], existing_questions: list[str], cards: list[dict[str, Any]], *, max_suggestions: int = 10,
) -> str:
    """Give M2 the actual updated card content, not only ledger event titles."""
    cards_by_id = {str(card.get("card_id", "")): card for card in cards}
    update_views: list[dict[str, Any]] = []
    for index, update in enumerate(updates, start=1):
        payload = update.get("payload", {}) if isinstance(update.get("payload"), dict) else {}
        card = cards_by_id.get(str(payload.get("card_id", "")))
        if card:
            card_view = _card_view(card, f"K{index}")
            update_views.append({
                **card_view,
                "concepts": list(card.get("concepts", [])),
                "applies_to": list(card.get("applies_to", [])),
                "evidence_level": str(card.get("evidence_level", "")),
            })
        else:
            # Legacy or non-card M1 updates remain visible, but do not pretend
            # that title-only events are full evidence-bearing knowledge cards.
            update_views.append({
                "reference": f"U{index}", "card_id": "", "title": str(payload.get("title", "M1 새 정보")),
                "claim": str(payload.get("finding", "")), "evidence_excerpt": "", "labels": [],
                "concepts": [], "applies_to": [], "conditions": "", "limits": "",
                "evidence_level": "", "source_name": "",
            })
    return render_prompt(
        "m2_research_question_suggestions.j2",
        updates=update_views,
        existing_questions=existing_questions,
        max_suggestions=max(1, min(max_suggestions, 10)),
    )


def knowledge_grouping_prompt(
    updates: list[dict[str, Any]], cards: list[dict[str, Any]], *, max_groups: int = 6,
) -> str:
    """Ask M2 to propose research-theme groups before the researcher selects cards."""
    cards_by_id = {str(card.get("card_id", "")): card for card in cards}
    items: list[dict[str, Any]] = []
    for index, update in enumerate(updates, start=1):
        payload = update.get("payload", {}) if isinstance(update.get("payload"), dict) else {}
        card = cards_by_id.get(str(payload.get("card_id", "")))
        if not card:
            continue
        view = _card_view(card, f"K{index}")
        items.append({
            **view,
            "context": str(card.get("context", "")),
            "implication": str(card.get("implication", "")),
            "concepts": list(card.get("concepts", [])),
            "applies_to": list(card.get("applies_to", [])),
        })
    return f"""You are helping a researcher organize newly approved knowledge before formulating research questions.
Group the knowledge cards by research-theme relevance, not merely lexical or embedding similarity.
A useful group should represent a coherent research issue, tension, comparison, mechanism, or design question that could reasonably lead to one research question.
Cards may appear in more than one group when they genuinely support different research angles. Do not force every card into a group.

Return JSON only with this schema:
{{
  "groups": [
    {{
      "name": "short research-theme name",
      "research_focus": "what common research issue connects these cards",
      "why_together": "why these cards should be interpreted together rather than only because they are similar",
      "research_question_alternatives": [
        {{
          "title": "short angle label",
          "question": "one focused research question",
          "rationale": "why this question is worth asking now from these cards",
          "research_context": "the concrete research context that should not be lost",
          "gap_or_tension": "the unresolved gap, contrast, or tension",
          "exploration_need": "what should be checked next if deeper exploration is needed"
        }}
      ],
      "card_ids": ["exact card_id", "..."]
    }}
  ]
}}

Rules:
- Return at most {max(1, min(max_groups, 8))} groups.
- For each group, return 2-4 genuinely different research-question alternatives (prefer 3).
- Alternatives must differ by research angle, not by superficial wording.
- Use only card_ids shown below.
- Prefer groups of 2 or more cards, but allow a strong singleton when it exposes an important independent research direction.
- Use source paper, context, implication, conditions, and limits when deciding why cards belong together.
- These are researcher-editable theme/question alternatives, not approved research questions.

New knowledge cards:
{items}
"""


def parse_knowledge_grouping(text: str, *, valid_card_ids: set[str], limit: int = 8) -> list[dict[str, Any]]:
    """Parse advisory grouping JSON defensively and keep only known card IDs."""
    import json
    import re

    raw = (text or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    groups = payload.get("groups", []) if isinstance(payload, dict) else []
    results: list[dict[str, Any]] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        ids = [str(cid) for cid in item.get("card_ids", []) if str(cid) in valid_card_ids]
        ids = list(dict.fromkeys(ids))
        name = str(item.get("name", "")).strip()
        if not ids or not name:
            continue
        alternatives: list[dict[str, str]] = []
        raw_alternatives = item.get("research_question_alternatives", [])
        if isinstance(raw_alternatives, list):
            for alt in raw_alternatives[:4]:
                if not isinstance(alt, dict):
                    continue
                question = str(alt.get("question", "")).strip()
                if not question:
                    continue
                alternatives.append({
                    "title": str(alt.get("title", "")).strip()[:120],
                    "question": question,
                    "rationale": str(alt.get("rationale", "")).strip(),
                    "research_context": str(alt.get("research_context", "")).strip(),
                    "gap_or_tension": str(alt.get("gap_or_tension", "")).strip(),
                    "exploration_need": str(alt.get("exploration_need", "")).strip(),
                })
        legacy_question = str(item.get("candidate_question", "")).strip()
        if not alternatives and legacy_question:
            alternatives.append({
                "title": "",
                "question": legacy_question,
                "rationale": str(item.get("why_together", "")).strip(),
                "research_context": str(item.get("research_focus", "")).strip(),
                "gap_or_tension": "",
                "exploration_need": "",
            })
        results.append({
            "name": name[:140],
            "research_focus": str(item.get("research_focus", "")).strip(),
            "why_together": str(item.get("why_together", "")).strip(),
            "research_question_alternatives": alternatives,
            "candidate_question": alternatives[0]["question"] if alternatives else legacy_question,
            "card_ids": ids,
        })
        if len(results) >= max(1, limit):
            break
    return results
