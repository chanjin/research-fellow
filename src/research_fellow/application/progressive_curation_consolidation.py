"""Candidate comparison and consolidation capabilities for progressive curation."""
from __future__ import annotations
import re
from dataclasses import replace
from itertools import combinations
from research_fellow.application.progressive_curation_models import ProgressiveCandidate, ConsolidationDecision

def candidate_pairs(candidates: list[ProgressiveCandidate], limit: int = 18) -> list[dict[str, object]]:
    """Python narrows only plausible pairs; the LLM decides their semantic relation."""
    ranked = []
    for first, second in combinations(candidates, 2):
        first_terms, second_terms = _terms(first), _terms(second)
        shared = first_terms & second_terms
        distance = abs(first.page_number - second.page_number)
        if not shared and distance > 1:
            continue
        score = len(shared) * 10 + max(0, 2 - distance)
        if score:
            ranked.append((score, first, second, sorted(shared)[:6]))
    pairs = []
    for index, (_, first, second, shared) in enumerate(sorted(ranked, key=lambda item: item[0], reverse=True)[:limit], start=1):
        pairs.append({
            "pair_id": f"pair-{index:03d}", "selection_reason": f"공유 용어: {', '.join(shared) or '인접 페이지'}",
            "first": _candidate_view(first), "second": _candidate_view(second),
        })
    return pairs


def consolidation_prompt(pairs: list[dict[str, object]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_candidate_consolidation.j2", candidate_pairs=pairs)


def consolidate_candidates(candidates: list[ProgressiveCandidate], llm_text: str | None) -> tuple[list[ProgressiveCandidate], list[ConsolidationDecision], list[str]]:
    """Apply only validated LLM relation decisions; independent cards stay intact."""
    decisions = _parse_consolidation(llm_text or "", {item.candidate_id for item in candidates})
    by_id = {item.candidate_id: item for item in candidates}
    removed: set[str] = set()
    warnings = []
    for decision in decisions:
        if decision.action == "keep_first":
            removed.add(decision.second_id)
        elif decision.action == "keep_second":
            removed.add(decision.first_id)
        elif decision.action in {"merge_into_first", "merge_into_second"}:
            keep, drop = (decision.first_id, decision.second_id) if decision.action == "merge_into_first" else (decision.second_id, decision.first_id)
            by_id[keep] = _merge_evidence(by_id[keep], by_id[drop], decision.reason)
            removed.add(drop)
        elif decision.action == "flag_for_researcher":
            warnings.append(f"{decision.first_id} / {decision.second_id}: {decision.reason}")
    kept = [by_id[item.candidate_id] for item in candidates if item.candidate_id not in removed]
    return kept, decisions, warnings


def _terms(candidate: ProgressiveCandidate) -> set[str]:
    card = candidate.card
    text = " ".join([str(card.get("title", "")), str(card.get("claim", "")), " ".join(card.get("labels", []))])
    return {term.lower() for term in re.findall(r"[\w가-힣]{2,}", text)}


def _candidate_view(candidate: ProgressiveCandidate) -> dict[str, object]:
    card = candidate.card
    return {
        "candidate_id": candidate.candidate_id, "title": card["title"], "claim": card["claim"],
        "evidence_excerpt": card["evidence_excerpt"],
        "labels": card.get("labels", []), "conditions": card["conditions"], "limits": card["limits"],
    }


def _parse_consolidation(text: str, valid_ids: set[str]) -> list[ConsolidationDecision]:
    aliases = {"후보": "candidates", "candidates": "candidates", "관계": "relation", "relation": "relation",
               "처리": "action", "action": "action", "이유": "reason", "reason": "reason"}
    result = []
    for block in [part.strip() for part in re.split(r"(?m)^---+\s*$", text) if part.strip()]:
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if normalized := aliases.get(key.strip().lower()):
                    fields[normalized] = value.strip()
        ids = [item.strip() for item in fields.get("candidates", "").split(",")]
        relation, action, reason = fields.get("relation", ""), fields.get("action", ""), fields.get("reason", "")
        if len(ids) != 2 or ids[0] == ids[1] or not set(ids).issubset(valid_ids):
            continue
        if relation not in {"independent", "duplicate", "contains", "contained_by", "partial_overlap", "related_but_distinct"}:
            continue
        if action not in {"retain_both", "keep_first", "keep_second", "merge_into_first", "merge_into_second", "flag_for_researcher"}:
            continue
        if not reason:
            continue
        result.append(ConsolidationDecision(ids[0], ids[1], relation, action, reason))
    return result


def _merge_evidence(kept: ProgressiveCandidate, dropped: ProgressiveCandidate, reason: str) -> ProgressiveCandidate:
    card = dict(kept.card)
    dropped_card = dropped.card
    card["evidence_pages"] = []
    card["citation_markers"] = []
    card["evidence_excerpt"] = f"{card['evidence_excerpt']}\n\n{dropped_card['evidence_excerpt']}"[:1600]
    return replace(kept, card=card, warnings=[*kept.warnings, f"{dropped.candidate_id}와 병합: {reason}"])
