"""Small-context, multi-call M1 curation with LLM-led duplicate review."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Callable

from research_fellow.infrastructure.candidate_draft_cache import CandidateDraftCache
from research_fellow.infrastructure.document_reader import ExtractedDocument, ExtractedPage
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class ProgressiveCandidate:
    candidate_id: str
    card: dict[str, object]
    page_number: int
    block_ids: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class ConsolidationDecision:
    first_id: str
    second_id: str
    relation: str
    action: str
    reason: str


def page_curation_prompt(document_title: str, source_kind: str, page: ExtractedPage, max_cards: int = 2) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m1_page_curation.j2", document_title=document_title, source_kind=source_kind, page=page, max_cards=max_cards)


def generate_page_candidates(
    document: ExtractedDocument, source_kind: str, labels: list[str], model: str,
    draft_for: Callable[[str], str | None], cache: CandidateDraftCache, pages: list[ExtractedPage] | None = None,
    on_page: Callable[[int, int, bool], None] | None = None,
) -> tuple[list[ProgressiveCandidate], list[str]]:
    """Call an LLM once per page, reusing cached non-authoritative drafts."""
    from research_fellow.infrastructure.prompt_templates import read_prompt_template
    from research_fellow.application.curation import normalize_candidate_draft

    template_source = read_prompt_template("m1_page_curation.j2") + f"\0{source_kind}\0{','.join(labels)}"
    selected_pages = pages or document.pages
    candidates: list[ProgressiveCandidate] = []
    warnings: list[str] = []
    for ordinal, page in enumerate(selected_pages, start=1):
        draft = cache.get(document_id=document.document_id, page_number=page.page_number, model=model, template_source=template_source)
        cache_hit = draft is not None
        if draft is None:
            draft = draft_for(page_curation_prompt(document.title, source_kind, page))
            if draft:
                cache.put(document_id=document.document_id, page_number=page.page_number, model=model, template_source=template_source, draft=draft)
        if on_page:
            on_page(ordinal, len(selected_pages), cache_hit)
        if not draft:
            warnings.append(f"p.{page.page_number}: LLM 초안을 만들지 못해 이 페이지를 건너뛰었습니다.")
            continue
        blocks = [part.strip() for part in re.split(r"(?m)^---+\s*$", draft) if part.strip()][:2]
        for index, block in enumerate(blocks, start=1):
            result = normalize_candidate_draft(
                title=document.title, source_kind=source_kind, page=page, labels=labels, index=index, text_draft=block,
            )
            candidate_id = f"kc-candidate-{document.document_id.rsplit('-', 1)[-1]}-p{page.page_number:03d}-{index}"
            card = {**result.card, "card_id": candidate_id}
            candidates.append(ProgressiveCandidate(
                candidate_id=candidate_id, card=card, page_number=page.page_number,
                block_ids=[block.block_id for block in page.blocks], warnings=result.warnings,
            ))
            warnings.extend(f"p.{page.page_number} 후보 {index}: {warning}" for warning in result.warnings)
    return candidates, warnings


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


def submit_progressive_candidates(
    ledger: Ledger, document: ExtractedDocument, candidates: list[ProgressiveCandidate], decisions: list[ConsolidationDecision], warnings: list[str],
) -> list[str]:
    """Create researcher decision requests only; no candidate enters JSONL here."""
    case_id = ledger.create_case("research", f"점진적 지식화: {document.title}")
    request_ids = []
    for candidate in candidates:
        request_ids.append(ledger.record(
            case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
            {
                "title": f"지식 카드 승인: {candidate.card['title']}", "card": candidate.card,
                "normalization": "progressive_page_curation", "warnings": candidate.warnings,
                "page_number": candidate.page_number, "block_ids": candidate.block_ids,
                "next_action": "승인 시 의미 기억에 저장하고 M2에 통지; 보류 시 보완을 요청",
            },
            subject_id=candidate.candidate_id,
        ))
    return request_ids


def _terms(candidate: ProgressiveCandidate) -> set[str]:
    card = candidate.card
    text = " ".join([str(card.get("title", "")), str(card.get("claim", "")), " ".join(card.get("labels", []))])
    return {term.lower() for term in re.findall(r"[\w가-힣]{2,}", text)}


def _candidate_view(candidate: ProgressiveCandidate) -> dict[str, object]:
    card = candidate.card
    return {
        "candidate_id": candidate.candidate_id, "title": card["title"], "claim": card["claim"],
        "evidence_excerpt": card["evidence_excerpt"], "evidence_pages": card["evidence_pages"],
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
    card["evidence_pages"] = sorted(set(card["evidence_pages"]) | set(dropped_card["evidence_pages"]))
    card["citation_markers"] = sorted(set(card["citation_markers"]) | set(dropped_card["citation_markers"]))
    card["evidence_excerpt"] = f"{card['evidence_excerpt']}\n\n{dropped_card['evidence_excerpt']}"[:1600]
    return replace(kept, card=card, warnings=[*kept.warnings, f"{dropped.candidate_id}와 병합: {reason}"])
