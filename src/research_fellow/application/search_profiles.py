"""Editable, researcher-approved M1 literature-search profiles."""

from __future__ import annotations

import re
from typing import Any, Callable

from research_fellow.infrastructure.arxiv import ArxivError, search as arxiv_search
from research_fellow.storage import Ledger


def keyword_prompt(profile: dict[str, Any]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_search_keywords.j2", profile=profile)


def abstract_relevance_prompt(profile: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_abstract_relevance.j2", profile=profile, candidates=candidates)


STOP_WORDS = {
    "a", "an", "and", "as", "at", "based", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "with",
}


def parse_keywords(text: str) -> list[str]:
    keywords = []
    for line in text.splitlines():
        value = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if is_english_search_term(value) and value not in keywords:
            keywords.append(value)
    return keywords[:8]


def parse_keyword_plan(text: str) -> tuple[list[str], list[str]]:
    """Read M2's explicit phrase/term plan, with a safe fallback for old prompts."""
    section = "phrases"
    phrases: list[str] = []
    core_terms: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        marker = line.rstrip(":").upper()
        if marker in {"PRIORITY_PHRASES", "PRIORITY PHRASES"}:
            section = "phrases"
            continue
        if marker in {"CORE_TERMS", "CORE TERMS"}:
            section = "terms"
            continue
        value = re.sub(r"^\\s*(?:[-*•]|\\d+[.)])\\s*", "", line).strip().strip('"')
        if not is_english_search_term(value):
            continue
        target = phrases if section == "phrases" else core_terms
        if value.lower() not in {item.lower() for item in target}:
            target.append(value)
    phrases = phrases[:8]
    core_terms = _clean_core_terms(core_terms)
    return (phrases or parse_keywords(text), core_terms)


def query_for(profile: dict[str, Any]) -> str:
    if invalid := [keyword for keyword in profile["keywords"] if not is_english_search_term(keyword)]:
        raise ValueError(f"영문 검색어가 아닌 키워드가 있습니다: {', '.join(invalid[:3])}")
    if not profile["keywords"]:
        raise ValueError("저장된 영문 탐색 키워드가 없습니다.")
    return _phrase_expression(profile["keywords"][0])


def query_ladder(profile: dict[str, Any]) -> list[tuple[str, str]]:
    """Build independent recall queries; never AND mutually alternative phrases.

    Search phrases are ordered by researcher/M2 priority, but every phrase is
    executed separately.  The final two queries expose both a transparent
    all-term intersection and a compact, LLM-selected five-term intersection.
    """
    keywords = profile["keywords"]
    if invalid := [keyword for keyword in keywords if not is_english_search_term(keyword)]:
        raise ValueError(f"영문 검색어가 아닌 키워드가 있습니다: {', '.join(invalid[:3])}")
    if not keywords:
        raise ValueError("저장된 영문 탐색 키워드가 없습니다.")
    plan = [(f"우선순위 {index} · 정확 구문", _phrase_expression(keyword)) for index, keyword in enumerate(keywords, start=1)]
    all_terms = _unique_content_terms(keywords)
    if all_terms:
        plan.append(("전체 구문 중복 제거어 AND", " AND ".join(f"all:{term}" for term in all_terms)))
    core_terms = _clean_core_terms(profile.get("core_terms", [])) or _derive_core_terms(keywords)
    if core_terms:
        plan.append(("LLM 선정 핵심 5개어 AND", " AND ".join(f"all:{term}" for term in core_terms)))
    return plan


def _arxiv_expression(keywords: list[str]) -> str:
    return " AND ".join(_keyword_expression(keyword) for keyword in keywords)


def _keyword_expression(keyword: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", keyword)
    if not tokens:
        raise ValueError(f"arXiv 검색식으로 만들 수 없는 키워드입니다: {keyword}")
    terms = [f"all:{token}" for token in tokens]
    return terms[0] if len(terms) == 1 else "(" + " AND ".join(terms) + ")"


def _phrase_expression(keyword: str) -> str:
    phrase = " ".join(re.findall(r"[A-Za-z0-9]+", keyword))
    if not phrase:
        raise ValueError(f"arXiv 검색식으로 만들 수 없는 키워드입니다: {keyword}")
    return f'all:"{phrase}"'


def _unique_content_terms(keywords: list[str]) -> list[str]:
    terms: list[str] = []
    for keyword in keywords:
        for token in re.findall(r"[A-Za-z0-9]+", keyword):
            normalized = token.lower()
            if len(normalized) < 3 or normalized in STOP_WORDS or normalized in terms:
                continue
            terms.append(normalized)
    return terms


def _clean_core_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        for token in re.findall(r"[A-Za-z0-9]+", value):
            normalized = token.lower()
            if len(normalized) < 3 or normalized in STOP_WORDS or normalized in terms:
                continue
            terms.append(normalized)
            if len(terms) == 5:
                return terms
    return terms


def _derive_core_terms(keywords: list[str]) -> list[str]:
    return _unique_content_terms(keywords)[:5]


def is_english_search_term(value: str) -> bool:
    return bool(value.strip()) and not any("가" <= character <= "힣" for character in value) and all(
        character.isascii() and (character.isalnum() or character in " -_()/+.#") for character in value
    )


def attach_relevance(candidates: list[dict[str, Any]], draft: str | None) -> list[dict[str, Any]]:
    """Attach M2's abstract-only triage without upgrading a candidate to knowledge."""
    reviews: dict[str, dict[str, str]] = {}
    for line in (draft or "").splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or parts[1].lower() not in {"high", "medium", "low"}:
            continue
        reviews[parts[0]] = {"level": parts[1].lower(), "rationale": parts[2]}
    return [{
        **candidate,
        "relevance": reviews.get(candidate.get("source_id", ""), {"level": "unreviewed", "rationale": "초록 기반 맥락 적합성 검토가 아직 수행되지 않았습니다."}),
        "evidence_status": "abstract_only_pending",
    } for candidate in candidates]


def shortlist_candidates(
    profile: dict[str, Any], candidates: list[dict[str, Any]], reviewer: Callable[[str], str | None] | None,
) -> list[dict[str, Any]]:
    """Keep the first 100 results as a log, but spend M2 review on only five.

    The deterministic first pass prevents a large, slow LLM prompt.  The approved
    Intent's question and context are deliberately included, not merely keywords.
    """
    context = " ".join([profile.get("title", ""), profile.get("question", ""), profile.get("context", ""), *profile.get("keywords", [])])
    terms = {term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", context)}

    def lexical_score(candidate: dict[str, Any]) -> tuple[int, str]:
        text = f"{candidate.get('title', '')} {candidate.get('summary', '')}".lower()
        matched = [term for term in terms if term in text]
        # Title matches are more discriminative than an incidental abstract token.
        title = candidate.get("title", "").lower()
        return (sum(3 if term in title else 1 for term in matched), ", ".join(matched[:8]))

    # A 100-paper prompt is too large for a local model. Each batch returns only
    # compact ID/level/rationale lines, while still seeing the complete Intent.
    reviewed_by_id: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(candidates), 25):
        batch = candidates[offset : offset + 25]
        reviewed_by_id.update({item["source_id"]: item for item in attach_relevance(
            batch, reviewer(abstract_relevance_prompt(profile, batch)) if reviewer else None,
        )})
    order = {"high": 0, "medium": 1, "low": 2, "unreviewed": 3}
    ranked = sorted(
        candidates,
        key=lambda item: (order.get(reviewed_by_id[item["source_id"]]["relevance"]["level"], 3), -lexical_score(item)[0]),
    )
    shortlist_ids = {item["source_id"] for item in ranked[:5]}
    enriched = []
    for candidate in candidates:
        score, matched = lexical_score(candidate)
        if candidate["source_id"] in shortlist_ids:
            enriched.append({**reviewed_by_id[candidate["source_id"]], "abstract_shortlist": True, "context_match_score": score, "context_match_terms": matched})
        else:
            enriched.append({
                **candidate,
                "abstract_shortlist": False,
                "context_match_score": score,
                "context_match_terms": matched,
                "relevance": reviewed_by_id[candidate["source_id"]]["relevance"],
                "evidence_status": "abstract_only_pending",
            })
    return enriched


def run_profile(ledger: Ledger, profile: dict[str, Any], trigger: str, reviewer: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    query = " AND ".join(profile.get("keywords", []))
    try:
        candidates, attempted = [], []
        for label, expression in query_ladder(profile):
            # Every strategy is independently observable in the execution log.
            # A per-query cap preserves diversity before the final 100-paper cap.
            found = arxiv_search(expression, max_results=30)
            attempted.append(f"{label} [{len(found)}건]: {expression}")
            known_ids = {item["source_id"]: index for index, item in enumerate(candidates)}
            for candidate in found:
                if candidate["source_id"] in known_ids:
                    prior = candidates[known_ids[candidate["source_id"]]]
                    scopes = prior.setdefault("query_scopes", [prior.get("query_scope", "")])
                    if label not in scopes:
                        scopes.append(label)
                    continue
                candidates.append({**candidate, "query_scope": label, "query_scopes": [label]})
                known_ids[candidate["source_id"]] = len(candidates) - 1
        candidates = candidates[:100]
        query = " → ".join(attempted)
        candidates = shortlist_candidates(profile, candidates, reviewer)
        status = "completed" if candidates else "completed_no_candidates"
        run_id = ledger.record_search_run(profile["profile_id"], trigger, query, candidates, status)
        # An Intent represents one discovery task. Keep the profile and its
        # audit trail, but remove it from both the visible and scheduled queue.
        ledger.complete_search_profile(profile["profile_id"])
        return {"run_id": run_id, "query": query, "candidates": candidates, "status": status, "error": ""}
    except (ArxivError, ValueError) as error:
        run_id = ledger.record_search_run(profile["profile_id"], trigger, query, [], "failed", str(error))
        return {"run_id": run_id, "query": query, "candidates": [], "status": "failed", "error": str(error)}


def scheduled_profiles(ledger: Ledger) -> list[dict[str, Any]]:
    """The nightly runner consumes active daily profiles; weekly profiles wait seven days."""
    from datetime import datetime

    due = []
    now = datetime.now().astimezone()
    for profile in ledger.search_profiles(active_only=True):
        if profile["cadence"] == "daily":
            due.append(profile)
        elif profile["cadence"] == "weekly" and (not profile["last_run_at"] or (now - datetime.fromisoformat(profile["last_run_at"])).days >= 7):
            due.append(profile)
    return due
