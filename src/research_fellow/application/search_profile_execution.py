"""External search execution and scheduling for approved search profiles."""
from __future__ import annotations
from typing import Any, Callable
from research_fellow.infrastructure.arxiv import ArxivError, search as arxiv_search
from research_fellow.infrastructure.semantic_scholar import enrich_citation_counts
from research_fellow.origin_lineage import normalize_origin_links
from research_fellow.storage import Ledger
from research_fellow.application.search_profile_strategy import query_ladder, shortlist_candidates
def search_profile_candidates(profile: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Run only external search + citation enrichment; no LLM triage."""
    candidates, attempted = [], []
    for label, expression in query_ladder(profile):
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
    candidates = enrich_citation_counts(candidates[:100])
    return " → ".join(attempted), candidates


def run_profile(ledger: Ledger, profile: dict[str, Any], trigger: str, reviewer: Callable[[str], str | None] | None = None) -> dict[str, Any]:
    query = " AND ".join(profile.get("keywords", []))
    try:
        query, candidates = search_profile_candidates(profile)
        candidates = shortlist_candidates(profile, candidates, reviewer)
        origin_links = normalize_origin_links(profile.get("origin_links", []))
        candidates = [{**candidate, "origin_links": origin_links} for candidate in candidates]
        status = "completed" if candidates else "completed_no_candidates"
        run_id = ledger.record_search_run(profile["profile_id"], trigger, query, candidates, status)
        # A run completes; a periodic profile does not. Daily/weekly profiles stay
        # active for the scheduler, while an explicitly manual one-shot profile closes.
        if profile.get("cadence") == "manual":
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
        if profile["cadence"] == "daily" and (not profile["last_run_at"] or (now - datetime.fromisoformat(profile["last_run_at"])).days >= 1):
            due.append(profile)
        elif profile["cadence"] == "weekly" and (not profile["last_run_at"] or (now - datetime.fromisoformat(profile["last_run_at"])).days >= 7):
            due.append(profile)
    return due
