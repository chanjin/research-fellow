"""Intent-scoped literature discovery capabilities."""
from __future__ import annotations
import json
from typing import Any, Callable
from research_fellow.origin_lineage import normalize_origin_links
from research_fellow.storage import Ledger
def intent_discovery_task_prompt(profile: dict[str, Any], previous_runs: list[dict[str, Any]], max_results: int = 12) -> str:
    """Build the editable LLM task used for every periodic Intent run."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt("m1_intent_discovery_task.j2", profile=profile, previous_runs=previous_runs[:5], max_results=max_results)


def run_intent_discovery_plan(
    ledger: Ledger, profile: dict[str, Any], plan: dict[str, Any], *, trigger: str,
    triage_drafter: Callable[[str], str | None], sources: list[str] | None = None, max_results: int = 12,
) -> dict[str, Any]:
    """Execute a validated LLM plan through retrieval, triage, and durable Run history."""
    from research_fellow.application.literature_discovery_sources import collect_multisource_candidates
    from research_fellow.application.literature_discovery_prompts import discovery_triage_prompt
    from research_fellow.application.literature_discovery_parsers import apply_discovery_triage
    sources = sources or ["arxiv", "semantic_scholar", "crossref"]
    queries = [str(value).strip() for value in plan.get("queries", []) if str(value).strip()][:8]
    if not queries:
        raise ValueError("LLM 탐색 계획에 실행 가능한 queries가 없습니다.")
    topic = str(profile.get("question") or profile.get("title") or "")
    context = str(profile.get("context") or "")
    try:
        candidates = collect_multisource_candidates(topic, context, queries, sources, max_results=max_results)
        triage_raw = triage_drafter(discovery_triage_prompt(topic, context, candidates, max_results)) or ""
        results = apply_discovery_triage(candidates, triage_raw, max_results) if candidates else []
        origin_links = normalize_origin_links(profile.get("origin_links", []))
        results = [{**candidate, "origin_links": origin_links} for candidate in results]
        status = "completed" if results else "completed_no_candidates"
        query_log = json.dumps({
            **plan, "sources": sources, "topic": topic, "research_context": context,
            "origin_links": origin_links,
        }, ensure_ascii=False)
        run_id = ledger.record_search_run(str(profile["profile_id"]), trigger, query_log, results, status)
        return {"run_id": run_id, "query": query_log, "candidates": results, "status": status, "error": "", "plan": plan}
    except Exception as error:
        query_log = json.dumps({
            **plan, "sources": sources, "topic": topic, "research_context": context,
            "origin_links": normalize_origin_links(profile.get("origin_links", [])),
        }, ensure_ascii=False)
        run_id = ledger.record_search_run(str(profile["profile_id"]), trigger, query_log, [], "failed", str(error))
        return {"run_id": run_id, "query": query_log, "candidates": [], "status": "failed", "error": str(error), "plan": plan}


def run_intent_discovery_task(
    ledger: Ledger, profile: dict[str, Any], *, trigger: str,
    planner: Callable[[str], str | None], triage_drafter: Callable[[str], str | None],
    sources: list[str] | None = None, max_results: int = 12,
) -> dict[str, Any]:
    from research_fellow.application.literature_discovery_parsers import parse_discovery_search_plan
    previous_runs = ledger.search_runs(str(profile["profile_id"]), limit=5)
    prompt = intent_discovery_task_prompt(profile, previous_runs, max_results)
    plan = parse_discovery_search_plan(planner(prompt) or "")
    return run_intent_discovery_plan(ledger, profile, plan, trigger=trigger, triage_drafter=triage_drafter, sources=sources, max_results=max_results)


def record_external_intent_discovery(
    ledger: Ledger,
    profile: dict[str, Any],
    response: str,
    *,
    trigger: str = "external_llm_task",
    max_results: int = 12,
) -> dict[str, Any]:
    """Validate and record papers found directly by a web-enabled external LLM.

    Intent discovery deliberately treats this as the primary route: the LLM
    searches and organises literature in one task.  The API-backed search-plan
    route remains available as an explicit fallback.
    """
    from research_fellow.application.literature_discovery_parsers import parse_external_literature_results

    topic = str(profile.get("question") or profile.get("title") or "").strip()
    context = str(profile.get("context") or profile.get("research_context") or "").strip()
    parsed = parse_external_literature_results(response, max_results)
    origin_links = normalize_origin_links(profile.get("origin_links", []))
    results = [{**paper, "origin_links": origin_links} for paper in parsed["papers"]]
    status = "completed" if results else "completed_no_candidates"
    query_log = json.dumps(
        {
            "mode": "external_llm_direct_discovery",
            "topic": topic,
            "research_context": context,
            "search_summary": parsed.get("search_summary", ""),
            "sources": ["external_llm_web_search"],
            "origin_links": origin_links,
        },
        ensure_ascii=False,
    )
    run_id = ledger.record_search_run(str(profile["profile_id"]), trigger, query_log, results, status)
    return {
        "run_id": run_id,
        "query": query_log,
        "candidates": results,
        "status": status,
        "error": "",
        "search_summary": parsed.get("search_summary", ""),
    }
