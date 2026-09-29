"""Parsers and triage transforms for literature discovery."""
from __future__ import annotations
import hashlib
import re
from typing import Any
from research_fellow.application.literature_discovery_formats import _json_payload, build_paper_labels, normalize_result_url, arxiv_id_from_paper, paper_access_links
def parse_discovery_search_plan(text: str) -> dict[str, Any]:
    payload = _json_payload(text)
    if not isinstance(payload, dict):
        raise ValueError("검색전략 응답은 JSON object여야 합니다.")
    raw_queries = payload.get("queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries: list[str] = []
    for item in raw_queries:
        query = re.sub(r"\s+", " ", str(item or "").strip())
        if query and query not in queries:
            queries.append(query)
    if not queries:
        raise ValueError("사용 가능한 검색식이 없습니다.")
    notes = payload.get("search_notes", [])
    if not isinstance(notes, list):
        notes = []
    return {
        "scope_summary": str(payload.get("scope_summary", "")).strip(),
        "queries": queries[:5],
        "search_notes": [str(item).strip() for item in notes if str(item).strip()][:6],
    }


def apply_discovery_triage(candidates: list[dict[str, Any]], text: str, max_results: int) -> list[dict[str, Any]]:
    payload = _json_payload(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise ValueError("문헌 선별 응답에 papers 배열이 없습니다.")
    by_id = {str(item.get("source_id", "")): item for item in candidates}
    ranked: list[dict[str, Any]] = []
    for item in payload["papers"]:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        base = by_id.get(source_id)
        if not base:
            continue
        try:
            score = int(float(item.get("relevance_score", 0)))
        except (TypeError, ValueError):
            score = 0
        ranked.append(
            {
                **base,
                "relevance_score": max(0, min(score, 100)),
                "quick_take": str(item.get("quick_take", "")).strip(),
                "why_relevant": str(item.get("why_relevant", "")).strip(),
                "caution": str(item.get("caution", "")).strip(),
                "discovery_source": "internal",
            }
        )
    ranked.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    return ranked[: max(5, min(int(max_results), 20))]


def parse_external_literature_results(text: str, max_results: int = 20) -> dict[str, Any]:
    payload = _json_payload(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
        raise ValueError("외부 LLM 응답에 papers 배열이 없습니다.")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["papers"], start=1):
        if not isinstance(item, dict):
            continue
        title = re.sub(r"\s+", " ", str(item.get("title", "")).strip())
        if not title:
            continue
        source_id = str(item.get("source_id", "")).strip()
        source_url = normalize_result_url(str(item.get("source_url", item.get("url", ""))))
        key = (source_id or title).lower()
        if key in seen:
            continue
        seen.add(key)
        authors = item.get("authors", [])
        if isinstance(authors, str):
            authors = [part.strip() for part in authors.split(",") if part.strip()]
        if not isinstance(authors, list):
            authors = []
        try:
            score = int(float(item.get("relevance_score", 0)))
        except (TypeError, ValueError):
            score = 0
        normalized_pdf = normalize_result_url(str(item.get("pdf_url", "")))
        provisional = {"source_id": source_id, "url": source_url, "pdf_url": normalized_pdf}
        arxiv_id = arxiv_id_from_paper(provisional)
        stable_external_id = arxiv_id or source_id or ("external-" + hashlib.sha1((source_url or title).encode("utf-8")).hexdigest()[:16])
        links = paper_access_links({**provisional, "source_id": stable_external_id})
        results.append(
            {
                "source_id": stable_external_id,
                "source": "arxiv" if arxiv_id else "external_llm",
                "url": links.get("source_url") or source_url,
                "html_url": links.get("html_url", ""),
                "pdf_url": links.get("pdf_url") or normalized_pdf,
                "title": title,
                "summary": str(item.get("abstract_or_summary", item.get("summary", ""))).strip(),
                "published": str(item.get("publication_year", item.get("published", ""))).strip(),
                "authors": [str(author).strip() for author in authors if str(author).strip()],
                "relevance_score": max(0, min(score, 100)),
                "quick_take": str(item.get("quick_take", "")).strip(),
                "why_relevant": str(item.get("why_relevant", "")).strip(),
                "caution": str(item.get("caution", "")).strip(),
                "discovery_source": "external",
                "discovery_sources": list(item.get("discovery_sources", [])) if isinstance(item.get("discovery_sources"), list) else ["External LLM"],
                "subjects": list(item.get("keywords", [])) if isinstance(item.get("keywords"), list) else [],
            }
        )
        results[-1]["labels"] = build_paper_labels(results[-1])
    results.sort(key=lambda item: item.get("relevance_score", 0), reverse=True)
    target = max(5, min(int(max_results), 20))
    return {"search_summary": str(payload.get("search_summary", "")).strip(), "papers": results[:target]}
