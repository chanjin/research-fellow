"""Best-effort citation metrics for arXiv candidates via Semantic Scholar Graph API."""
from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=paperId,citationCount,influentialCitationCount,year"


def enrich_citation_counts(candidates: list[dict[str, Any]], timeout_seconds: int = 30) -> list[dict[str, Any]]:
    """Attach citation metadata without making literature search depend on the service.

    Semantic Scholar accepts ARXIV:<id> identifiers in batches. Any network,
    rate-limit, or parsing failure leaves the candidate usable with an explicit
    unavailable citation state.
    """
    if not candidates:
        return []
    ids = [f"ARXIV:{str(item.get('source_id', '')).strip()}" for item in candidates]
    payload = json.dumps({"ids": ids}).encode("utf-8")
    try:
        request = Request(
            BATCH_URL,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "ResearchFellow/0.1 citation-enrichment"},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except Exception:
        return [{**item, "citation_count": None, "influential_citation_count": None, "citation_source": "unavailable"} for item in candidates]

    enriched: list[dict[str, Any]] = []
    for item, row in zip(candidates, rows if isinstance(rows, list) else []):
        row = row or {}
        enriched.append({
            **item,
            "citation_count": row.get("citationCount"),
            "influential_citation_count": row.get("influentialCitationCount"),
            "citation_source": "semantic_scholar" if row else "unavailable",
        })
    if len(enriched) < len(candidates):
        enriched.extend({**item, "citation_count": None, "influential_citation_count": None, "citation_source": "unavailable"} for item in candidates[len(enriched):])
    return enriched
