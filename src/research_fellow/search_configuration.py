"""Configuration shared by interactive and scheduled literature discovery."""

from __future__ import annotations

import os
from collections.abc import Mapping


SEARCH_SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "crossref": "Crossref",
}


def configured_search_sources(environ: Mapping[str, str] | None = None) -> list[str]:
    """Return validated internal search sources from process configuration."""
    values = environ if environ is not None else os.environ
    raw = values.get("RESEARCH_FELLOW_SEARCH_SOURCES", "")
    requested = [item.strip().lower() for item in raw.split(",") if item.strip()]
    configured = list(dict.fromkeys(item for item in requested if item in SEARCH_SOURCE_LABELS))
    return configured or list(SEARCH_SOURCE_LABELS)
