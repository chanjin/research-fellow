"""Deterministic arXiv metadata lookup for approved M1 search profiles."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivError(RuntimeError):
    pass


def search(query: str, max_results: int = 8, timeout_seconds: int = 30) -> list[dict[str, Any]]:
    """Search a complete arXiv API expression, e.g. ``all:\"agent design\" AND all:evaluation``."""
    params = urlencode({"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"})
    try:
        request = Request(
            f"https://export.arxiv.org/api/query?{params}",
            headers={"User-Agent": "ResearchFellow/0.1 local-literature-discovery"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            root = ElementTree.fromstring(response.read())
    except Exception as error:
        raise ArxivError(f"arXiv 검색을 완료하지 못했습니다: {error}") from error
    return [{
        "source_id": entry.findtext(f"{ATOM}id", default="").strip().rsplit("/", 1)[-1], "source": "arxiv",
        "url": entry.findtext(f"{ATOM}id", default="").strip(),
        "title": " ".join(entry.findtext(f"{ATOM}title", default="").split()),
        "summary": " ".join(entry.findtext(f"{ATOM}summary", default="").split()),
        "published": entry.findtext(f"{ATOM}published", default=""),
        "authors": [author.findtext(f"{ATOM}name", default="") for author in entry.findall(f"{ATOM}author")],
    } for entry in root.findall(f"{ATOM}entry")]
