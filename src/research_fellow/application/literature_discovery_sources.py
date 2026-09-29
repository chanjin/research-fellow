"""External literature-source access and candidate aggregation."""
from __future__ import annotations
import re
import hashlib
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
from research_fellow.infrastructure.arxiv import search as arxiv_search
from research_fellow.application.literature_discovery_formats import _canonical_paper_key, _merge_paper, arxiv_id_from_paper, normalize_result_url, build_paper_labels
def google_scholar_url(paper: dict[str, Any]) -> str:
    title = str(paper.get("title", "")).strip()
    return f"https://scholar.google.com/scholar?q={quote_plus(title)}" if title else "https://scholar.google.com/"


def _request_json(url: str, *, timeout: int = 25) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "ResearchFellow/0.1 literature-discovery (mailto:research@example.invalid)"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_semantic_scholar(query: str, limit: int = 10) -> list[dict[str, Any]]:
    params = urlencode({
        "query": query,
        "limit": max(1, min(int(limit), 20)),
        "fields": "paperId,title,abstract,year,authors,url,externalIds,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy",
    })
    payload = _request_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}")
    results: list[dict[str, Any]] = []
    for item in payload.get("data", []) or []:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        external_ids = item.get("externalIds") or {}
        doi = str(external_ids.get("DOI") or "").strip()
        arxiv = str(external_ids.get("ArXiv") or "").strip()
        source_id = arxiv or doi or str(item.get("paperId") or "").strip()
        source_url = f"https://doi.org/{doi}" if doi else (f"https://arxiv.org/abs/{arxiv}" if arxiv else str(item.get("url") or ""))
        open_pdf = item.get("openAccessPdf") or {}
        fields = item.get("fieldsOfStudy") or []
        s2fields = item.get("s2FieldsOfStudy") or []
        subjects = list(fields) + [x.get("category", "") for x in s2fields if isinstance(x, dict)]
        paper = {
            "source_id": source_id, "source": "semantic_scholar", "url": source_url,
            "pdf_url": str(open_pdf.get("url") or ""), "title": str(item.get("title") or "").strip(),
            "summary": str(item.get("abstract") or "").strip(), "published": str(item.get("year") or ""),
            "authors": [str(a.get("name") or "").strip() for a in (item.get("authors") or []) if isinstance(a, dict) and a.get("name")],
            "doi": doi, "subjects": [str(x).strip() for x in subjects if str(x).strip()],
            "discovery_sources": ["Semantic Scholar"],
        }
        paper["labels"] = build_paper_labels(paper)
        results.append(paper)
    return results


def search_crossref(query: str, limit: int = 10) -> list[dict[str, Any]]:
    params = urlencode({"query.bibliographic": query, "rows": max(1, min(int(limit), 20))})
    payload = _request_json(f"https://api.crossref.org/works?{params}")
    items = ((payload.get("message") or {}).get("items") or [])
    results: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title_list = item.get("title") or []
        title = str(title_list[0] if title_list else "").strip()
        if not title:
            continue
        doi = str(item.get("DOI") or "").strip()
        year = ""
        for date_key in ("published-print", "published-online", "issued"):
            parts = (((item.get(date_key) or {}).get("date-parts") or [[]])[0] or [])
            if parts:
                year = str(parts[0]); break
        authors=[]
        for a in item.get("author") or []:
            if not isinstance(a, dict):
                continue
            name = " ".join(part for part in [str(a.get("given") or "").strip(), str(a.get("family") or "").strip()] if part)
            if name: authors.append(name)
        pdf_url = ""
        for link in item.get("link") or []:
            if isinstance(link, dict) and "pdf" in str(link.get("content-type", "")).lower():
                pdf_url = str(link.get("URL") or "").strip(); break
        paper = {
            "source_id": doi or str(item.get("URL") or title), "source": "crossref",
            "url": f"https://doi.org/{doi}" if doi else str(item.get("URL") or ""),
            "pdf_url": pdf_url, "title": title, "summary": _strip_markup(str(item.get("abstract") or "")),
            "published": year, "authors": authors, "doi": doi,
            "subjects": [str(x).strip() for x in (item.get("subject") or []) if str(x).strip()],
            "venue": str(((item.get("container-title") or [""])[0]) or "").strip(),
            "discovery_sources": ["Crossref"],
        }
        paper["labels"] = build_paper_labels(paper)
        results.append(paper)
    return results


def collect_multisource_candidates(
    topic: str, context: str, queries: list[str], sources: list[str], max_results: int = 12,
    *, arxiv_searcher: Callable[[str, int], list[dict[str, Any]]] = arxiv_search,
    semantic_searcher: Callable[[str, int], list[dict[str, Any]]] = search_semantic_scholar,
    crossref_searcher: Callable[[str, int], list[dict[str, Any]]] = search_crossref,
) -> list[dict[str, Any]]:
    target = max(5, min(int(max_results), 20))
    per_source = max(5, min(12, target))
    collected: list[dict[str, Any]] = []
    selected = set(sources or ["arxiv"])
    if "arxiv" in selected:
        for item in collect_arxiv_candidates(queries, max_results=max(target, 10), searcher=arxiv_searcher):
            item = dict(item); item["source"] = "arxiv"; item["discovery_sources"] = ["arXiv"]
            item["labels"] = build_paper_labels(item); collected.append(item)
    plain_query = re.sub(r"\s+", " ", f"{topic} {context}".strip())[:500]
    if "semantic_scholar" in selected:
        try: collected.extend(semantic_searcher(plain_query, per_source))
        except Exception: pass
    if "crossref" in selected:
        try: collected.extend(crossref_searcher(plain_query, per_source))
        except Exception: pass
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for paper in collected:
        key = _canonical_paper_key(paper)
        if not key or key == "title:":
            continue
        if key not in merged:
            merged[key] = dict(paper); order.append(key)
        else:
            merged[key] = _merge_paper(merged[key], paper)
    return [merged[key] for key in order][: max(target * 3, 30)]


def download_discovery_pdf(paper: dict[str, Any], root: Path, *, max_bytes: int = 50 * 1024 * 1024) -> str:
    """Download a verified PDF to local shelf storage and return its path."""
    links = paper_access_links(paper)
    pdf_url = links.get("pdf_url", "")
    if not pdf_url:
        raise ValueError("직접 다운로드할 PDF URL이 없습니다.")
    parsed = urlparse(pdf_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("HTTP(S) PDF URL만 다운로드할 수 있습니다.")
    request = Request(pdf_url, headers={"User-Agent": "ResearchFellow/0.1 literature-discovery"})
    with urlopen(request, timeout=60) as response:
        content_type = str(response.headers.get("Content-Type", "")).lower()
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("PDF가 50MB를 초과해 자동 보관하지 않았습니다.")
    if not data.startswith(b"%PDF") and "application/pdf" not in content_type:
        raise ValueError("해당 URL에서 PDF 파일을 확인하지 못했습니다.")
    root.mkdir(parents=True, exist_ok=True)
    identifier = links.get("arxiv_id") or str(paper.get("source_id", "")) or str(paper.get("title", "paper"))
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", identifier).strip("._") or hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:16]
    target = root / f"{safe}.pdf"
    target.write_bytes(data)
    return str(target)


def collect_arxiv_candidates(
    queries: list[str], max_results: int = 12, *, searcher: Callable[[str, int], list[dict[str, Any]]] = arxiv_search,
) -> list[dict[str, Any]]:
    """Run a small multi-query arXiv search and deduplicate candidates."""
    target = max(5, min(int(max_results), 20))
    per_query = max(5, min(10, target))
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for query in queries[:5]:
        for paper in searcher(query, per_query):
            source_id = str(paper.get("source_id", "")).strip()
            title = re.sub(r"\s+", " ", str(paper.get("title", "")).strip())
            title_key = title.lower()
            if (source_id and source_id in seen_ids) or (title_key and title_key in seen_titles):
                continue
            if source_id:
                seen_ids.add(source_id)
            if title_key:
                seen_titles.add(title_key)
            enriched = {**paper, "title": title, "source": paper.get("source") or "arxiv", "discovery_sources": paper.get("discovery_sources") or ["arXiv"]}
            enriched["labels"] = build_paper_labels(enriched)
            merged.append(enriched)
            if len(merged) >= max(target * 2, 20):
                break
        if len(merged) >= max(target * 2, 20):
            break
    return merged[: max(target * 2, 20)]
