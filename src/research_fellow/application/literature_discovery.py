"""Interactive M1 literature discovery for quick researcher-led exploration.

This module is intentionally lighter than the approved-Intent literature review.
It helps a researcher understand a topic, inspect 5-20 candidate papers, and move
promising papers into the shelf. It does not create approved knowledge cards.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, urlencode, unquote, urlparse
from urllib.request import Request, urlopen

from research_fellow.infrastructure.arxiv import search as arxiv_search


def _even_backslashes(value: str) -> str:
    """Make LaTeX backslash runs JSON-safe without double-escaping valid pairs."""
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            output.append(value[index])
            index += 1
            continue
        end = index
        while end < len(value) and value[end] == "\\":
            end += 1
        count = end - index
        output.append("\\" * (count if count % 2 == 0 else count + 1))
        index = end
    return "".join(output)


def _repair_llm_json_escapes(value: str) -> str:
    r"""Repair common copy/paste JSON errors caused by LaTeX and raw paths.

    External LLMs frequently put ``$\pi$`` or ``$\text{{...}}$`` directly in a
    JSON string. JSON then treats the slash as an escape (or rejects ``\p``).
    First protect dollar-delimited math, including escapes that JSON would
    otherwise accept but corrupt (for example ``\t``). Then repair remaining
    invalid single-backslash escapes while inside JSON strings.
    """
    repaired = re.sub(r"\$(?:\\.|[^$])*\$", lambda match: _even_backslashes(match.group(0)), value, flags=re.S)
    output: list[str] = []
    in_string = False
    index = 0
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    while index < len(repaired):
        char = repaired[index]
        if char == '"':
            # A quote is escaped only when preceded by an odd slash run.
            slash_count = 0
            scan = index - 1
            while scan >= 0 and repaired[scan] == "\\":
                slash_count += 1
                scan -= 1
            if slash_count % 2 == 0:
                in_string = not in_string
            output.append(char)
            index += 1
            continue
        if in_string and char == "\\":
            end = index
            while end < len(repaired) and repaired[end] == "\\":
                end += 1
            count = end - index
            next_char = repaired[end] if end < len(repaired) else ""
            if count % 2 == 1 and next_char not in valid_escapes:
                count += 1
            output.append("\\" * count)
            index = end
            continue
        output.append(char)
        index += 1
    return "".join(output)


def _json_payload(text: str) -> Any:
    value = (text or "").strip()
    if not value:
        raise ValueError("LLM 응답이 비어 있습니다.")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    starts = [pos for pos in (value.find("{"), value.find("[")) if pos >= 0]
    if starts:
        start = min(starts)
        end_obj = value.rfind("}")
        end_arr = value.rfind("]")
        end = max(end_obj, end_arr)
        if end >= start:
            value = value[start : end + 1]
    repaired = _repair_llm_json_escapes(value)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 응답을 해석하지 못했습니다: {error}. "
            "수식의 백슬래시는 자동 보정했지만, 따옴표·쉼표 또는 JSON 구조도 확인해 주세요."
        ) from error





_STOPWORDS = {
    "the","and","for","with","from","that","this","into","using","based","toward","towards","via","are","was","were","has","have","had","not","but","its","their","our","your","can","may","model","models","paper","study","approach","method","methods","analysis","results","data","task","tasks","learning","research"
}

def _strip_markup(text: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", value).strip()


def build_paper_labels(paper: dict[str, Any], max_labels: int = 6) -> list[str]:
    """Build lightweight initial labels from source keywords/subjects plus title/abstract terms."""
    explicit: list[str] = []
    for key in ("keywords", "subjects", "fields_of_study", "labels"):
        raw = paper.get(key, [])
        if isinstance(raw, str):
            raw = [part.strip() for part in re.split(r"[,;|]", raw) if part.strip()]
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    item = item.get("category") or item.get("name") or item.get("label") or ""
                label = re.sub(r"\s+", " ", str(item or "").strip())
                if label and label.lower() not in {x.lower() for x in explicit}:
                    explicit.append(label)
    text = f"{paper.get('title','')} {paper.get('summary','')}".lower()
    tokens = re.findall(r"[a-z][a-z0-9-]{3,}", text)
    counts: dict[str, int] = {}
    for token in tokens:
        if token in _STOPWORDS or token.isdigit():
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = [token for token, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    labels = explicit[:]
    for token in ranked:
        if len(labels) >= max_labels:
            break
        if token.lower() not in {x.lower() for x in labels}:
            labels.append(token)
    return labels[:max_labels]


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


def _canonical_paper_key(paper: dict[str, Any]) -> str:
    doi = str(paper.get("doi", "")).strip().lower()
    if doi:
        return "doi:" + doi
    arxiv = arxiv_id_from_paper(paper)
    if arxiv:
        return "arxiv:" + arxiv.lower()
    title = re.sub(r"[^a-z0-9]+", " ", str(paper.get("title", "")).lower()).strip()
    return "title:" + title


def _merge_paper(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("summary", "pdf_url", "url", "published", "doi", "venue"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming[key]
    if len(incoming.get("authors", []) or []) > len(merged.get("authors", []) or []):
        merged["authors"] = incoming.get("authors", [])
    sources = []
    for source in list(merged.get("discovery_sources", [])) + list(incoming.get("discovery_sources", [])):
        if source and source not in sources: sources.append(source)
    merged["discovery_sources"] = sources
    subjects = []
    for item in list(merged.get("subjects", [])) + list(incoming.get("subjects", [])):
        if item and item not in subjects: subjects.append(item)
    merged["subjects"] = subjects
    merged["labels"] = build_paper_labels(merged)
    return merged


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

def normalize_result_url(value: str) -> str:
    """Extract a real HTTP(S) URL from plain or Markdown-style LLM output."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Prefer the visible URL when an LLM returns [https://real](google-search-wrapper).
    match = re.match(r"^\[\s*(https?://[^\]\s]+)\s*\]\((https?://[^)]+)\)$", raw, flags=re.I)
    if match:
        raw = match.group(1)
    else:
        # Generic Markdown link: use href unless the label itself is already a URL.
        match = re.match(r"^\[([^\]]+)\]\((https?://[^)]+)\)$", raw, flags=re.I)
        if match:
            label, href = match.groups()
            raw = label.strip() if label.strip().lower().startswith(("http://", "https://")) else href.strip()
    raw = raw.strip("<> \t\r\n")
    if raw.startswith("http://") or raw.startswith("https://"):
        # Unwrap a common Google search URL only when it contains a direct URL query.
        parsed = urlparse(raw)
        if "google." in parsed.netloc.lower() and parsed.path.startswith("/search"):
            q = parse_qs(parsed.query).get("q", [""])[0]
            q = unquote(q).strip()
            if q.startswith(("http://", "https://")):
                return q
        return raw
    return ""


def arxiv_id_from_paper(paper: dict[str, Any]) -> str:
    candidates = [
        str(paper.get("source_id", "")),
        str(paper.get("url", "")),
        str(paper.get("source_url", "")),
        str(paper.get("pdf_url", "")),
    ]
    for raw in candidates:
        value = normalize_result_url(raw) or raw.strip()
        match = re.search(r"(?:arxiv\.org/(?:abs|html|pdf)/)?((?:[a-z-]+/)?\d{4}\.\d{4,5})(?:v\d+)?", value, flags=re.I)
        if match:
            return match.group(1)
    return ""


def paper_access_links(paper: dict[str, Any]) -> dict[str, str]:
    """Return normalized source/html/pdf links suitable for the discovery UI."""
    source_url = normalize_result_url(str(paper.get("url", paper.get("source_url", ""))))
    pdf_url = normalize_result_url(str(paper.get("pdf_url", "")))
    arxiv_id = arxiv_id_from_paper({**paper, "url": source_url, "pdf_url": pdf_url})
    if arxiv_id:
        return {
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            "html_url": f"https://arxiv.org/html/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "arxiv_id": arxiv_id,
        }
    return {"source_url": source_url, "html_url": "", "pdf_url": pdf_url, "arxiv_id": ""}


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

def discovery_search_plan_prompt(topic: str, context: str, max_results: int) -> str:
    return f"""You are the literature-discovery planner for a domain research fellow.
The researcher wants a QUICK exploratory search, not a full systematic review.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

TASK
1. Interpret the research topic in academic terms.
2. Produce 3-5 English arXiv Boolean search expressions that balance recall and precision.
3. Prefer conceptually distinct query variants rather than minor wording changes.
4. Do not invent paper titles, authors, or citations. This step only creates search expressions.
5. The downstream UI will inspect approximately {max_results} candidate papers.

Return ONLY valid JSON:
{{
  "scope_summary": "one short paragraph",
  "queries": [
    "all:\"...\" AND all:\"...\"",
    "..."
  ],
  "search_notes": ["...", "..."]
}}
"""


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


def discovery_triage_prompt(topic: str, context: str, candidates: list[dict[str, Any]], max_results: int) -> str:
    rows = []
    for index, paper in enumerate(candidates[:30], start=1):
        rows.append(
            {
                "ref": f"P{index}",
                "source_id": paper.get("source_id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "abstract": paper.get("summary", ""),
                "url": paper.get("url", ""),
            }
        )
    return f"""You are helping a researcher QUICKLY understand a literature area.
Do not perform a systematic review. Rank only the retrieved papers below.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

RETRIEVED PAPERS
{json.dumps(rows, ensure_ascii=False, indent=2)}

TASK
Select up to {max(5, min(int(max_results), 20))} papers that are most useful for orientation.
For each selected paper:
- give a relevance score from 0 to 100;
- give a 1-2 sentence quick_take of what the paper appears to contribute, based only on title/abstract;
- explain why_relevant to the research topic;
- optionally state a caution if the abstract is only indirectly relevant.
Do not invent facts beyond the supplied metadata and abstract.

Return ONLY valid JSON:
{{
  "papers": [
    {{
      "source_id": "...",
      "relevance_score": 0,
      "quick_take": "...",
      "why_relevant": "...",
      "caution": "..."
    }}
  ]
}}
"""


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


def external_literature_discovery_prompt(topic: str, context: str, max_results: int, sources: list[str] | None = None) -> str:
    target = max(5, min(int(max_results), 20))
    source_names = ", ".join(sources or ["arXiv", "Semantic Scholar", "Crossref", "Google Scholar"])
    return f"""Act as a literature-discovery assistant with web access.
I am doing a QUICK exploratory literature search, not a systematic review.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

SEARCH SOURCES TO CONSULT
{source_names}
Use these sources as discovery aids; verify the paper itself rather than trusting a single index.

Find approximately {target} real academic papers that help me understand this topic.
Prioritize directly relevant work, then a small number of useful adjacent/foundational papers.
Verify that every paper exists. Never invent titles, authors, URLs, DOI, arXiv IDs, or publication years.
Prefer a stable source page URL (arXiv HTML/abstract page, DOI landing page, publisher page, or official paper page). The URL is for reading the source page; do NOT treat it as a PDF download URL.

For each paper give:
- title
- authors
- publication_year
- source_url
- source_id (arXiv ID or DOI when available; otherwise a stable unique identifier or empty string)
- pdf_url (optional; only when you can verify a direct/open PDF URL, otherwise empty)
- abstract_or_summary (brief and factual; if you do not have the abstract, label it as a summary)
- relevance_score 0-100
- quick_take: 1-2 sentences on what the paper contributes
- why_relevant: why I should read it for this research topic
- caution: any important limitation or indirectness

Return ONLY valid JSON with this schema:
If any field contains LaTeX, escape every backslash for JSON (for example, write $\\\\pi$ in the JSON source, not $\\pi$).
{{
  "search_summary": "short orientation to the literature",
  "papers": [
    {{
      "title": "...",
      "authors": ["..."],
      "publication_year": "2025",
      "source_url": "https://...",
      "source_id": "...",
      "pdf_url": "",
      "abstract_or_summary": "...",
      "relevance_score": 90,
      "quick_take": "...",
      "why_relevant": "...",
      "caution": "..."
    }}
  ]
}}
"""


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
