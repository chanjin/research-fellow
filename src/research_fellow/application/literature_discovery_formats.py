"""Normalization and structured-output helpers for literature discovery."""
from __future__ import annotations
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, unquote, urlparse

_STOPWORDS = {
    "the","and","for","with","from","that","this","into","using","based","toward","towards","via","are","was","were","has","have","had","not","but","its","their","our","your","can","may","model","models","paper","study","approach","method","methods","analysis","results","data","task","tasks","learning","research"
}

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


def _repair_unescaped_json_quotes(value: str) -> str:
    """Escape prose quotes that an LLM placed inside a JSON string.

    A quote closes a JSON string only when the following token can legally
    follow a key or value.  Quotes followed by ordinary prose are preserved as
    content and escaped.  This deliberately runs only after strict parsing has
    failed.
    """
    output: list[str] = []
    in_string = False
    index = 0
    while index < len(value):
        char = value[index]
        if char != '"':
            output.append(char)
            index += 1
            continue
        slash_count = 0
        scan = index - 1
        while scan >= 0 and value[scan] == "\\":
            slash_count += 1
            scan -= 1
        if slash_count % 2:
            output.append(char)
            index += 1
            continue
        if not in_string:
            in_string = True
            output.append(char)
            index += 1
            continue

        look = index + 1
        while look < len(value) and value[look].isspace():
            look += 1
        next_char = value[look] if look < len(value) else ""
        closes_string = next_char in {"", ":", "}", "]"}
        if next_char == ",":
            after_comma = look + 1
            while after_comma < len(value) and value[after_comma].isspace():
                after_comma += 1
            following = value[after_comma] if after_comma < len(value) else ""
            closes_string = following in {'"', "{", "[", "}", "]", ""}
        elif next_char == '"':
            # A missing comma between fields:  "value" "next_key": ...
            closes_string = bool(re.match(r'"(?:\\.|[^"\\])*"\s*:', value[look:]))
        if closes_string:
            in_string = False
            output.append(char)
        else:
            output.append('\\"')
        index += 1
    return "".join(output)


def _repair_llm_json_structure(value: str) -> str:
    """Repair conservative, high-frequency structural mistakes in LLM JSON."""
    repaired = _repair_unescaped_json_quotes(value)
    # Missing comma between a value/container and the next object, array or key.
    repaired = re.sub(r'([}\]])\s*(?=[{\[])', r'\1,', repaired)
    repaired = re.sub(r'([}\]])\s*(?="(?:\\.|[^"\\])*"\s*:)', r'\1,', repaired)
    repaired = re.sub(r'("|\b(?:true|false|null)|-?\d+(?:\.\d+)?)\s+(?="(?:\\.|[^"\\])*"\s*:)', r'\1,', repaired)
    # Trailing commas are common in generated lists and objects.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


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
    except json.JSONDecodeError as first_error:
        structurally_repaired = _repair_llm_json_structure(repaired)
        try:
            return json.loads(structurally_repaired)
        except json.JSONDecodeError as error:
            lines = structurally_repaired.splitlines()
            problem_line = lines[error.lineno - 1].strip() if 0 < error.lineno <= len(lines) else ""
            excerpt = problem_line[max(0, error.colno - 45): error.colno + 45]
            location = f"line {error.lineno}, column {error.colno}"
            detail = f" 문제 위치: {excerpt}" if excerpt else ""
            raise ValueError(
                f"JSON 응답을 자동 보정한 뒤에도 해석하지 못했습니다({location}): {error.msg}."
                f"{detail} 응답 전체를 다시 만들 필요 없이 표시된 줄의 따옴표·쉼표·괄호만 수정해 주세요."
            ) from first_error


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
