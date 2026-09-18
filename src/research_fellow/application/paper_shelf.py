"""Paper-shelf helpers for M1 research assets.

The shelf preserves papers and their review notes. It deliberately does not
promote a paper summary into approved knowledge or a card into a paper fact.
"""

from __future__ import annotations

import html
import re
import uuid
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from research_fellow.infrastructure.document_reader import ExtractedDocument


@dataclass(frozen=True)
class StoredPaperUpload:
    """Small UploadedFile-compatible wrapper for extraction from shelf storage."""

    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def store_paper_upload(uploaded_file: Any, root: Path) -> str:
    """Persist an explicitly registered original file under the application data directory."""
    root.mkdir(parents=True, exist_ok=True)
    original_name = Path(str(getattr(uploaded_file, "name", "paper.pdf"))).name
    target = root / f"{uuid.uuid4().hex[:10]}-{original_name}"
    target.write_bytes(uploaded_file.getvalue())
    return str(target)


def pasted_paper_text_upload(title: str, text: str, *, min_chars: int = 300) -> StoredPaperUpload:
    """Convert researcher-pasted paper text into a shelf-storable source document."""
    normalized=str(text or "").replace("\r\n","\n").replace("\r","\n").strip()
    if len(normalized)<min_chars:
        raise ValueError(f"붙여넣은 원문은 최소 {min_chars:,}자 이상이어야 합니다. 초록이 아니라 읽을 본문을 입력하세요.")
    safe_title=re.sub(r"[^0-9A-Za-z가-힣._-]+","_",str(title or "paper")).strip("_.")[:80] or "paper"
    header=f"Paper title: {str(title or '').strip()}\nSource: researcher-pasted full text\n\n"
    return StoredPaperUpload(name=f"{safe_title}.txt",content=(header+normalized+"\n").encode("utf-8"))


def _response_content_type(response: Any) -> str:
    try:
        return str(response.headers.get("Content-Type", "")).lower()
    except Exception:
        return ""


def _response_url(response: Any, fallback: str) -> str:
    try:
        return str(response.geturl() or fallback)
    except Exception:
        return fallback


def _read_limited(response: Any, limit: int) -> bytes:
    try:
        data = response.read(limit + 1)
    except TypeError:
        data = response.read()
    if len(data) > limit:
        raise ValueError("원문 응답이 허용 크기를 초과했습니다.")
    return data


def _looks_like_pdf(url: str, content_type: str, payload: bytes) -> bool:
    return (
        payload.startswith(b"%PDF")
        or "application/pdf" in content_type
        or urlparse(url).path.lower().endswith(".pdf")
    )


def _discover_pdf_urls(page_url: str, html_text: str) -> list[str]:
    """Find likely PDF links from common scholarly landing-page metadata."""
    candidates: list[str] = []
    patterns = [
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        r'<meta[^>]+name=["\']pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:pdf["\'][^>]+content=["\']([^"\']+)["\']',
        r'<link[^>]+type=["\']application/pdf["\'][^>]+href=["\']([^"\']+)["\']',
        r'<a[^>]+href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html_text, flags=re.IGNORECASE):
            candidate = html.unescape(match.group(1)).strip()
            if candidate:
                resolved = urljoin(page_url, candidate)
                if resolved not in candidates:
                    candidates.append(resolved)
    return candidates[:8]


def _download_pdf_candidate(url: str) -> tuple[bytes | None, list[str]]:
    """Fetch a URL and return PDF bytes, or PDF links discovered on an HTML landing page."""
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 ResearchFellow/0.1",
            "Accept": "application/pdf,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=60) as response:
        resolved_url = _response_url(response, url)
        content_type = _response_content_type(response)
        if "text/html" in content_type or "application/xhtml+xml" in content_type:
            payload = _read_limited(response, 2 * 1024 * 1024)
            charset = "utf-8"
            try:
                charset = response.headers.get_content_charset() or "utf-8"
            except Exception:
                pass
            text = payload.decode(charset, errors="replace")
            return None, _discover_pdf_urls(resolved_url, text)

        payload = _read_limited(response, 80 * 1024 * 1024)
        if _looks_like_pdf(resolved_url, content_type, payload):
            return payload, []
        return None, []


class _ReadableHTMLParser(HTMLParser):
    """Extract readable scholarly text from a source HTML page without JS rendering."""

    _BLOCK_TAGS = {
        "article", "section", "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "blockquote", "pre", "figcaption", "td", "th", "br"
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if not self._skip_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self._parts.append(text + " ")

    def readable_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def document_from_source_url(paper: dict[str, Any], max_bytes: int = 8 * 1024 * 1024) -> StoredPaperUpload:
    """Fetch a researcher-supplied scholarly HTML source as a text document.

    ``source_url`` is treated as the canonical/readable paper page, not as a PDF
    download location. Local PDFs remain separately managed through ``pdf_path``.
    """
    source_url = str(paper.get("source_url") or "").strip()
    if not source_url:
        raise ValueError("원문 URL이 등록되어 있지 않습니다.")
    request = Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0 ResearchFellow/0.1",
            "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            resolved_url = _response_url(response, source_url)
            content_type = _response_content_type(response)
            payload = _read_limited(response, max_bytes)
            if _looks_like_pdf(resolved_url, content_type, payload):
                raise ValueError("등록된 원문 URL은 PDF 주소입니다. 이 항목은 HTML 원문 URL로 사용하고 PDF는 로컬 파일로 별도 등록해 주세요.")
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type and "text/plain" not in content_type:
                raise ValueError(f"원문 URL에서 읽을 수 있는 HTML/텍스트를 받지 못했습니다. Content-Type: {content_type or 'unknown'}")
            charset = "utf-8"
            try:
                charset = response.headers.get_content_charset() or "utf-8"
            except Exception:
                pass
            raw_text = payload.decode(charset, errors="replace")
            if "text/plain" in content_type:
                readable = raw_text.strip()
            else:
                parser = _ReadableHTMLParser()
                parser.feed(raw_text)
                readable = parser.readable_text()
            if len(readable) < 500:
                raise ValueError("원문 URL에서 충분한 본문 텍스트를 읽지 못했습니다. 로그인 또는 JavaScript 렌더링이 필요한 페이지일 수 있습니다.")
            title = re.sub(r"[^A-Za-z0-9._-]+", "_", str(paper.get("title") or "source"))[:80] or "source"
            header = f"Source URL: {resolved_url}\nPaper title: {paper.get('title') or ''}\n\n"
            return StoredPaperUpload(name=f"{title}.txt", content=(header + readable).encode("utf-8"))
    except HTTPError as error:
        if error.code in {401,403}:
            raise ValueError(
                f"원문 사이트가 자동 접근을 차단했습니다(HTTP {error.code}). "
                "브라우저에서 원문을 내려받아 PDF/TXT/MD로 등록하거나 본문 텍스트를 붙여 넣으세요."
            ) from error
        raise ValueError(f"원문 URL 요청에 실패했습니다(HTTP {error.code}).") from error
    except URLError as error:
        raise ValueError(f"원문 URL에 연결하지 못했습니다: {error.reason}") from error


def ensure_shelf_pdf(paper: dict[str, Any], root: Path) -> str:
    """Return an already-registered local PDF path.

    Source URLs are canonical HTML/full-text links and are intentionally not
    converted into or used to auto-download PDFs. PDF assets are researcher-
    managed separately through ``pdf_path``.
    """
    current = str(paper.get("pdf_path") or "").strip()
    if current and Path(current).exists():
        return current
    return ""


def document_from_shelf_path(path: str) -> StoredPaperUpload:
    source = Path(path)
    return StoredPaperUpload(name=source.name, content=source.read_bytes())


def paper_analysis_prompt(document: ExtractedDocument, paper: dict[str, Any], research_question: str) -> str:
    """Keep a local-model review bounded while retaining the source distinction."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    sections = []
    for page in document.pages[:8]:
        sections.append(f"[p.{page.page_number}]\n{page.text[:3000]}")
    return render_prompt(
        "m1_paper_shelf_analysis.j2",
        title=paper["title"], authors=", ".join(paper.get("authors", [])),
        research_question=research_question, source_text="\n\n".join(sections)[:18000],
    )


def suggested_paper_labels(summary: str, max_labels: int = 10) -> list[str]:
    """Accept the explicit label line only; malformed model output changes nothing."""
    match = re.search(r"(?im)^\s*(?:labels?|레이블)\s*:\s*(.+)$", summary)
    if not match:
        return []
    labels: list[str] = []
    for raw in match.group(1).split(","):
        label = " ".join(raw.strip(" -•#\t").split())
        if 1 < len(label) <= 48 and label.casefold() not in {item.casefold() for item in labels}:
            labels.append(label)
        if len(labels) == max_labels:
            break
    return labels
