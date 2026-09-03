"""Safe, dependency-free web-page extraction for M1 source registration."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests


class WebPageExtractionError(ValueError):
    """Raised when a URL cannot safely yield readable research text."""


@dataclass(frozen=True)
class ExtractedWebPage:
    url: str
    title: str
    author: str
    publication_year: str
    markdown: str
    warnings: tuple[str, ...] = ()


class _ReadableHtml(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "aside", "form", "button", "iframe"}
    _BLOCKS = {"p", "li", "blockquote", "pre", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta: dict[str, str] = {}
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._text: list[str] = []
        self._article_text: list[str] = []
        self._in_title = False
        self._article_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): (value or "") for key, value in attrs}
        self._tag_stack.append(tag)
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "article":
            self._article_depth += 1
        if tag == "meta":
            key = (attributes.get("name") or attributes.get("property") or "").lower()
            value = attributes.get("content", "").strip()
            if key and value:
                self.meta.setdefault(key, value)
        if tag in self._BLOCKS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._BLOCKS:
            self._append("\n")
        if tag == "title":
            self._in_title = False
        if tag == "article" and self._article_depth:
            self._article_depth -= 1
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(html.unescape(data).split())
        if not cleaned:
            return
        if self._in_title:
            self.title = f"{self.title} {cleaned}".strip()
        if self._skip_depth:
            return
        self._append(cleaned + " ")

    def _append(self, value: str) -> None:
        self._text.append(value)
        if self._article_depth:
            self._article_text.append(value)

    def readable_text(self) -> str:
        raw = "".join(self._article_text if len("".join(self._article_text)) >= 500 else self._text)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _validate_public_http_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebPageExtractionError("http 또는 https 웹페이지 주소를 입력하세요.")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise WebPageExtractionError("로컬 또는 내부 주소는 가져올 수 없습니다.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
        if any(ipaddress.ip_address(address).is_private or ipaddress.ip_address(address).is_loopback for address in addresses):
            raise WebPageExtractionError("사설 네트워크 주소는 가져올 수 없습니다.")
    except socket.gaierror as error:
        raise WebPageExtractionError("웹페이지 주소의 호스트를 찾을 수 없습니다.") from error
    return parsed.geturl()


def fetch_web_page(url: str, max_chars: int = 60_000) -> ExtractedWebPage:
    """Fetch a public HTML page and retain article-like text as Markdown."""
    requested_url = _validate_public_http_url(url)
    try:
        response = requests.get(
            requested_url,
            headers={"User-Agent": "Research-Fellow-M1/0.1 (+local research curation)"},
            timeout=(8, 25),
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise WebPageExtractionError(f"웹페이지를 가져오지 못했습니다: {error}") from error
    final_url = _validate_public_http_url(response.url)
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        raise WebPageExtractionError("현재는 HTML 웹페이지 링크만 지원합니다. PDF는 파일로 업로드하세요.")
    if len(response.content) > 5_000_000:
        raise WebPageExtractionError("웹페이지가 너무 큽니다(5MB 초과). 읽을 부분을 복사해 노트로 등록하세요.")
    parser = _ReadableHtml()
    parser.feed(response.text)
    text = parser.readable_text()
    if len(text) < 240:
        raise WebPageExtractionError("정제 후 사용할 본문이 충분하지 않습니다. 로그인·동적 렌더링 페이지일 수 있습니다.")
    title = parser.meta.get("og:title") or parser.meta.get("twitter:title") or parser.title or urlparse(final_url).netloc
    author = parser.meta.get("author") or parser.meta.get("article:author") or ""
    date = parser.meta.get("article:published_time") or parser.meta.get("date") or parser.meta.get("dc.date") or ""
    year_match = re.search(r"(?:19|20)\d{2}", date)
    year = year_match.group(0) if year_match else ""
    cleaned = text[:max_chars].strip()
    warning = () if len(text) <= max_chars else (f"정제 본문이 {max_chars:,}자로 잘렸습니다.",)
    markdown = f"# {title}\n\nSource URL: {final_url}\n"
    if author:
        markdown += f"Author: {author}\n"
    if date:
        markdown += f"Published: {date}\n"
    markdown += f"\n{cleaned}\n"
    return ExtractedWebPage(final_url, title.strip(), author.strip(), year, markdown, warning)
