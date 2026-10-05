"""Export helpers for researcher-reviewed Research Question answer drafts."""
from __future__ import annotations

from html import escape
from io import BytesIO
import re


_CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.55; color: #111; }
h1 { font-size: 21pt; margin: 0 0 12pt; }
h2 { font-size: 16pt; margin-top: 18pt; }
h3 { font-size: 13pt; margin-top: 14pt; }
p, li { margin: 0 0 7pt; }
blockquote { border-left: 3px solid #999; padding-left: 10px; color: #444; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; }
th, td { border: 1px solid #bbb; padding: 5px 7px; vertical-align: top; }
code { font-family: monospace; background: #f3f3f3; padding: 1px 3px; }
pre { font-family: monospace; font-size: 9pt; background: #f5f5f5; padding: 8px; white-space: pre-wrap; }
""".strip()


def answer_markdown_bytes(markdown_text: str) -> bytes:
    return str(markdown_text or "").encode("utf-8")


def _inline_markdown(text: str) -> str:
    """Render a conservative subset of inline Markdown without third-party packages."""
    value = escape(str(text or ""))
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2">\1</a>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    value = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", value)
    return value


def _is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_to_html(markdown_text: str) -> str:
    """Render the Markdown constructs used by Research Answer drafts.

    This intentionally avoids a mandatory Markdown dependency so opening the
    Research workspace never fails because an optional export package is absent.
    """
    lines = str(markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    in_code = False
    code_lines: list[str] = []
    code_language = ""
    i = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                out.append(f"<p>{_inline_markdown(text)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_type
        if list_type:
            out.append(f"</{list_type}>")
            list_type = None

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if in_code:
            if stripped.startswith("```"):
                language_class = f' class="language-{escape(code_language)}"' if code_language else ""
                out.append(f"<pre><code{language_class}>{escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                code_language = ""
                in_code = False
            else:
                code_lines.append(raw)
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            close_list()
            in_code = True
            code_language = stripped[3:].strip()
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            i += 1
            continue

        if i + 1 < len(lines) and "|" in raw and _is_table_separator(lines[i + 1]):
            flush_paragraph()
            close_list()
            headers = _split_table_row(raw)
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline_markdown(c)}</th>" for c in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                cells = _split_table_row(lines[i])
                out.append("<tr>" + "".join(f"<td>{_inline_markdown(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{_inline_markdown(heading.group(2))}</h{level}>")
            i += 1
            continue

        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            flush_paragraph()
            close_list()
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            close_list()
            quote_parts: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_parts.append(lines[i].strip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote><p>{_inline_markdown(' '.join(quote_parts))}</p></blockquote>")
            continue

        unordered = re.match(r"^[-+*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.)]\s+(.+)$", stripped)
        if unordered or ordered:
            flush_paragraph()
            desired = "ul" if unordered else "ol"
            if list_type != desired:
                close_list()
                out.append(f"<{desired}>")
                list_type = desired
            item = (unordered or ordered).group(1)
            out.append(f"<li>{_inline_markdown(item)}</li>")
            i += 1
            continue

        close_list()
        paragraph.append(stripped)
        i += 1

    if in_code:
        language_class = f' class="language-{escape(code_language)}"' if code_language else ""
        out.append(f"<pre><code{language_class}>{escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    close_list()
    return "\n".join(out)


def answer_html(markdown_text: str, *, title: str = "Research Answer") -> str:
    body = _markdown_to_html(markdown_text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def answer_html_bytes(markdown_text: str, *, title: str = "Research Answer") -> bytes:
    return answer_html(markdown_text, title=title).encode("utf-8")


def answer_pdf_bytes(markdown_text: str, *, title: str = "Research Answer") -> bytes:
    try:
        import pymupdf as fitz  # Lazy import: Research UI must still load if PDF export is unavailable.
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF 내보내기에는 PyMuPDF가 필요합니다. 프로젝트 의존성을 다시 설치한 뒤 사용하세요."
        ) from exc

    html = answer_html(markdown_text, title=title)
    story = fitz.Story(html=html, user_css=_CSS, em=10.5)
    output = BytesIO()
    writer = fitz.DocumentWriter(output)
    page_rect = fitz.paper_rect("a4")
    margin = 48
    body_rect = fitz.Rect(
        page_rect.x0 + margin,
        page_rect.y0 + margin,
        page_rect.x1 - margin,
        page_rect.y1 - margin,
    )

    def rectfn(_rect_num: int, _filled: fitz.Rect):
        return page_rect, body_rect, fitz.Identity

    story.write(writer, rectfn)
    writer.close()
    return output.getvalue()
