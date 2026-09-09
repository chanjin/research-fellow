"""Paper-shelf helpers for M1 research assets.

The shelf preserves papers and their review notes. It deliberately does not
promote a paper summary into approved knowledge or a card into a paper fact.
"""

from __future__ import annotations

import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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




def ensure_shelf_pdf(paper: dict[str, Any], root: Path) -> str:
    """Return a usable local PDF path, downloading an arXiv original when needed.

    Shelf metadata is durable and may sync across machines, while PDF files are
    intentionally machine-local. This helper repairs a missing/stale local path
    on demand from the paper source URL.
    """
    current = str(paper.get("pdf_path") or "").strip()
    if current and Path(current).exists():
        return current

    source_url = str(paper.get("source_url") or "").strip()
    source_id = str(paper.get("source_id") or "").strip()
    if not source_url and source_id:
        source_url = f"https://arxiv.org/abs/{source_id}"
    if "arxiv.org" not in source_url:
        return ""

    if "/abs/" in source_url:
        pdf_url = source_url.replace("/abs/", "/pdf/")
    elif "/pdf/" in source_url:
        pdf_url = source_url
    else:
        return ""
    if not pdf_url.lower().endswith(".pdf"):
        pdf_url += ".pdf"

    root.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", source_id or Path(pdf_url).stem)
    target = root / f"{safe_id or uuid.uuid4().hex[:10]}.pdf"
    if not target.exists():
        request = Request(pdf_url, headers={"User-Agent": "ResearchFellow/0.1 paper-shelf"})
        with urlopen(request, timeout=60) as response:
            target.write_bytes(response.read())
    return str(target)

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
