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
