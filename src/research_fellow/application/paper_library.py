"""Researcher-facing operations for the durable paper/evidence library.

The paper library remains a projection over the existing paper shelf/analysis
state. These helpers only update researcher-owned annotations; knowledge cards
and paper analyses keep their existing lifecycles.
"""
from __future__ import annotations

from typing import Any
from datetime import datetime
from io import BytesIO
import re

from research_fellow.storage import Ledger


def _labels(value: list[str] | str) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("#", " ").replace(";", ",").split(",")
    else:
        raw = value
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
        label = " ".join(str(item or "").strip().split())
        key = label.casefold()
        if label and key not in seen:
            seen.add(key)
            result.append(label[:80])
    return result[:20]


def update_paper_researcher_metadata(
    ledger: Ledger,
    paper_id: str,
    *,
    labels: list[str] | str,
    researcher_note: str,
    full_text_url: str | None = None,
) -> dict[str, Any]:
    """Update researcher-owned paper labels/comment without changing analysis or knowledge."""
    paper = ledger.shelf_paper(str(paper_id))
    if not paper:
        raise ValueError("서재함에서 논문을 찾을 수 없습니다.")
    cleaned_labels = _labels(labels)
    ledger.update_shelf_paper(
        str(paper_id),
        shelf_status=str(paper.get("shelf_status") or "reference"),
        reading_status=str(paper.get("reading_status") or "unread"),
        labels=cleaned_labels,
        full_text_url=(str(full_text_url or "").strip() if full_text_url is not None else None),
    )
    analysis = ledger.paper_analysis(str(paper_id)) or {}
    ledger.save_paper_analysis(
        str(paper_id),
        research_question=str(analysis.get("research_question") or ""),
        summary=str(analysis.get("summary") or ""),
        reading_raw_output=str(analysis.get("reading_raw_output") or ""),
        researcher_note=str(researcher_note or "").strip(),
        generated=False,
    )
    return {
        "paper_id": str(paper_id),
        "labels": cleaned_labels,
        "researcher_note": str(researcher_note or "").strip(),
        "full_text_url": str(full_text_url or paper.get("full_text_url") or "").strip(),
    }


def _safe_filename_part(value: str, *, max_chars: int) -> str:
    text = re.sub(r'[\\/:*?"<>|]+', " ", str(value or "")).strip()
    text = re.sub(r"\s+", "-", text).strip("-._ ")
    return (text[:max_chars] or "paper")


def _extract_pdf_text_for_storage(content: bytes, *, max_chars: int = 150000) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(content))
    except Exception as error:
        raise ValueError(f"PDF를 열 수 없습니다: {error}") from error
    chunks: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            chunks.append(f"[PAGE {index}]\n{text}")
    raw = "\n\n".join(chunks).strip()
    if not raw:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF인지 확인해 주세요.")
    heading = re.compile(r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references|bibliography)\s*$")
    min_offset = max(5_000, int(len(raw) * 0.35))
    references_trimmed = False
    for match in heading.finditer(raw):
        if match.start() >= min_offset:
            raw = raw[:match.start()].rstrip()
            references_trimmed = True
            break
    truncated = len(raw) > max_chars
    text = raw[:max_chars].rstrip() if truncated else raw
    note = f"로컬 PDF {len(reader.pages)}페이지에서 텍스트 {len(text):,}자를 추출했습니다."
    if references_trimmed:
        note += " References/Bibliography 이후 텍스트는 제외했습니다."
    if truncated:
        note += f" Prompt 크기를 위해 본문 앞쪽 {max_chars:,}자까지만 포함했습니다."
    return text, note


def attach_paper_local_pdf(
    ledger: Ledger,
    paper_id: str,
    *,
    filename: str,
    content: bytes,
    storage_root,
    workspace_keyword: str = "workspace",
) -> dict[str, Any]:
    """Attach a researcher-provided local PDF using a stable researcher-facing filename."""
    from pathlib import Path

    paper = ledger.shelf_paper(str(paper_id))
    if not paper:
        raise ValueError("서재함에서 논문을 찾을 수 없습니다.")
    name = Path(str(filename or "paper.pdf")).name
    if not name.lower().endswith(".pdf"):
        raise ValueError("PDF 파일만 원문으로 연결할 수 있습니다.")
    payload = bytes(content or b"")
    if not payload.startswith(b"%PDF"):
        raise ValueError("선택한 파일이 유효한 PDF로 보이지 않습니다.")

    root = Path(storage_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%y%m%d")
    title_part = _safe_filename_part(str(paper.get("title") or "paper"), max_chars=20)
    workspace_part = _safe_filename_part(str(workspace_keyword or "workspace"), max_chars=24)
    base = f"{date_prefix}-{title_part}-{workspace_part}"
    target = root / f"{base}.pdf"
    suffix = 2
    while target.exists():
        target = root / f"{base}-{suffix}.pdf"
        suffix += 1
    target.write_bytes(payload)
    ledger.update_shelf_pdf_path(str(paper_id), str(target))
    try:
        full_text, extraction_note = _extract_pdf_text_for_storage(payload)
    except Exception:
        # Keep the original PDF for manual recovery even when it is image-only or
        # otherwise not extractable; the caller can surface the extraction error.
        raise
    ledger.save_paper_full_text(
        str(paper_id), content=full_text, source_type="uploaded_pdf",
        source_name=target.name, extraction_note=extraction_note,
    )
    return {
        "paper_id": str(paper_id), "pdf_path": str(target), "filename": target.name,
        "full_text_content": full_text, "full_text_extraction_note": extraction_note,
    }
