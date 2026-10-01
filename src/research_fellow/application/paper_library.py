"""Researcher-facing operations for the durable paper/evidence library.

The paper library remains a projection over the existing paper shelf/analysis
state. These helpers only update researcher-owned annotations; knowledge cards
and paper analyses keep their existing lifecycles.
"""
from __future__ import annotations

from typing import Any

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
