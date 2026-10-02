"""Researcher-facing per-paper review actions for M1 literature rounds.

One literature candidate is reviewed as a small researcher job: inspect provenance
and links, decide whether to preserve it, then (for preserved papers) run a
question-scoped first review and stage knowledge-card candidates.  Knowledge
approval remains a separate Decision interaction.
"""
from __future__ import annotations
import re
from typing import Any, Mapping

from research_fellow.origin_lineage import merge_origin_links, normalize_origin_links
from research_fellow.storage import Ledger
from research_fellow.application.paper_review_tasks import (
    apply_inline_paper_review_response,
    ensure_paper_review_task,
)


def _norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", str(value or "").casefold()).strip()


def _profile_for_intent(ledger: Ledger, intent_id: str) -> dict[str, Any]:
    return next((dict(x) for x in ledger.search_profiles(include_deleted=True) if str(x.get("intent_id") or "") == intent_id), {})


def find_existing_shelf_paper(ledger: Ledger, candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Best-effort duplicate lookup without creating another paper row."""
    source_id = str(candidate.get("source_id") or "").strip().casefold()
    title_key = _norm_title(str(candidate.get("title") or ""))
    abstract_url = str(candidate.get("abstract_url") or candidate.get("source_url") or "").strip().rstrip("/").casefold()
    for paper in ledger.shelf_papers():
        if source_id and str(paper.get("source_id") or "").strip().casefold() == source_id:
            return paper
        if abstract_url and str(paper.get("abstract_url") or paper.get("source_url") or "").strip().rstrip("/").casefold() == abstract_url:
            return paper
        if title_key and _norm_title(str(paper.get("title") or "")) == title_key:
            return paper
    return None


def _research_origins(ledger: Ledger, paper: Mapping[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for link in normalize_origin_links(paper.get("origin_links") or []):
        if str(link.get("origin_type") or "") != "researcher_question":
            continue
        rq_id = str(link.get("origin_id") or "")
        rq = ledger.research_question_thread(rq_id) or {}
        question = str(rq.get("question") or link.get("research_question") or link.get("research_title") or link.get("label") or "").strip()
        if rq_id and rq_id not in seen:
            seen.add(rq_id)
            result.append({"rq_id": rq_id, "question": question})
    return result


def candidate_library_context(ledger: Ledger, candidate: Mapping[str, Any]) -> dict[str, Any]:
    existing = find_existing_shelf_paper(ledger, candidate)
    if not existing:
        return {"already_in_library": False, "library_paper": None, "linked_research_questions": []}
    return {
        "already_in_library": True,
        "library_paper": existing,
        "linked_research_questions": _research_origins(ledger, existing),
    }


def preserve_literature_candidate(
    ledger: Ledger,
    candidate: Mapping[str, Any],
    *,
    intent_id: str = "",
    create_review_task: bool = True,
) -> dict[str, Any]:
    item = dict(candidate)
    authors = item.get("authors") or []
    if isinstance(authors, str):
        authors = [x.strip() for x in authors.split(",") if x.strip()]
    abstract_url = str(item.get("abstract_url") or item.get("source_url") or item.get("url") or "").strip()
    full_text_url = str(item.get("full_text_url") or "").strip()
    pdf_url = str(item.get("pdf_url") or "").strip()
    profile = _profile_for_intent(ledger, intent_id)
    # SearchProfile carries the canonical RQ lineage. Merge it with any discovery
    # lineage instead of inventing an {intent_id: ...} link that normalization drops.
    origin_links = merge_origin_links(item.get("origin_links") or [], profile.get("origin_links") or [])
    existing_before = find_existing_shelf_paper(ledger, item)
    paper = ledger.upsert_shelf_paper({
        "paper_id": str((existing_before or {}).get("paper_id") or ""),
        "title": str(item.get("title") or "Untitled paper"),
        "authors": list(authors),
        "publication_year": str(item.get("publication_year") or item.get("year") or item.get("published") or "")[:4],
        "source_url": abstract_url,  # legacy compatibility
        "abstract_url": abstract_url,
        "full_text_url": full_text_url,
        "pdf_url": pdf_url,
        "source_id": str(item.get("source_id") or ""),
        "pdf_path": "",
        "labels": list((existing_before or {}).get("labels") or []) or ["M1 discovery"],
        "shelf_status": str((existing_before or {}).get("shelf_status") or "reference"),
        "reading_status": str((existing_before or {}).get("reading_status") or "unread"),
        "asset_type": "paper",
        "intake_source": str((existing_before or {}).get("intake_source") or "m1_external_discovery_researcher_selected"),
        "origin_links": origin_links,
        "abstract": str(item.get("summary") or item.get("abstract_or_summary") or item.get("abstract") or ""),
    })
    task = ensure_paper_review_task(ledger, intent_id=intent_id, paper=paper, candidate=item) if create_review_task else None
    from research_fellow.application.auto_literature import selected_paper_review_prompt
    review_prompt = selected_paper_review_prompt(profile, [{**item, "paper_id": str(paper.get("paper_id") or ""), "full_text_url": full_text_url, "pdf_url": pdf_url}])
    return {
        "paper": paper,
        "review_task": task,
        "review_prompt": review_prompt,
        "already_in_library": existing_before is not None,
        "linked_research_questions": _research_origins(ledger, paper),
        "abstract_url": abstract_url,
        "full_text_url": full_text_url,
        "pdf_url": pdf_url,
    }


def apply_candidate_review_response(
    ledger: Ledger,
    *,
    intent_id: str,
    paper_id: str,
    candidate: Mapping[str, Any],
    response: str,
) -> dict[str, Any]:
    """Apply one inline Review response and stage read-only knowledge candidates."""
    return apply_inline_paper_review_response(
        ledger,
        intent_id=intent_id,
        paper_id=paper_id,
        candidate=candidate,
        response=response,
    )



def _extract_local_pdf_text(path: Path, *, max_chars: int = 80000) -> tuple[str, str]:
    """Extract source text from a researcher-provided PDF for manual LLM review.

    The PDF path remains the durable asset reference. Extracted text is transient
    prompt context so we do not introduce a second source-of-truth field/table.
    """
    if not path.is_file():
        raise ValueError("연결된 로컬 PDF 파일을 찾을 수 없습니다.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception as error:
        raise ValueError(f"PDF를 열 수 없습니다: {error}") from error

    chunks: list[str] = []
    total = 0
    truncated = False
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if not text:
            continue
        chunk = f"[PAGE {index}]\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            truncated = True
            total = max_chars
            break
        chunks.append(chunk)
        total += len(chunk) + 2

    extracted = "\n\n".join(chunks).strip()
    if not extracted:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF인지 확인해 주세요.")
    note = f"로컬 PDF {len(reader.pages)}페이지에서 텍스트 {len(extracted):,}자를 추출했습니다."
    if truncated:
        note += f" Prompt 크기를 위해 앞쪽 {max_chars:,}자까지만 포함했습니다."
    return extracted, note

def attach_candidate_local_pdf(
    ledger: Ledger,
    *,
    intent_id: str,
    paper_id: str,
    candidate: Mapping[str, Any],
    filename: str,
    content: bytes,
    storage_root,
) -> dict[str, Any]:
    """Attach a local PDF during candidate review and rebuild the review prompt."""
    from pathlib import Path
    from research_fellow.application.paper_library import attach_paper_local_pdf
    from research_fellow.application.auto_literature import selected_paper_review_prompt

    attached = attach_paper_local_pdf(
        ledger, paper_id, filename=filename, content=content, storage_root=storage_root,
    )
    paper = ledger.shelf_paper(str(paper_id)) or {}
    profile = _profile_for_intent(ledger, intent_id)
    local_path = Path(str(attached.get("pdf_path") or "")).expanduser()
    local_name = local_path.name if local_path.name else Path(filename).name

    full_text, extraction_note = _extract_local_pdf_text(local_path)
    prompt_item = {
        **dict(candidate),
        "paper_id": str(paper_id),
        # Keep a real web URL as URL metadata when one exists. The local PDF is
        # supplied to the LLM as extracted text, not as a pseudo URL.
        "full_text_url": str(candidate.get("full_text_url") or paper.get("full_text_url") or ""),
        "pdf_path": str(attached.get("pdf_path") or ""),
        "local_pdf_filename": local_name,
        "full_text_content": full_text,
        "full_text_extraction_note": extraction_note,
    }
    review_prompt = selected_paper_review_prompt(profile, [prompt_item])
    return {
        **attached,
        "full_text_url": str(prompt_item.get("full_text_url") or ""),
        "full_text_extraction_note": extraction_note,
        "review_prompt": review_prompt,
    }
