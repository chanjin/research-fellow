"""Researcher-facing per-paper review actions for M1 literature rounds.

One literature candidate is reviewed as a small researcher job: inspect provenance
and links, decide whether to preserve it, then (for preserved papers) run a
question-scoped first review and stage knowledge-card candidates.  Knowledge
approval remains a separate Decision interaction.
"""
from __future__ import annotations
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Mapping

import requests

from research_fellow.origin_lineage import merge_origin_links, normalize_origin_links
from research_fellow.infrastructure.web_reader import WebPageExtractionError, _validate_public_http_url
from research_fellow.storage import Ledger
from research_fellow.application.paper_review_tasks import (
    apply_inline_paper_review_response,
    build_rq_paper_review_prompt,
    ensure_paper_review_task,
)


class _ArticleTextParser(HTMLParser):
    """Small dependency-free HTML text extractor with article/main preference."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._article_depth = 0
        self._main_depth = 0
        self.all_text: list[str] = []
        self.article_text: list[str] = []
        self.main_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.casefold()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "article":
            self._article_depth += 1
        if tag == "main":
            self._main_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "article" and self._article_depth:
            self._article_depth -= 1
        if tag == "main" and self._main_depth:
            self._main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", str(data or "")).strip()
        if not text:
            return
        self.all_text.append(text)
        if self._article_depth:
            self.article_text.append(text)
        if self._main_depth:
            self.main_text.append(text)

    def best_text(self) -> str:
        parts = self.article_text or self.main_text or self.all_text
        return "\n".join(parts).strip()


def _trim_references_tail(text: str) -> tuple[str, bool]:
    """Deterministically drop a references/bibliography tail when a late standalone heading is found.

    A late-position guard avoids cutting on table-of-contents entries or incidental
    mentions near the beginning of a paper. Appendices that follow References are
    intentionally excluded together with the reference list.
    """
    clean = str(text or "").strip()
    if not clean:
        return clean, False

    heading = re.compile(r"(?im)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references|bibliography)\s*$")
    minimum_offset = max(5_000, int(len(clean) * 0.35))
    for match in heading.finditer(clean):
        if match.start() >= minimum_offset:
            return clean[: match.start()].rstrip(), True
    return clean, False


def _prepare_full_text(text: str, *, max_chars: int) -> tuple[str, bool, bool]:
    """Remove the references tail first, then enforce the prompt-size limit."""
    clean, references_trimmed = _trim_references_tail(text)
    if len(clean) <= max_chars:
        return clean, False, references_trimmed
    return clean[:max_chars].rstrip(), True, references_trimmed


def _extract_pdf_bytes_text(content: bytes, *, max_chars: int = 150000) -> tuple[str, str]:
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

    raw_text = "\n\n".join(chunks).strip()
    if not raw_text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF인지 확인해 주세요.")
    extracted, truncated, references_trimmed = _prepare_full_text(raw_text, max_chars=max_chars)
    note = f"원격 PDF {len(reader.pages)}페이지에서 텍스트 {len(extracted):,}자를 추출했습니다."
    if references_trimmed:
        note += " References/Bibliography 이후 텍스트는 제외했습니다."
    if truncated:
        note += f" Prompt 크기를 위해 본문 앞쪽 {max_chars:,}자까지만 포함했습니다."
    return extracted, note


def acquire_remote_full_text(url: str, *, max_chars: int = 150000, timeout: float = 20.0) -> tuple[str, str]:
    """Fetch a verified HTML/PDF URL and normalize it into canonical prompt text."""
    target = str(url or "").strip()
    try:
        requested_url = _validate_public_http_url(target)
    except WebPageExtractionError as error:
        raise ValueError(str(error)) from error
    try:
        response = requests.get(
            requested_url,
            timeout=(8, timeout),
            allow_redirects=True,
            headers={"User-Agent": "ResearchFellow/0.1 (+paper full-text acquisition)"},
        )
        response.raise_for_status()
        final_url = _validate_public_http_url(str(getattr(response, "url", "") or requested_url))
    except Exception as error:
        raise ValueError(f"원문 URL에 접근하지 못했습니다: {error}") from error

    content_type = str(response.headers.get("content-type") or "").lower()
    lowered_url = final_url.casefold()
    is_pdf = "application/pdf" in content_type or lowered_url.endswith(".pdf") or "/pdf/" in lowered_url
    if is_pdf:
        if len(response.content) > 25_000_000:
            raise ValueError("원격 PDF가 너무 큽니다(25MB 초과). 로컬 PDF 업로드를 사용해 주세요.")
        return _extract_pdf_bytes_text(response.content, max_chars=max_chars)

    is_html = "text/html" in content_type or "application/xhtml+xml" in content_type or not content_type
    if not is_html:
        raise ValueError(f"지원하지 않는 원문 형식입니다: {content_type or 'unknown content type'}")
    if len(response.content) > 5_000_000:
        raise ValueError("HTML 원문이 너무 큽니다(5MB 초과). PDF 원문을 사용해 주세요.")
    try:
        html = response.text
        parser = _ArticleTextParser()
        parser.feed(html)
        extracted, truncated, references_trimmed = _prepare_full_text(parser.best_text(), max_chars=max_chars)
    except Exception as error:
        raise ValueError(f"HTML 원문 텍스트를 추출하지 못했습니다: {error}") from error
    if len(extracted) < 500:
        raise ValueError("HTML 원문에서 충분한 본문 텍스트를 추출하지 못했습니다.")
    note = f"원격 HTML에서 텍스트 {len(extracted):,}자를 추출했습니다."
    if references_trimmed:
        note += " References/Bibliography 이후 텍스트는 제외했습니다."
    if truncated:
        note += f" Prompt 크기를 위해 본문 앞쪽 {max_chars:,}자까지만 포함했습니다."
    return extracted, note


def _candidate_with_acquired_full_text(candidate: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    item = dict(candidate)
    existing = str(item.get("full_text_content") or "").strip()
    if existing:
        return item, ""

    urls: list[str] = []
    for value in (item.get("full_text_url"), item.get("pdf_url")):
        url = str(value or "").strip()
        if url and url not in urls:
            urls.append(url)
    if not urls:
        return item, "원문 URL이 없어 자동 텍스트 추출을 수행하지 못했습니다."

    errors: list[str] = []
    for url in urls:
        try:
            full_text, note = acquire_remote_full_text(url)
        except ValueError as error:
            errors.append(f"{url}: {error}")
            continue
        item["full_text_content"] = full_text
        item["full_text_extraction_note"] = note
        item["full_text_source_url"] = url
        return item, ""
    return item, " / ".join(errors)


def _norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]+", " ", str(value or "").casefold()).strip()


def _profile_for_intent(ledger: Ledger, intent_id: str) -> dict[str, Any]:
    return next((dict(x) for x in ledger.search_profiles(include_deleted=True) if str(x.get("intent_id") or "") == intent_id), {})


def _research_question_for_intent(ledger: Ledger, intent_id: str) -> dict[str, Any]:
    linked = ledger.research_questions_for_intent(intent_id)
    return dict(linked[0]) if linked else {}


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


def candidate_review_state(ledger: Ledger, candidate: Mapping[str, Any], *, intent_id: str) -> dict[str, Any]:
    """Project durable paper/review/full-text state back into the sequential Review UI."""
    paper = find_existing_shelf_paper(ledger, candidate)
    if not paper:
        return {"paper": None, "reviewed": False, "full_text_ready": False}
    paper_id = str(paper.get("paper_id") or "")
    rq = _research_question_for_intent(ledger, intent_id)
    rq_id = str(rq.get("rq_id") or "")
    analysis = ledger.paper_question_analysis(paper_id, rq_id) if rq_id else ledger.paper_analysis(paper_id)
    full_text = ledger.paper_full_text(paper_id) or {}
    linked_rq_ids = {str(x.get("rq_id") or "") for x in _research_origins(ledger, paper)}
    result: dict[str, Any] = {
        "paper": paper,
        "paper_id": paper_id,
        "reviewed": bool(analysis and str(analysis.get("summary") or "").strip()),
        "analysis": analysis or {},
        "full_text_ready": bool(str(full_text.get("content") or "").strip()),
        "full_text_extraction_note": str(full_text.get("extraction_note") or ""),
        "full_text_source_type": str(full_text.get("source_type") or ""),
        "full_text_source_name": str(full_text.get("source_name") or ""),
        "linked_to_rq": bool(rq_id and rq_id in linked_rq_ids),
    }
    if result["reviewed"]:
        result["summary"] = str((analysis or {}).get("summary") or "")
        requests: list[str] = []
        cards: list[dict[str, Any]] = []
        for row in ledger.phenomena(type_="decision_request"):
            if str(row.get("subject_type") or "") != "knowledge_card":
                continue
            payload = dict(row.get("payload") or {})
            if str(payload.get("paper_id") or "") != paper_id:
                continue
            card = payload.get("card")
            if isinstance(card, Mapping):
                cards.append(dict(card))
            request_id = str(row.get("phenomenon_id") or "")
            if request_id:
                requests.append(request_id)
        result["knowledge_request_ids"] = requests
        result["knowledge_cards"] = cards
        return result
    if result["full_text_ready"]:
        prompt_item = {
            **dict(candidate),
            "paper_id": paper_id,
            "full_text_content": str(full_text.get("content") or ""),
            "full_text_extraction_note": str(full_text.get("extraction_note") or ""),
            "full_text_url": str(candidate.get("full_text_url") or paper.get("full_text_url") or ""),
            "pdf_path": str(paper.get("pdf_path") or ""),
        }
        result["review_prompt"] = build_rq_paper_review_prompt(ledger, intent_id=intent_id, candidate=prompt_item)
    return result


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
    paper_id = str(paper.get("paper_id") or "")
    cached_full_text = ledger.paper_full_text(paper_id) or {}
    if str(cached_full_text.get("content") or "").strip():
        prompt_item = {
            **item,
            "paper_id": paper_id,
            "full_text_url": full_text_url,
            "pdf_url": pdf_url,
            "full_text_content": str(cached_full_text.get("content") or ""),
            "full_text_extraction_note": str(cached_full_text.get("extraction_note") or ""),
        }
        acquisition_error = ""
    else:
        prompt_item, acquisition_error = _candidate_with_acquired_full_text({
            **item,
            "paper_id": paper_id,
            "full_text_url": full_text_url,
            "pdf_url": pdf_url,
        })
        if not acquisition_error and str(prompt_item.get("full_text_content") or "").strip():
            source_url = str(prompt_item.get("full_text_source_url") or full_text_url or pdf_url)
            ledger.save_paper_full_text(
                paper_id,
                content=str(prompt_item.get("full_text_content") or ""),
                source_type="remote_pdf" if source_url.lower().endswith(".pdf") or "/pdf/" in source_url.lower() else "remote_html",
                source_url=source_url,
                extraction_note=str(prompt_item.get("full_text_extraction_note") or ""),
            )
    review_prompt = ""
    if not acquisition_error:
        review_prompt = build_rq_paper_review_prompt(
            ledger, intent_id=intent_id, candidate=prompt_item,
        )
    return {
        "paper": paper,
        "review_task": task,
        "review_prompt": review_prompt,
        "full_text_ready": not bool(acquisition_error),
        "full_text_extraction_note": str(prompt_item.get("full_text_extraction_note") or ""),
        "full_text_acquisition_error": acquisition_error,
        "already_in_library": existing_before is not None,
        "linked_research_questions": _research_origins(ledger, paper),
        "abstract_url": abstract_url,
        "full_text_url": full_text_url,
        "pdf_url": pdf_url,
    }


def refresh_candidate_remote_full_text(
    ledger: Ledger,
    *,
    intent_id: str,
    paper_id: str,
    candidate: Mapping[str, Any],
    full_text_url: str,
) -> dict[str, Any]:
    """Re-acquire HTML/PDF full text after a researcher corrects the source URL."""
    target = str(full_text_url or "").strip()
    if not target:
        raise ValueError("원문 URL을 입력해 주세요.")
    paper = ledger.shelf_paper(str(paper_id)) or {}
    if not paper:
        raise ValueError("서재함 논문을 찾을 수 없습니다.")
    ledger.update_shelf_paper(
        str(paper_id),
        full_text_url=target,
        shelf_status=str(paper.get("shelf_status") or "reference"),
        reading_status=str(paper.get("reading_status") or "unread"),
    )
    prompt_item, acquisition_error = _candidate_with_acquired_full_text({
        **dict(candidate),
        "paper_id": str(paper_id),
        "full_text_url": target,
    })
    if acquisition_error:
        raise ValueError(acquisition_error)
    ledger.save_paper_full_text(
        str(paper_id),
        content=str(prompt_item.get("full_text_content") or ""),
        source_type="remote_pdf" if target.lower().endswith(".pdf") or "/pdf/" in target.lower() else "remote_html",
        source_url=target,
        extraction_note=str(prompt_item.get("full_text_extraction_note") or ""),
    )
    review_prompt = build_rq_paper_review_prompt(
        ledger, intent_id=intent_id, candidate=prompt_item,
    )
    return {
        "paper_id": str(paper_id),
        "full_text_url": target,
        "full_text_ready": True,
        "full_text_extraction_note": str(prompt_item.get("full_text_extraction_note") or ""),
        "review_prompt": review_prompt,
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



def _extract_local_pdf_text(path: Path, *, max_chars: int = 150000) -> tuple[str, str]:
    """Extract source text from a researcher-provided PDF for manual LLM review.

    The PDF path remains the durable original asset reference; extracted text is also
    cached durably so repeated RQ reviews do not need to re-read the binary PDF.
    """
    if not path.is_file():
        raise ValueError("연결된 로컬 PDF 파일을 찾을 수 없습니다.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
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

    raw_text = "\n\n".join(chunks).strip()
    if not raw_text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF인지 확인해 주세요.")
    extracted, truncated, references_trimmed = _prepare_full_text(raw_text, max_chars=max_chars)
    note = f"로컬 PDF {len(reader.pages)}페이지에서 텍스트 {len(extracted):,}자를 추출했습니다."
    if references_trimmed:
        note += " References/Bibliography 이후 텍스트는 제외했습니다."
    if truncated:
        note += f" Prompt 크기를 위해 본문 앞쪽 {max_chars:,}자까지만 포함했습니다."
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
    workspace_keyword: str = "workspace",
) -> dict[str, Any]:
    """Attach a local PDF during candidate review and rebuild the review prompt."""
    from research_fellow.application.paper_library import attach_paper_local_pdf

    attached = attach_paper_local_pdf(
        ledger, paper_id, filename=filename, content=content, storage_root=storage_root,
        workspace_keyword=workspace_keyword,
    )
    paper = ledger.shelf_paper(str(paper_id)) or {}
    local_path = Path(str(attached.get("pdf_path") or "")).expanduser()
    local_name = local_path.name if local_path.name else Path(filename).name

    full_text = str(attached.get("full_text_content") or "").strip()
    extraction_note = str(attached.get("full_text_extraction_note") or "").strip()
    if not full_text:
        cached = ledger.paper_full_text(str(paper_id)) or {}
        full_text = str(cached.get("content") or "").strip()
        extraction_note = str(cached.get("extraction_note") or extraction_note)
    if not full_text:
        full_text, extraction_note = _extract_local_pdf_text(local_path)
        ledger.save_paper_full_text(
            str(paper_id), content=full_text, source_type="uploaded_pdf",
            source_name=local_name, extraction_note=extraction_note,
        )
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
    review_prompt = build_rq_paper_review_prompt(
        ledger, intent_id=intent_id, candidate=prompt_item
    )
    return {
        **attached,
        "full_text_url": str(prompt_item.get("full_text_url") or ""),
        "full_text_extraction_note": extraction_note,
        "review_prompt": review_prompt,
    }
