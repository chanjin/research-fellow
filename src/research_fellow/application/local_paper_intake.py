"""Local-PDF intake into the canonical Research Question -> Paper Review flow."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Mapping
from io import BytesIO
import re

from research_fellow.application.literature_candidate_review import find_existing_shelf_paper
from research_fellow.application.paper_library import attach_paper_local_pdf, _extract_pdf_text_for_storage
from research_fellow.application.paper_review_tasks import build_rq_paper_review_prompt, apply_inline_paper_review_response
from research_fellow.domain.research import CurationIntent
from research_fellow.origin_lineage import merge_origin_links
from research_fellow.application.structured_output import extract_json_object



def extract_local_pdf_metadata(*, filename: str, content: bytes) -> dict[str, Any]:
    """Best-effort PDF metadata extraction for researcher uploads."""
    payload = bytes(content or b"")
    if not payload.startswith(b"%PDF"):
        raise ValueError("유효한 PDF 파일을 선택해 주세요.")
    fallback_title = Path(str(filename or "paper.pdf")).stem.strip()
    title = fallback_title
    authors: list[str] = []
    year = ""
    first_page = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(payload))
        meta = reader.metadata or {}
        raw_title = str(getattr(meta, "title", "") or meta.get("/Title") or "").strip()
        if raw_title and raw_title.casefold() not in {"untitled", "microsoft word", "document"} and len(raw_title) > 3:
            title = raw_title
        raw_author = str(getattr(meta, "author", "") or meta.get("/Author") or "").strip()
        if raw_author:
            authors = [x.strip() for x in re.split(r"[;,]", raw_author) if x.strip()]
        raw_date = str(getattr(meta, "creation_date", "") or meta.get("/CreationDate") or "")
        metadata_year = ""
        m = re.search(r"(?:19|20)\d{2}", raw_date)
        if m:
            metadata_year = m.group(0)
        if reader.pages:
            try:
                first_page = str(reader.pages[0].extract_text() or "")[:5000]
            except Exception:
                first_page = ""
    except Exception:
        pass
    if first_page:
        years = re.findall(r"\b(?:19|20)\d{2}\b", first_page)
        if years:
            year = years[0]
    if not year:
        year = locals().get("metadata_year", "")
    return {
        "title": title,
        "authors": authors,
        "publication_year": year,
        "source": "pdf_metadata",
    }


def _latest_unreported_intent_id(ledger: Any, rq_id: str) -> str:
    links = ledger.research_question_intents(str(rq_id))
    if not links:
        return ""
    reports = ledger.phenomena(type_="advice_report")
    for link in links:
        intent_id = str(link.get("intent_id") or "").strip()
        if not intent_id or intent_id.startswith("intent-local-"):
            # Legacy local-PDF review contexts are not literature rounds.
            continue
        published = any(
            row.get("subject_type") == "auto_literature_report"
            and row.get("status") == "completed"
            and str((row.get("payload") or {}).get("intent_id") or "") == intent_id
            for row in reports
        )
        if not published:
            return intent_id
        break
    return ""

def _authors(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _manual_review_intent(ledger: Any, rq: Mapping[str, Any], *, title: str) -> str:
    """Create only the RQ-scoped review context; do not start a discovery round."""
    rq_id = str(rq.get("rq_id") or "").strip()
    question = str(rq.get("question") or "").strip()
    intent_id = f"intent-local-{uuid.uuid4().hex[:12]}"
    intent = CurationIntent(
        intent_id=intent_id,
        title=f"로컬 논문 리뷰 · {title[:45]}",
        purpose=f"연구자가 직접 제공한 논문을 연구질문 관점에서 검토한다: {question}",
        question=question,
        research_context="\n".join(x for x in [
            f"Research question: {question}",
            f"Research context: {str(rq.get('research_context') or '').strip()}" if rq.get("research_context") else "",
            f"Researcher-provided paper: {title}",
        ] if x),
        labels=["local paper"],
        priority="보통",
        expected_evidence="제공된 논문 원문에 근거한 연구질문 관련 주장, 근거, 한계와 적용 조건",
        completion_condition="논문 리뷰와 지식카드 후보가 생성되어 연구자가 검토할 수 있다.",
        execution_mode="manual",
        created_by="m2",
        origin_links=[{
            "origin_type": "researcher_question",
            "origin_id": rq_id,
            "label": question[:100],
            "source_card_ids": [],
        }],
    )
    ledger.create_search_profile(intent.model_dump(mode="json"))
    ledger.link_research_question_intent(rq_id, intent_id, "")
    return intent_id


def add_local_pdf_to_research_question(
    ledger: Any,
    *,
    rq_id: str,
    title: str,
    filename: str,
    content: bytes,
    storage_root: Path,
    workspace_keyword: str,
    publication_year: str = "",
    authors: str | list[str] = "",
) -> dict[str, Any]:
    rq = ledger.research_question_thread(str(rq_id)) or ledger.research_question(str(rq_id))
    if not rq:
        raise ValueError("연구질문을 찾을 수 없습니다.")
    payload = bytes(content or b"")
    metadata = extract_local_pdf_metadata(filename=filename, content=payload)
    paper_title = str(title or "").strip() or str(metadata.get("title") or "").strip() or Path(str(filename or "paper.pdf")).stem.strip()
    if not paper_title:
        raise ValueError("논문 제목을 입력해 주세요.")
    effective_authors = _authors(authors) or list(metadata.get("authors") or [])
    effective_year = str(publication_year or "").strip()[:4] or str(metadata.get("publication_year") or "").strip()[:4]
    if not payload.startswith(b"%PDF"):
        raise ValueError("유효한 PDF 파일을 선택해 주세요.")

    source_id = "localpdf:" + hashlib.sha256(payload).hexdigest()[:20]
    existing = find_existing_shelf_paper(ledger, {"source_id": source_id, "title": paper_title}) or {}
    origin_links = merge_origin_links(existing.get("origin_links") or [], [{
        "origin_type": "researcher_question",
        "origin_id": str(rq_id),
        "label": str(rq.get("question") or "")[:100],
        "source_card_ids": [],
    }])
    paper = ledger.upsert_shelf_paper({
        "paper_id": str(existing.get("paper_id") or ""),
        "title": paper_title,
        "authors": effective_authors,
        "publication_year": effective_year,
        "source_id": source_id,
        "labels": list(existing.get("labels") or []) or ["Local PDF"],
        "shelf_status": str(existing.get("shelf_status") or "reference"),
        "reading_status": str(existing.get("reading_status") or "unread"),
        "asset_type": "paper",
        "intake_source": "researcher_local_pdf",
        "origin_links": origin_links,
    })
    attached = attach_paper_local_pdf(
        ledger,
        str(paper.get("paper_id") or ""),
        filename=filename,
        content=payload,
        storage_root=storage_root,
        workspace_keyword=workspace_keyword,
    )
    intent_id = _latest_unreported_intent_id(ledger, str(rq_id))
    reused_round = bool(intent_id)
    if not intent_id:
        intent_id = _manual_review_intent(ledger, rq, title=paper_title)
    candidate = {
        "paper_id": str(paper.get("paper_id") or ""),
        "title": paper_title,
        "authors": effective_authors,
        "publication_year": effective_year,
        "source_id": source_id,
        "full_text_content": str(attached.get("full_text_content") or ""),
        "full_text_extraction_note": str(attached.get("full_text_extraction_note") or ""),
        "origin_links": origin_links,
    }
    review_prompt = build_rq_paper_review_prompt(ledger, intent_id=intent_id, candidate=candidate)
    ledger.add_research_question_change(
        str(rq_id), "local_paper_added", f"로컬 PDF 논문을 연구질문에 추가했습니다: {paper_title}"
    )
    return {
        "rq_id": str(rq_id),
        "intent_id": intent_id,
        "paper": ledger.shelf_paper(str(paper.get("paper_id") or "")) or paper,
        "review_prompt": review_prompt,
        "review_candidate": candidate,
        "reused_current_round": reused_round,
        "metadata": {"title": paper_title, "authors": effective_authors, "publication_year": effective_year},
        "full_text_extraction_note": str(attached.get("full_text_extraction_note") or ""),
    }



def apply_local_pdf_review_response(
    ledger: Any, *, review_context: Mapping[str, Any], response: str
) -> dict[str, Any]:
    """Apply a local-PDF review inline on the Research surface.

    Local paper intake is not a separate literature round and should not create
    a durable paper_first_review Attention task.  It reuses the canonical
    RQ-conditioned review parser/write path directly.
    """
    context = dict(review_context or {})
    intent_id = str(context.get("intent_id") or "").strip()
    paper = dict(context.get("paper") or {})
    candidate = dict(context.get("review_candidate") or {})
    paper_id = str(paper.get("paper_id") or candidate.get("paper_id") or "").strip()
    if not intent_id or not paper_id or not candidate:
        raise ValueError("로컬 논문 리뷰 컨텍스트를 찾을 수 없습니다. PDF를 다시 추가해 주세요.")
    result = apply_inline_paper_review_response(
        ledger, intent_id=intent_id, paper_id=paper_id, candidate=candidate, response=str(response or ""),
    )
    return {**result, "intent_id": intent_id, "rq_id": str(context.get("rq_id") or "")}


def prepare_research_question_from_local_papers(
    *, initial_question: str, context: str, files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    """Build an external-LLM prompt that refines an RQ from multiple local papers."""
    question = str(initial_question or "").strip()
    if not question:
        raise ValueError("초기 연구질문을 입력해 주세요.")
    if not files:
        raise ValueError("하나 이상의 PDF 논문을 선택해 주세요.")
    papers: list[dict[str, Any]] = []
    blocks: list[str] = []
    remaining = 120_000
    for index, (filename, content) in enumerate(files, start=1):
        payload = bytes(content or b"")
        meta = extract_local_pdf_metadata(filename=filename, content=payload)
        text, note = _extract_pdf_text_for_storage(payload)
        excerpt = text[: max(0, min(45_000, remaining))]
        remaining -= len(excerpt)
        paper = {
            "filename": str(filename),
            "title": str(meta.get("title") or Path(filename).stem),
            "authors": list(meta.get("authors") or []),
            "publication_year": str(meta.get("publication_year") or ""),
            "extraction_note": note,
        }
        papers.append(paper)
        blocks.append(
            f"### SEED PAPER {index}\nTitle: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\n"
            f"Year: {paper['publication_year']}\n\nFULL TEXT EXCERPT:\n{excerpt}"
        )
        if remaining <= 0:
            break
    prompt = f"""You are helping a researcher define a precise research question before literature discovery.

INITIAL RESEARCH QUESTION
{question}

RESEARCHER CONTEXT
{str(context or '').strip()}

The researcher supplied the seed papers below. Use only the supplied texts to improve the question and context.
Do not answer the research question. Identify terminology, scope, assumptions, mechanisms, boundaries, and missing dimensions that make the question more researchable.
The refined question should remain faithful to the researcher's intent while being specific enough to drive a subsequent literature search for related and contrasting papers.

Return JSON only with this exact shape:
{{
  "refined_question": "one clear research question",
  "refined_context": "2-5 concise paragraphs describing the research context, key concepts, scope and why the seed papers matter",
  "refinement_note": "brief explanation of what was clarified or added",
  "search_focus": ["3-8 concepts, mechanisms, comparisons, or evidence gaps to search next"]
}}
Escape any double quotes inside JSON strings as \".

SEED PAPERS
{'\n\n'.join(blocks)}
"""
    return {"prompt": prompt, "papers": papers, "initial_question": question, "initial_context": str(context or "").strip()}


def parse_research_question_from_local_papers_response(response: str) -> dict[str, Any]:
    data = extract_json_object(str(response or ""), message="연구질문 보강 응답은 JSON object여야 합니다.")
    question = str(data.get("refined_question") or "").strip()
    context = str(data.get("refined_context") or "").strip()
    if not question:
        raise ValueError("LLM 응답에서 refined_question을 찾지 못했습니다.")
    return {
        "question": question, "context": context,
        "refinement_note": str(data.get("refinement_note") or "").strip(),
        "search_focus": [str(x).strip() for x in (data.get("search_focus") or []) if str(x).strip()],
    }

def start_research_question_from_local_pdf(
    ledger: Any,
    *,
    question: str,
    context: str,
    title: str,
    filename: str,
    content: bytes,
    storage_root: Path,
    workspace_keyword: str,
    publication_year: str = "",
    authors: str | list[str] = "",
) -> dict[str, Any]:
    research_question = str(question or "").strip()
    if not research_question:
        raise ValueError("이 논문으로 시작할 연구질문을 입력해 주세요.")
    paper_title = str(title or "").strip() or Path(str(filename or "paper.pdf")).stem.strip()
    rq = ledger.create_research_question_thread(
        question=research_question,
        source_type="local_paper",
        rationale="연구자가 제공한 로컬 논문에서 시작된 연구질문입니다.",
        research_context=str(context or "").strip(),
        source_payload={"paper_title": paper_title, "source_filename": Path(str(filename or "paper.pdf")).name},
        status="interested",
    )
    added = add_local_pdf_to_research_question(
        ledger,
        rq_id=str(rq.get("rq_id") or ""),
        title=paper_title,
        filename=filename,
        content=content,
        storage_root=storage_root,
        workspace_keyword=workspace_keyword,
        publication_year=publication_year,
        authors=authors,
    )
    return {**dict(rq), "local_paper": added}
