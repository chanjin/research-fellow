"""Structured, cached document extraction for M1 curation.

PyMuPDF is the primary PDF reader because it exposes page text blocks and
coordinates.  ``pypdf`` remains a compatibility fallback.  The optional cache
stores only derived text/structure, never uploaded PDF bytes or page images.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_MAX_PAGES = 20
DEFAULT_CHARS_PER_PAGE = 6_000
EXTRACTION_SCHEMA_VERSION = 2


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot provide usable text."""


@dataclass(frozen=True)
class ExtractedBlock:
    """A text block with its source-page location when the PDF provides one."""

    block_id: str
    text: str
    bbox: tuple[float, float, float, float] | None = None
    block_type: str = "text"


@dataclass(frozen=True)
class ExtractedPage:
    """Text tied to its original one-based PDF page or note segment number."""

    page_number: int
    text: str
    section: str | None = None
    truncated: bool = False
    blocks: tuple[ExtractedBlock, ...] = ()


@dataclass(frozen=True)
class ExtractedDocument:
    """Extraction result with page/block provenance useful for a curation case."""

    document_id: str
    file_name: str
    title: str
    author: str
    source_format: str
    original_page_count: int
    pages: list[ExtractedPage]
    extraction_note: str
    extraction_engine: str = "python-text"
    cache_hit: bool = False


def extracted_document_text(document: ExtractedDocument) -> str:
    """Render a deterministic plain-text export with page and block markers."""
    header = [
        f"# {document.title}", f"Source: {document.file_name}", f"Format: {document.source_format}",
        f"Engine: {document.extraction_engine}", f"Extraction note: {document.extraction_note}",
    ]
    if document.author:
        header.append(f"Author: {document.author}")
    body = []
    for page in document.pages:
        body.append(f"\n\n===== Page {page.page_number}" + (f" · {page.section}" if page.section else "") + " =====\n")
        if page.blocks:
            body.extend(f"[block {block.block_id}]\n{block.text}\n" for block in page.blocks)
        else:
            body.append(page.text)
    return "\n".join(header) + "".join(body) + "\n"


def extract_pages(
    uploaded_file: Any,
    max_pages: int = DEFAULT_MAX_PAGES,
    chars_per_page: int = DEFAULT_CHARS_PER_PAGE,
    cache_dir: str | Path | None = None,
) -> list[ExtractedPage]:
    """Backward-compatible page extraction API used by the curation service."""
    return extract_document(uploaded_file, max_pages, chars_per_page, cache_dir).pages


def extract_document(
    uploaded_file: Any,
    max_pages: int = DEFAULT_MAX_PAGES,
    chars_per_page: int = DEFAULT_CHARS_PER_PAGE,
    cache_dir: str | Path | None = None,
) -> ExtractedDocument:
    """Return page/block text, reusing a content-addressed cache when supplied."""
    if max_pages < 1 or chars_per_page < 1:
        raise ValueError("max_pages와 chars_per_page는 1 이상이어야 합니다.")
    file_name = str(getattr(uploaded_file, "name", "uploaded-document"))
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise DocumentExtractionError("PDF, TXT, 또는 Markdown 파일만 지원합니다.")
    try:
        content = uploaded_file.getvalue()
    except AttributeError as error:
        raise DocumentExtractionError("업로드 파일의 바이트를 읽을 수 없습니다.") from error

    document_id = _document_id(suffix, content)
    cache_path = _cache_path(cache_dir, document_id, max_pages, chars_per_page)
    if cache_path and cache_path.exists():
        cached = _read_cache(cache_path)
        if cached:
            return replace(cached, cache_hit=True, extraction_note=f"{cached.extraction_note} 추출 캐시를 재사용했습니다.")

    if suffix == ".pdf":
        document = _extract_pdf_document(file_name, content, document_id, max_pages, chars_per_page)
    else:
        document = _extract_text_document(file_name, content, document_id, max_pages, chars_per_page)
    if cache_path:
        _write_cache(cache_path, document)
    return document


def _document_id(suffix: str, content: bytes) -> str:
    kind = "pdf" if suffix == ".pdf" else "note"
    return f"{kind}-{hashlib.sha256(content).hexdigest()[:16]}"


def _cache_path(cache_dir: str | Path | None, document_id: str, max_pages: int, chars_per_page: int) -> Path | None:
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    return root / f"{document_id}-v{EXTRACTION_SCHEMA_VERSION}-p{max_pages}-c{chars_per_page}.json"


def _read_cache(path: Path) -> ExtractedDocument | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
            return None
        pages = [
            ExtractedPage(
                page_number=item["page_number"], text=item["text"], section=item.get("section"),
                truncated=bool(item.get("truncated")),
                blocks=tuple(
                    ExtractedBlock(
                        block_id=block["block_id"], text=block["text"],
                        bbox=tuple(block["bbox"]) if block.get("bbox") else None,
                        block_type=block.get("block_type", "text"),
                    )
                    for block in item.get("blocks", [])
                ),
            )
            for item in payload["pages"]
        ]
        return ExtractedDocument(
            document_id=payload["document_id"], file_name=payload["file_name"], title=payload["title"],
            author=payload.get("author", ""), source_format=payload["source_format"],
            original_page_count=payload["original_page_count"], pages=pages,
            extraction_note=payload["extraction_note"], extraction_engine=payload.get("extraction_engine", "unknown"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, document: ExtractedDocument) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": EXTRACTION_SCHEMA_VERSION, **asdict(document), "cache_hit": False}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Cache failure must not block research work; extraction remains usable.
        return


def _extract_pdf_document(file_name: str, content: bytes, document_id: str, max_pages: int, chars_per_page: int) -> ExtractedDocument:
    pymupdf_error: Exception | None = None
    try:
        return _extract_pymupdf_document(file_name, content, document_id, max_pages, chars_per_page)
    except DocumentExtractionError as error:
        if "암호" in str(error):
            raise
        pymupdf_error = error
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as error:
        pymupdf_error = error

    try:
        fallback = _extract_pypdf_document(file_name, content, document_id, max_pages, chars_per_page)
    except DocumentExtractionError:
        if pymupdf_error:
            raise DocumentExtractionError(
                f"{file_name}의 PyMuPDF 구조 추출과 pypdf fallback이 모두 실패했습니다. "
                "스캔 PDF라면 OCR 처리 후 다시 업로드하세요."
            ) from pymupdf_error
        raise
    return replace(fallback, extraction_note=f"{fallback.extraction_note} PyMuPDF를 사용할 수 없어 pypdf fallback을 사용했습니다.")


def _extract_pymupdf_document(file_name: str, content: bytes, document_id: str, max_pages: int, chars_per_page: int) -> ExtractedDocument:
    import fitz  # PyMuPDF

    document = fitz.open(stream=content, filetype="pdf")
    try:
        if document.needs_pass and not document.authenticate(""):
            raise DocumentExtractionError("암호로 보호된 PDF입니다. 암호를 제거한 파일을 업로드하세요.")
        pages: list[ExtractedPage] = []
        failed_pages: list[int] = []
        for index in range(min(document.page_count, max_pages)):
            try:
                page = document.load_page(index)
                raw_blocks = page.get_text("blocks", sort=True)
                blocks = _pymupdf_blocks(raw_blocks, index + 1, chars_per_page)
            except Exception:
                failed_pages.append(index + 1)
                continue
            if not blocks:
                continue
            page_text = "\n\n".join(block.text for block in blocks)
            truncated = len(page_text) > chars_per_page
            page_text = page_text[:chars_per_page].rstrip()
            pages.append(ExtractedPage(index + 1, page_text, _section_from_blocks(blocks), truncated, tuple(blocks)))
        if not pages:
            raise DocumentExtractionError("PyMuPDF에서 추출 가능한 텍스트 블록을 찾지 못했습니다.")
        metadata = document.metadata or {}
        notes = ["PyMuPDF 블록·좌표 기반으로 추출했으며 원본 PDF 바이트와 이미지는 저장하지 않습니다."]
        if document.page_count > max_pages:
            notes.append(f"처음 {max_pages}쪽만 추출했습니다(전체 {document.page_count}쪽).")
        if failed_pages:
            notes.append(f"텍스트 블록을 읽지 못한 쪽: {', '.join(map(str, failed_pages))}.")
        return ExtractedDocument(
            document_id=document_id, file_name=file_name, title=str(metadata.get("title") or Path(file_name).stem),
            author=str(metadata.get("author") or ""), source_format="pdf", original_page_count=document.page_count,
            pages=pages, extraction_note=" ".join(notes), extraction_engine="pymupdf-blocks",
        )
    finally:
        document.close()


def _pymupdf_blocks(raw_blocks: list[Any], page_number: int, chars_per_page: int) -> list[ExtractedBlock]:
    blocks = []
    consumed = 0
    for index, raw in enumerate(raw_blocks, start=1):
        if len(raw) < 5 or (len(raw) > 6 and raw[6] != 0):  # ignore non-text image blocks
            continue
        text = _normalize_pdf_text(str(raw[4] or ""))
        if not text:
            continue
        remaining = chars_per_page - consumed
        if remaining <= 0:
            break
        text = text[:remaining].rstrip()
        consumed += len(text)
        blocks.append(ExtractedBlock(
            block_id=f"p{page_number}-b{index:02d}", text=text,
            bbox=(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])), block_type="text",
        ))
    return blocks


def _extract_pypdf_document(file_name: str, content: bytes, document_id: str, max_pages: int, chars_per_page: int) -> ExtractedDocument:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise DocumentExtractionError("암호로 보호된 PDF입니다. 암호를 제거한 파일을 업로드하세요.")
    except DocumentExtractionError:
        raise
    except Exception as error:
        raise DocumentExtractionError(f"{file_name} 파일을 PDF로 열 수 없습니다.") from error
    pages, failed_pages = [], []
    for page_number, page in enumerate(reader.pages[:max_pages], start=1):
        try:
            text = _normalize_pdf_text(page.extract_text() or "")
        except Exception:
            failed_pages.append(page_number)
            continue
        if not text:
            continue
        truncated = len(text) > chars_per_page
        text = text[:chars_per_page].rstrip()
        block = ExtractedBlock(f"p{page_number}-b01", text)
        pages.append(ExtractedPage(page_number, text, _section_for(text), truncated, (block,)))
    if not pages:
        raise DocumentExtractionError(f"{file_name}에서 추출 가능한 텍스트를 찾지 못했습니다.")
    metadata = reader.metadata or {}
    notes = ["pypdf 텍스트 fallback을 사용했으며 원본 PDF 바이트와 이미지는 저장하지 않습니다."]
    if len(reader.pages) > max_pages:
        notes.append(f"처음 {max_pages}쪽만 추출했습니다(전체 {len(reader.pages)}쪽).")
    if failed_pages:
        notes.append(f"텍스트를 읽지 못한 쪽: {', '.join(map(str, failed_pages))}.")
    return ExtractedDocument(
        document_id=document_id, file_name=file_name, title=str(metadata.get("/Title") or Path(file_name).stem),
        author=str(metadata.get("/Author") or ""), source_format="pdf", original_page_count=len(reader.pages),
        pages=pages, extraction_note=" ".join(notes), extraction_engine="pypdf-fallback",
    )


def _extract_text_document(file_name: str, content: bytes, document_id: str, max_pages: int, chars_per_page: int) -> ExtractedDocument:
    try:
        text = content.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise DocumentExtractionError(f"{file_name}은 UTF-8 TXT 또는 Markdown 파일이어야 합니다.") from error
    if not text:
        raise DocumentExtractionError(f"{file_name}에서 사용할 텍스트를 찾지 못했습니다.")
    source_parts = [part.strip() for part in text.split("\f") if part.strip()]
    segments = [part[index : index + chars_per_page] for part in source_parts for index in range(0, len(part), chars_per_page)]
    pages = []
    for number, segment in enumerate(segments[:max_pages], start=1):
        block = ExtractedBlock(f"p{number}-b01", segment)
        pages.append(ExtractedPage(number, segment, _section_for(segment), False, (block,)))
    note = "원본 노트 파일은 저장하지 않고, 추출 텍스트 구간만 사용합니다."
    if len(segments) > max_pages:
        note += f" 처음 {max_pages}개 구간만 추출했습니다(전체 {len(segments)}개 구간)."
    return ExtractedDocument(
        document_id=document_id, file_name=file_name, title=Path(file_name).stem, author="researcher",
        source_format="text", original_page_count=len(segments), pages=pages, extraction_note=note,
        extraction_engine="python-text",
    )


def _normalize_pdf_text(text: str) -> str:
    """Restore line-wrapped words while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])[-‐‑‒–]\s*\n\s*(?=[a-z])", "", text)
    text = re.sub(r"(?<=\w)\s*\n\s*-\s*\n?\s*(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n\s*\n+ ?", "\n\n", text)
    return text.strip()


def _section_from_blocks(blocks: list[ExtractedBlock]) -> str | None:
    return next((_section_for(block.text) for block in blocks if _section_for(block.text)), None)


def _section_for(text: str) -> str | None:
    """Keep a likely heading, without mistaking ordinary sentences for one."""
    for raw_line in text.splitlines()[:12]:
        line = raw_line.strip().lstrip("#").strip()
        if not line or len(line) > 140:
            continue
        if raw_line.lstrip().startswith("#") or re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+", line):
            return line
        if line.isupper() and len(line.split()) <= 16:
            return line
    return None
