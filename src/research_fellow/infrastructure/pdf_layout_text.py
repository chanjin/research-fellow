"""Compatibility PDF text extraction used by local paper intake.

The application historically imports ``extract_pdf_text_layout`` from this
module. Keep that API available while using the project's required ``pypdf``
dependency, so existing installations do not fail on a missing module.
"""

from __future__ import annotations

from io import BytesIO


def extract_pdf_text_layout(
    content: bytes,
    *,
    max_chars: int = 150000,
    source_label: str = "로컬 PDF",
) -> tuple[str, str]:
    """Extract page-labeled text and return it with a concise extraction note."""
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(bytes(content or b"")))
    except Exception as error:
        raise ValueError(f"PDF를 열 수 없습니다: {error}") from error

    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = str(page.extract_text() or "").strip()
        except Exception:
            text = ""
        if text:
            pages.append(f"[PAGE {page_number}]\n{text}")

    full_text = "\n\n".join(pages).strip()
    if not full_text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다. 스캔 이미지 PDF인지 확인해 주세요.")

    truncated = len(full_text) > max_chars
    if truncated:
        full_text = full_text[:max_chars].rstrip()
    note = f"{source_label} {len(reader.pages)}페이지에서 텍스트 {len(full_text):,}자를 추출했습니다."
    if truncated:
        note += f" 저장 크기를 위해 본문 앞쪽 {max_chars:,}자까지만 포함했습니다."
    return full_text, note
