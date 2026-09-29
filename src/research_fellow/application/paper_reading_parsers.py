"""Structured-output parsing for M1 paper reading."""
from __future__ import annotations
import re
from typing import Any

_READING_FIELD_ALIASES = {
        "question": "question", "질문": "question",
        "연구자 판단이 필요한 질문": "question", "연구자 판단 질문": "question", "판단이 필요한 질문": "question",
        "tentative answer": "tentative_answer", "잠정 답변": "tentative_answer", "잠정적 답변": "tentative_answer", "claim": "tentative_answer", "주장": "tentative_answer",
        "evidence": "evidence", "근거": "evidence", "증거": "evidence",
        "uncertainty": "uncertainty", "불확실성": "uncertainty", "유보": "uncertainty", "한계": "uncertainty", "한계·유보": "uncertainty", "한계 및 유보": "uncertainty",
        "research relevance": "research_relevance", "연구 관련성": "research_relevance", "연구적 관련성": "research_relevance",
        "suggested labels": "suggested_labels", "추천 레이블": "suggested_labels", "제안 레이블": "suggested_labels", "제안 라벨": "suggested_labels", "labels": "suggested_labels", "레이블": "suggested_labels",
        "suggested title": "suggested_title", "추천 카드 제목": "suggested_title", "제안 타이틀": "suggested_title", "제안 제목": "suggested_title", "card title": "suggested_title", "카드 제목": "suggested_title",
        "suggested concepts": "suggested_concepts", "추천 핵심 개념": "suggested_concepts", "제안 개념": "suggested_concepts", "concepts": "suggested_concepts", "핵심 개념": "suggested_concepts",
        "suggested applies to": "suggested_applies_to", "추천 적용 대상": "suggested_applies_to", "제안 적용 대상": "suggested_applies_to", "applies to": "suggested_applies_to", "적용 대상": "suggested_applies_to",
        "suggested conditions": "suggested_conditions", "추천 적용 조건": "suggested_conditions", "제안 조건": "suggested_conditions", "conditions": "suggested_conditions", "적용 조건": "suggested_conditions",
        "suggested limits": "suggested_limits", "추천 한계": "suggested_limits", "제한": "suggested_limits", "limits": "suggested_limits",
        "card context": "suggested_context", "context": "suggested_context", "카드 맥락": "suggested_context", "맥락": "suggested_context",
        "design implication": "suggested_implication", "implication": "suggested_implication", "설계 함의": "suggested_implication", "함의": "suggested_implication",
        "source excerpt": "suggested_source_excerpt", "nearby source text": "suggested_source_excerpt", "주변 원문": "suggested_source_excerpt", "원문 맥락": "suggested_source_excerpt",
}


def _reading_field_name(raw_key: str) -> str:
    """Accept both `Question: text` and Markdown heading forms.

    Local models often emit `**질문 1**` followed by a question on the next
    line. Keep the normalizer shared with the unmatched-section diagnostic.
    """
    normalized = re.sub(r"\s*\([^)]*\)", "", raw_key)
    normalized = re.sub(r"\s*\d+\s*$", "", re.sub(r"^[\s#*\-•]+|[\s#*]+$", "", normalized)).lower()
    return _READING_FIELD_ALIASES.get(normalized, "")


def unconsumed_reading_sections(text: str) -> list[str]:
    """Return explicit model sections whose field labels were not recognized.

    This is a diagnostic only: it never changes the saved raw response or
    knowledge-card data.  It makes local-model format drift visible without
    flooding the UI with normally parsed prose.
    """
    sections: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        nonlocal pending
        value = "\n".join(pending).strip()
        if value:
            sections.append(value)
        pending = []

    structural = re.compile(r"(?i)^(?:research|paper) summary|연구\s*(?:서머리|요약)|m1\s*research-context interpretation|m1\s*연구.*해석|suggested shelf labels|추천 서재 레이블|질문\s*블록\s*\d+$")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if pending:
                pending.append(line)
            continue
        key = stripped.split(":", 1)[0] if ":" in stripped else stripped
        is_known = bool(_reading_field_name(key) or _reading_field_name(stripped))
        is_separator = bool(re.fullmatch(r"---+", stripped))
        clean_heading = re.sub(r"^[\s#*\-•]+|[\s#*]+$", "", stripped)
        is_structural = bool(structural.match(clean_heading))
        is_heading = ":" in stripped and len(key.strip()) <= 60 or (stripped.startswith(("#", "**")) and stripped.endswith("**"))
        if is_known or is_separator or is_structural:
            flush()
        elif is_heading:
            flush()
            pending = [line]
        elif pending:
            pending.append(line)
    flush()
    return sections


def parse_reading_questions(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    values: dict[str, str] = {}
    current = ""

    def append_current() -> None:
        evidence: list[str] = []
        for item in values.get("evidence", "").split(";"):
            cleaned = item.strip(" -*•")
            if not cleaned:
                continue
            # Models often place a page marker and its hint in separate
            # semicolon fragments. Keep them as one provenance entry.
            if evidence and not re.search(r"\bp\.\s*\d+\b", cleaned, flags=re.I):
                evidence[-1] = f"{evidence[-1]}; {cleaned}"
            else:
                evidence.append(cleaned)
        if len(values.get("question", "")) >= 6 and len(values.get("tentative_answer", "")) >= 12 and evidence:
            results.append({
                "question": values["question"],
                "tentative_answer": values["tentative_answer"],
                "evidence": evidence[:5],
                "uncertainty": values.get("uncertainty", "원문 범위를 넘어선 일반화는 유보합니다."),
                "research_relevance": values.get("research_relevance", ""),
                "suggested_labels": values.get("suggested_labels", ""),
                "suggested_title": values.get("suggested_title", ""),
                "suggested_concepts": values.get("suggested_concepts", ""),
                "suggested_applies_to": values.get("suggested_applies_to", ""),
                "suggested_conditions": values.get("suggested_conditions", ""),
                "suggested_limits": values.get("suggested_limits", ""),
                "suggested_context": values.get("suggested_context", ""),
                "suggested_implication": values.get("suggested_implication", ""),
                "suggested_source_excerpt": values.get("suggested_source_excerpt", ""),
            })

    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"---+", stripped):
            append_current()
            values, current = {}, ""
            continue
        labelled_field = ""
        inline_value = ""
        if ":" in stripped:
            key, inline_value = stripped.split(":", 1)
            labelled_field = _reading_field_name(key)
        heading_field = _reading_field_name(stripped)
        field = labelled_field or heading_field
        if field:
            # Some local models omit `---` between `질문 1`, `질문 2`, ... .
            # A new question is itself an unambiguous block boundary.
            if field == "question" and values.get("question"):
                append_current()
                values = {}
            current = field
            # A Markdown heading can leave its closing emphasis marker after
            # the colon when we split the line.
            values[current] = inline_value.strip(" *") if labelled_field else ""
            continue
        if current and stripped:
            content = stripped.strip(" -*•")
            # Keep Markdown evidence bullets independently usable as
            # provenance entries rather than collapsing them into prose.
            if current == "evidence" and stripped.startswith(("-", "*", "•")) and values.get(current):
                values[current] = f"{values[current]}; {content}".strip()
            else:
                values[current] = f"{values.get(current, '')} {content}".strip()
    append_current()
    return results[:10]


def parse_reading_summary(text: str) -> str:
    """Keep the free-form summary separate from colon or Markdown question blocks."""
    match = re.search(
        r"(?ims)^\s*(?:\*{0,2}\s*)?(?:(?:paper|research) summary|연구\s*(?:서머리|요약)|논문\s*요약)\s*(?:\*{0,2})\s*:?\s*"
        r"(.*?)(?=\n\s*(?:---+\s*$|(?:\*{0,2}\s*)?(?:question|질문)\s*\d*\s*(?:\*{0,2})\s*(?::|\n))|\Z)",
        text,
    )
    if match:
        return match.group(1).strip()
    # Some local models start immediately with a paper title and summary,
    # then introduce blocks as "연구자 판단이 필요한 질문". Keep that useful
    # summary rather than discarding it merely because its heading was omitted.
    question_start = re.search(
        r"(?im)^\s*(?:\*{0,2}\s*)?(?:(?:question|질문)|연구자\s*판단(?:이\s*필요한)?\s*질문)\s*(?:\*{0,2})\s*:?",
        text,
    )
    prefix = text[:question_start.start()].strip() if question_start else ""
    return prefix.strip("- \n") if len(prefix) >= 80 else ""


def parse_second_pass_reviews(text: str) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    aliases = {"id": "question_id", "refined answer": "refined_answer", "additional evidence": "additional_evidence", "remaining uncertainty": "remaining_uncertainty"}
    for block in re.split(r"(?m)^---+\s*$", text):
        values: dict[str, str] = {}
        current = ""
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                current = aliases.get(key.strip().lower(), "")
                if current:
                    values[current] = value.strip()
            elif current and line.strip():
                values[current] = f"{values.get(current, '')} {line.strip(' -*•')}".strip()
        evidence = [value.strip(" -*•") for value in values.get("additional_evidence", "").split(";") if value.strip(" -*•")]
        if values.get("question_id") and len(values.get("refined_answer", "")) >= 12:
            reviews.append({"question_id": values["question_id"], "refined_answer": values["refined_answer"], "additional_evidence": evidence[:5], "remaining_uncertainty": values.get("remaining_uncertainty", "")})
    return reviews
