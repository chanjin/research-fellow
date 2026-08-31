"""Paper-level reading prompts; no draft here is approved knowledge."""

from __future__ import annotations

import re
import uuid
from typing import Any

from research_fellow.domain.knowledge import KnowledgeCard
from research_fellow.infrastructure.document_reader import ExtractedDocument
from research_fellow.storage import Ledger


def reading_prompt(document: ExtractedDocument, paper: dict[str, Any], context: str) -> str:
    source = "\n\n".join(f"[p.{page.page_number}] {page.text[:2200]}" for page in document.pages[:10])[:20000]
    return f"""You are an academic reading assistant. Read only the source text below.
For at most four items, write Korean in this exact block format, separated by ---.
First, write a substantial, evidence-grounded Korean research summary, then write the question blocks.
Research summary:
Write freely in several paragraphs (roughly 800–1,500 Korean characters when the source supports it). Explain the research problem, motivation, method and material, key observations/results, the authors' interpretation, research significance, and limits. Do not force a fixed list format. Keep clear distinctions between the paper's findings and your cautious interpretation.

Question: a question that requires the researcher's judgment
Tentative answer: an interpretation grounded only in this paper
Evidence: page and short source hint; page and short source hint
Uncertainty: scope or evidence limitation

Paper: {paper['title']}
Research context: {context or 'not supplied'}
Source text:
{source}

Output check before responding:
- Use Korean only.
- First write the free-form Research summary, then the question blocks. Do not write content outside those two sections.
- Return at most four blocks separated by ---.
- Every block must contain Question, Tentative answer, Evidence, and Uncertainty.
- Every Evidence value must include p.N and a short source hint.
"""


def parse_reading_questions(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    aliases = {
        "question": "question", "질문": "question",
        "연구자 판단이 필요한 질문": "question", "연구자 판단 질문": "question", "판단이 필요한 질문": "question",
        "tentative answer": "tentative_answer", "잠정 답변": "tentative_answer", "잠정적 답변": "tentative_answer",
        "evidence": "evidence", "근거": "evidence", "증거": "evidence",
        "uncertainty": "uncertainty", "불확실성": "uncertainty", "유보": "uncertainty",
    }
    def field_name(raw_key: str) -> str:
        """Accept both `Question: text` and Markdown heading forms.

        Local models often emit `**질문 1**` followed by a question on the
        next line, even when asked for a colon-delimited block.  This is still
        a well-grounded reading response, so parsing must be permissive while
        validation below remains strict.
        """
        normalized = re.sub(r"\s*\d+\s*$", "", re.sub(r"^[\s#*\-•]+|[\s#*]+$", "", raw_key)).lower()
        return aliases.get(normalized, "")

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
            results.append({"question": values["question"], "tentative_answer": values["tentative_answer"], "evidence": evidence[:4], "uncertainty": values.get("uncertainty", "원문 범위를 넘어선 일반화는 유보합니다.")})

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
            labelled_field = field_name(key)
        heading_field = field_name(stripped)
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
    return results[:4]


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


def second_pass_prompt(document: ExtractedDocument, paper: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    """Re-read broader paper context to test and enrich the first-pass interpretations."""
    question_text = "\n".join(f"ID: {item['question_id']}\nQuestion: {item['question']}\nFirst answer: {item['tentative_answer']}\nFirst evidence: {'; '.join(item['evidence'])}" for item in questions)
    source = "\n\n".join(f"[p.{page.page_number}] {page.text[:2000]}" for page in document.pages[:16])[:32000]
    return f"""You are conducting a second, question-driven reading of one academic paper.
Re-read the source text to verify, correct, and enrich each first-pass answer below. Do not write a paper summary and do not add facts outside the source.

Paper: {paper['title']}
First-pass questions:
{question_text}

For every ID, return one block separated by ---:
ID: exact ID
Refined answer: improved, research-useful answer; correct the first answer if needed
Additional evidence: p.N short source hint; p.N short source hint
Remaining uncertainty: what this paper still cannot establish

Source text for second pass:
{source}

Output check: Korean only; one block for every supplied ID; no essay; every additional evidence item contains p.N.
"""


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


def promote_question(ledger: Ledger, paper: dict[str, Any], item: dict[str, Any], comment: str) -> str:
    """Turn a reviewed reading interpretation into a claim candidate, not a question card."""
    card = KnowledgeCard(
        card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}", title=item["question"][:72], source_kind="external_paper",
        claim=item["tentative_answer"], explanation=comment, labels=[], evidence_excerpt="\n".join(item["evidence"])[:1600],
        evidence_pages=[], citation_markers=[], conditions="not_assessed", limits=item["uncertainty"],
        provenance={
            "source_name": paper["title"], "paper_id": paper["paper_id"], "grounding": "paper_reading_review",
            "reading_question": item["question"],
        },
    ).model_dump(mode="json")
    case_id = ledger.create_case("research", f"Paper reading promotion: {paper['title'][:72]}")
    return ledger.record(case_id, "decision_request", "m1", ["researcher"], "knowledge_card", {
        "title": f"논문 읽기 기반 주장(Claim) 후보 승인: {card['title']}", "card": card,
        "paper_id": paper["paper_id"], "reading_question_id": item["question_id"],
        "next_action": "원문 근거와 연구자 첨삭을 확인한 뒤 승인 또는 보완 요청",
    }, subject_id=card["card_id"])


def promote_ontology_candidate(ledger: Ledger, paper: dict[str, Any], candidate: dict[str, Any], comment: str) -> str:
    """Request researcher approval for an abstraction before it enters ontology work."""
    case_id = ledger.create_case("research", f"Paper ontology proposal: {paper['title'][:72]}")
    payload = {
        "candidate_id": candidate["candidate_id"],
        "paper_id": paper["paper_id"],
        "paper_title": paper["title"],
        "statement": candidate["candidate_text"],
        "evidence": candidate["evidence"],
        "researcher_comment": comment,
    }
    return ledger.record(case_id, "decision_request", "m1", ["researcher"], "ontology_candidate", {
        "title": f"논문 기반 온톨로지 후보 승인: {candidate['candidate_text'][:72]}",
        "ontology_candidate": payload,
        "next_action": "일반화가 원문 근거와 연구 주제에 맞는지 검토하고, 승인 시 온톨로지 정리 대상으로 보냅니다.",
    }, subject_id=candidate["candidate_id"])
