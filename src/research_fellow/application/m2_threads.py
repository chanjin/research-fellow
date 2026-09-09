from __future__ import annotations

import json
import re
from typing import Any


def build_m2_thread_review_prompt(
    *,
    question: str,
    context: str,
    cards: list[dict[str, Any]],
    source_type: str,
    source_payload: dict[str, Any] | None = None,
) -> str:
    source_payload = source_payload or {}
    evidence_lines: list[str] = []
    for card in cards:
        provenance = card.get("provenance") or {}
        evidence_lines.append(
            "\n".join(
                [
                    f"### {card.get('card_id', '')} · {card.get('title', '')}",
                    f"Claim: {card.get('claim', '')}",
                    f"Conditions: {card.get('conditions', '')}",
                    f"Limits: {card.get('limits', '')}",
                    f"Source: {provenance.get('source_name', provenance.get('source_id', ''))}",
                ]
            )
        )
    source_note = json.dumps(source_payload, ensure_ascii=False, indent=2) if source_payload else "{}"
    return f"""# M2 Knowledge-based Review

You are the M2 domain advisory agent. Answer the research question using ONLY the approved knowledge cards below as factual evidence.
Separate supported findings from your interpretation. If the evidence is insufficient, say exactly what is missing and propose literature supplementation topics.

## Question
{question.strip()}

## Question Source
{source_type}
{source_note}

## Research Context / Researcher Comment
{context.strip() or '(none)'}

## Approved Knowledge Cards
{chr(10).join(evidence_lines) if evidence_lines else '(no approved evidence selected)'}

## Required Report Structure
### Current Answer
A concise answer to the question based on the selected cards.

### Evidence
For each important point, cite the supporting card ID such as [kc-...].

### Interpretation and Implications
Explain what follows from the evidence for the current research context. Clearly mark inference as interpretation.

### Constraints and Unresolved Issues
List conditions, limitations, contradictions, or unresolved issues.

### Knowledge Gaps / M1 Supplementation Need
If more literature is required, state concrete missing evidence and search directions. If no supplementation is needed, write "None".

### Question Refinement
If the question should be narrowed, split, or made more precise, propose one refined question. Otherwise write "No change".
"""


def validate_manual_m2_report(text: str, *, valid_card_ids: set[str] | None = None) -> tuple[bool, str]:
    cleaned = (text or "").strip()
    if len(cleaned) < 120:
        return False, "응답이 너무 짧습니다. 질문에 대한 답변·근거·한계가 포함된 전체 보고서를 붙여넣으세요."
    required_any = ["Current Answer", "현재", "Evidence", "근거"]
    if not any(token.lower() in cleaned.lower() for token in required_any):
        return False, "답변 또는 근거 섹션을 확인할 수 없습니다. 제공된 복구 프롬프트의 보고서 구조를 유지해주세요."
    if valid_card_ids:
        cited = set(re.findall(r"\[(kc-[^\]\s]+)\]", cleaned))
        invalid = sorted(cited - valid_card_ids)
        if invalid:
            return False, f"선택한 근거카드에 없는 ID가 인용되었습니다: {', '.join(invalid[:5])}"
    return True, "외부 LLM 응답 형식을 확인했습니다."


def extract_knowledge_gaps(report: str) -> str:
    text = report or ""
    patterns = [
        r"(?is)###\s*Knowledge Gaps\s*/\s*M1 Supplementation Need\s*\n(.*?)(?=\n###|\Z)",
        r"(?is)###\s*Knowledge Gaps.*?\n(.*?)(?=\n###|\Z)",
        r"(?is)###\s*지식 공백.*?\n(.*?)(?=\n###|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value.lower() not in {"none", "없음", "해당 없음", "no"}:
                return value
    return ""


def extract_refined_question(report: str) -> str:
    text = report or ""
    patterns = [
        r"(?is)###\s*Question Refinement\s*\n(.*?)(?=\n###|\Z)",
        r"(?is)###\s*질문 구체화.*?\n(.*?)(?=\n###|\Z)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip().strip('"')
            if value.lower() not in {"no change", "none", "변경 없음", "없음"}:
                return value.splitlines()[0].strip(" -•")
    return ""
