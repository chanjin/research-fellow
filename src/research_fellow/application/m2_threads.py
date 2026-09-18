from __future__ import annotations

import json
import re
from typing import Any
from research_fellow.origin_lineage import origin_labels


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
                    f"Origin lineage: {'; '.join(origin_labels(card.get('origin_links', []))) or 'none'}",
                ]
            )
        )
    source_note = json.dumps(source_payload, ensure_ascii=False, indent=2) if source_payload else "{}"
    return f"""# M2 Research Decision Brief

You are the M2 Research Direction and Argument Advisor. Create a Research Decision Brief using ONLY the approved knowledge cards below as factual evidence.
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

## Required Brief Structure
### 판단할 논점
State the decision or clarification needed now, not merely the incoming question.

### 현시점 권고
Give one concise, qualified recommendation. If evidence is insufficient, explicitly recommend withholding judgment.

### 확인된 근거
For each important point, cite supporting card IDs such as [kc-...]. Separate facts from interpretation.

### 해석과 선택지
Explain implications for the current research context and present viable alternatives where relevant.

### 반론·한계·숨은 가정
List conditions, limitations, contradictions, counterarguments, and hidden assumptions.

### 다음 행동
Choose one: researcher confirmation, M1 evidence supplementation, external fact checking, or no further action. If M1 supplementation is needed, state concrete missing evidence and search directions. If it is not needed, write "None".

### 질문 구체화
If the question should be narrowed, split, or made more precise, propose one refined question. Otherwise write "No change".
"""


def validate_manual_m2_report(text: str, *, valid_card_ids: set[str] | None = None) -> tuple[bool, str]:
    cleaned = (text or "").strip()
    if len(cleaned) < 120:
        return False, "응답이 너무 짧습니다. 질문에 대한 답변·근거·한계가 포함된 전체 보고서를 붙여넣으세요."
    required_any = ["Current Answer", "현재", "Evidence", "근거", "판단할 논점", "현시점 권고"]
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
