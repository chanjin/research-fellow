"""Prompt construction for M1 paper reading and evidence investigation."""
from __future__ import annotations
from typing import Any
from research_fellow.infrastructure.document_reader import ExtractedDocument
from research_fellow.infrastructure.prompt_renderer import apply_review_language_policy

def reading_prompt(document: ExtractedDocument, paper: dict[str, Any], context: str) -> str:
    source = "\n\n".join(f"[p.{page.page_number}] {page.text[:2200]}" for page in document.pages[:10])[:20000]
    return apply_review_language_policy(f"""You are an academic reading assistant. Read only the source text below.
Write researcher-facing sentences in Korean in this exact block format, separated by ---. Return one to five high-value question blocks by default; return more only when each item has distinct, sufficient evidence. Never return more than ten blocks.
First, write a substantial, evidence-grounded Korean research summary, then an M1 interpretation for the supplied research context, suggested shelf labels, and the question blocks.
Research summary:
Start with this separate three-part overview so a researcher can understand the paper at a glance:
- 대상 문제: what concrete problem, gap, or decision the paper addresses, and for whom or in which setting it matters
- 해결 접근: how the paper addresses it; name the method, system, data/material, comparison, or reasoning approach actually used
- 핵심 결과: the main observed result, effect, capability, or negative finding; distinguish reported evidence from author interpretation

Write freely in several paragraphs (roughly 800–1,500 Korean characters when the source supports it). Explain the research problem, motivation, method and material, key observations/results, the authors' interpretation, research significance, and limits. Do not force a fixed list format. Keep clear distinctions between the paper's findings and your cautious interpretation.

M1 research-context interpretation:
- Explain how this paper can contribute to the stated research context.
- State what the paper cannot establish for that context.
- Include at least one counterpoint, application condition, or boundary of applicability.

Suggested shelf labels:
Labels: up to ten concise English labels, separated by commas. Cover topic, method, evidence type, or application context where supported. Do not use generic labels such as paper or AI.

Use these exact Korean field labels in every question block. Do not omit a field; when the source is insufficient, write "원문에서 확인 필요" rather than leaving it blank.
질문: 연구자의 판단이 필요한 질문
잠정 답변: 이 논문에만 근거한 해석
근거: 독립적으로 확인 가능한 p.N과 짧은 원문 단서 2~5개. 논문의 핵심 맥락을 더 잘 이해하는 데 도움이 되는 서로 다른 위치를 우선하고, 한 곳만 가능하면 그 사실을 한계·유보에 명시
한계·유보: 적용 범위 또는 근거의 한계
연구 관련성: 이 질문이 알리는 현재 가설·설계 선택·평가 쟁점·탐색 방향
레이블: 간결한 영문 레이블, 쉼표 구분
카드 제목: Claim을 반복하지 않는 짧은 한국어 명사구
핵심 개념: 이후 관계 작업에 쓸 도메인 개념, 쉼표 구분
적용 대상: Claim이 다루는 객체·상황·과업, 쉼표 구분
적용 조건: 원문에 근거한 전제·관찰 범위·설계 제약
카드 맥락: 이 주장이 원 논문의 어떤 과업·비교·문제 설정에서 나온 것인지 1~2문장. 현재 연구 프로젝트, 입력된 Research context, 숏페이퍼 또는 Revision To-do를 언급하지 말 것
설계 함의: 이 결과가 연구·설계·메모리/검색 선택 또는 의사결정에 주는 의미를 1~2문장
주변 원문: 위 주장을 해석하는 데 필요한 짧은 원문 주변 문맥. 가능하면 p.N 표시 포함

Paper: {paper['title']}
Research context: {context or 'not supplied'}
Source text:
{source}

Output check before responding:
- Use Korean for narrative sentences. Keep 레이블, 핵심 개념, 적용 대상 values as concise English terms/phrases.
- First write Research summary beginning with 대상 문제, 해결 접근, 핵심 결과; then M1 research-context interpretation, Suggested shelf labels, then the question blocks. Do not write content outside these sections.
- Return one to five complete question blocks by default, and never more than ten, separated by ---.
- Every block must contain exactly these Korean field labels: 질문, 잠정 답변, 근거, 한계·유보, 연구 관련성, 레이블, 카드 제목, 핵심 개념, 적용 대상, 적용 조건, 카드 맥락, 설계 함의, 주변 원문.
- Every Evidence value must include p.N and a short source hint.
- Card context must remain an independent, source-intrinsic description of this paper. Keep project-specific usefulness only in Research relevance; never copy the supplied Research context into Card context.
- Return 2 to 5 independent evidence locations when the source supports them. Never return more than 5 evidence locations for one question.
- Prefer fewer complete blocks to an incomplete response. Keep every non-evidence field concise (one or two sentences).
""")


def second_pass_prompt(document: ExtractedDocument, paper: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    """Re-read broader paper context to test and enrich the first-pass interpretations."""
    question_text = "\n".join(f"ID: {item['question_id']}\nQuestion: {item['question']}\nFirst answer: {item['tentative_answer']}\nFirst evidence: {'; '.join(item['evidence'])}" for item in questions)
    source = "\n\n".join(f"[p.{page.page_number}] {page.text[:2000]}" for page in document.pages[:16])[:32000]
    return apply_review_language_policy(f"""You are conducting a second, question-driven reading of one academic paper.
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

Output check: Korean narrative sentences; keep short taxonomy/tag terms in English; one block for every supplied ID; no essay; every additional evidence item contains p.N.
""")
