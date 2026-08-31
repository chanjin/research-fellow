"""Safe editing and validation of the project's versioned Jinja prompt assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError

from research_fellow.infrastructure.prompt_renderer import PROMPT_DIR, prompt_environment


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    label: str
    purpose: str


PROMPT_CATALOG = (
    PromptTemplate("m1_curation.j2", "M1 · PDF/노트 핵심 주장", "본문에서 독립 주장과 원문 발췌를 만드는 초안"),
    PromptTemplate("m1_claim_discovery.j2", "M1 · 주장 발견", "제약 없이 최대 10개 핵심 주장을 찾는 초안"),
    PromptTemplate("m1_claim_labels.j2", "M1 · 주장 레이블", "주장별 핵심 레이블만 제안하는 초안"),
    PromptTemplate("m1_claim_consolidation.j2", "M1 · 주장 정리", "발견 주장의 중복·주변성만 판단하는 초안"),
    PromptTemplate("m1_claim_qualification.j2", "M1 · 주장 카드 보강", "정리된 주장에 근거·조건·한계를 부여하는 초안"),
    PromptTemplate("m1_page_curation.j2", "M1 · 점진적 구간 카드", "작은 문맥의 텍스트 구간별 후보 카드 초안"),
    PromptTemplate("m1_candidate_consolidation.j2", "M1 · 후보 통합 판단", "독립·중복·포함 관계의 LLM 판단"),
    PromptTemplate("m1_relation_proposal.j2", "M1 · 관계 후보", "승인 카드 두 장의 관계 초안"),
    PromptTemplate("m1_claim_verification.j2", "M1 · 주장 검증", "승인 지식으로 특정 주장을 검토"),
    PromptTemplate("m1_gap_search_plan.j2", "M1 · 공백 탐색 계획", "지식 공백을 위한 검색 쿼리 초안"),
    PromptTemplate("m1_source_triage.j2", "M1 · 문헌 후보 선별", "도구가 찾은 문헌 메타데이터 검토"),
    PromptTemplate("m1_paper_shelf_analysis.j2", "M1 · 중요 논문 서재 분석", "논문 본문과 연구 맥락을 분리해 읽기용 분석 요약 생성"),
    PromptTemplate("m1_paper_reading_questions.j2", "M1 · 논문 읽기 질문", "원문 근거와 함께 연구자 첨삭을 요청하는 논문별 읽기 질문"),
    PromptTemplate("m1_lineage_review.j2", "M1 · 계보 검토", "승인 지식의 계보·관계 후보 검토"),
    PromptTemplate("m1_lineage_overview.j2", "M1 · 계보 종합 의견", "승인 관계 그래프의 중심 주제·흐름·공백 해석"),
    PromptTemplate("m1_revalidation_review.j2", "M1 · 재검증", "상충·출처 재확인 후보 검토"),
    PromptTemplate("m2_research_review.j2", "M2 · 연구 검토", "연구 질문의 근거 기반 검토"),
    PromptTemplate("m2_research_direction.j2", "M2 · 연구 상태·방향", "ResearchState 기반 검토와 M1 Intent 후보"),
    PromptTemplate("m2_research_question_suggestions.j2", "M2 · 신규 연구질문 추천", "M1 새 정보와 기존 질문을 바탕으로 한 연구자 선택지"),
    PromptTemplate("m2_research_context_mapping.j2", "M2 · 연구자 메모 정리", "자유 연구자 메모를 ResearchState 보조 항목으로 정리"),
    PromptTemplate("m2_knowledge_update_report.j2", "M2 · 지식 업데이트 보고", "승인 지식 변화의 연구자 보고"),
    PromptTemplate("m2_meaning_summary.j2", "M2 · 연구 활동 의미 요약", "승인 지식·관계·M2 보고서를 사실 중심으로 묶는 읽기 전용 요약"),
    PromptTemplate("m2_delta_meaning_summary.j2", "M2 · 연구 활동 Delta 요약", "마지막 저장 요약 이후 변화만 기존 지식 맥락에서 해석"),
    PromptTemplate("m2_search_keywords.j2", "M2 · 탐색 키워드 재생성", "승인된 탐색 맥락을 바탕으로 M1 논문 탐색용 키워드를 다시 제안"),
    PromptTemplate("m2_abstract_relevance.j2", "M2 · 초록 맥락 적합성", "M1 후보 논문의 초록을 승인된 탐색 맥락과 대조해 후보 적합성만 선별"),
    PromptTemplate("m2_external_interpretation.j2", "M2 · 외부 요청 해석", "외부 자문 범위 확인 초안"),
    PromptTemplate("m2_external_response.j2", "M2 · 외부 자문 답변", "근거·조건·한계를 분리한 외부 답변"),
    PromptTemplate("m2_advisory_plan.j2", "M2 · 자문 답변 계획", "최종 답변 전 판단 질문과 하위 질문을 구성"),
    PromptTemplate("m2_subquestion_judgment.j2", "M2 · 하위 질문 판단", "관계 기반 지식 클러스터로 하나의 하위 질문을 검토"),
    PromptTemplate("m2_advisory_synthesis.j2", "M2 · 자문 통합", "하위 판단을 근거·조건·한계가 보이는 최종 답변으로 통합"),
)

_ALLOWED = {item.name for item in PROMPT_CATALOG}


def template_path(name: str) -> Path:
    if name not in _ALLOWED:
        raise ValueError("등록된 Jinja 프롬프트만 편집할 수 있습니다.")
    return PROMPT_DIR / name


def read_prompt_template(name: str) -> str:
    return template_path(name).read_text(encoding="utf-8")


def validate_prompt_template(source: str) -> None:
    """Compile without executing; missing variables remain a render-time error."""
    if not source.strip():
        raise ValueError("프롬프트는 비어 있을 수 없습니다.")
    try:
        Environment().parse(source)
    except TemplateSyntaxError as error:
        raise ValueError(f"Jinja 문법 오류 (줄 {error.lineno}): {error.message}") from error


def render_prompt_source(source: str, **context: object) -> str:
    """Render the current editor value, including an unsaved development draft."""
    validate_prompt_template(source)
    try:
        return Environment(
            undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True
        ).from_string(source).render(**context).strip()
    except Exception as error:
        raise ValueError(f"프롬프트 문맥 렌더링 오류: {error}") from error


def save_prompt_template(name: str, source: str) -> None:
    """Validate first, then atomically replace one versioned prompt asset."""
    validate_prompt_template(source)
    target = template_path(name)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(source.rstrip() + "\n", encoding="utf-8")
    temporary.replace(target)
    # Existing Jinja environments cache named templates. Clear it after a save
    # so both the playground and the production screen read the edited file.
    prompt_environment().cache.clear()
