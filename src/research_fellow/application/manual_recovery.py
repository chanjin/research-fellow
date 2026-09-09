"""External-LLM manual recovery helpers for failed long-running stages."""
from __future__ import annotations

import json
from typing import Any

from research_fellow.application.advising import parse_research_question_suggestions, parse_rq_priority_assessment
from research_fellow.application.search_profiles import parse_auto_search_strategy, parse_keyword_plan
from research_fellow.application.llm_retry import classify_output


def external_recovery_prompt(failure: dict[str, Any]) -> str:
    context = dict(failure.get("context") or {})
    prompt = str(context.get("recovery_prompt") or "").strip()
    if prompt:
        return prompt
    return (
        "아래 실패 작업을 외부 LLM에서 대신 수행하세요.\n"
        f"Stage: {failure.get('stage', '')}\n"
        f"Error: {failure.get('error_type', '')} - {failure.get('error_message', '')}\n\n"
        "필수 출력 형식을 끝까지 완성하고 설명문보다 요구된 결과를 우선하세요."
    )


def _parse_abstract_lines(text: str) -> dict[str, tuple[str, str]]:
    parsed: dict[str, tuple[str, str]] = {}
    for raw in text.splitlines():
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) != 3:
            continue
        level = parts[1].lower()
        if level not in {"high", "medium", "low"}:
            continue
        parsed[parts[0]] = (level, parts[2])
    return parsed


def validate_external_response(stage: str, response: str, context: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Validate an externally pasted response before it is allowed to replace an LLM call."""
    context = dict(context or {})
    ok, error_type, message = classify_output(response, validator=lambda value: len(value.strip()) >= 10)
    if not ok:
        return False, f"{error_type}: {message}"
    text = response.strip()
    try:
        if stage == "search_strategy":
            plan = parse_auto_search_strategy(text)
            if not (plan.get("phrases") or plan.get("queries")):
                return False, "검색 phrase 또는 Boolean query를 찾지 못했습니다."
        elif stage == "search_strategy_legacy":
            phrases, _ = parse_keyword_plan(text)
            if not phrases:
                return False, "영문 검색키워드를 찾지 못했습니다."
        elif stage == "abstract_screening":
            parsed = _parse_abstract_lines(text)
            expected = [str(item) for item in context.get("source_ids", []) if str(item)]
            if not parsed:
                return False, "`source_id | high|medium|low | 이유` 형식의 평가를 찾지 못했습니다."
            missing = [source_id for source_id in expected if source_id not in parsed]
            if missing:
                return False, f"현재 batch 논문 {len(missing)}건의 평가가 빠졌습니다: {', '.join(missing[:5])}"
        elif stage == "rq_generation":
            valid_ids = {str(item) for item in context.get("source_card_ids", []) if str(item)}
            if not parse_research_question_suggestions(text, valid_card_ids=valid_ids or None, limit=20):
                return False, "유효한 연구질문 후보를 파싱하지 못했습니다."
        elif stage == "rq_prioritization":
            actionable = list(context.get("actionable_questions") or [])
            if actionable and not parse_rq_priority_assessment(text, actionable, limit=3):
                return False, "연구질문 중요도 평가 결과를 파싱하지 못했습니다."
        elif stage in {"fulltext_review", "synthesis"}:
            if len(text) < 20:
                return False, "응답이 너무 짧습니다."
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return False, str(error)
    return True, "외부 응답 형식이 유효합니다."
