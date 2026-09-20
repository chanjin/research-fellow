"""Deterministic Short Paper specification, validation, and freeze helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


WORKFLOW_STAGE_LABELS = {
    "project_setup": "프로젝트 설정",
    "evidence_building": "근거 구성",
    "short_drafting": "숏페이퍼 초안",
    "revision": "리비전",
    "validation": "마일스톤 검증",
    "short_paper_frozen": "Short Paper 동결",
    "full_paper_expanding": "Full Paper 확장",
    "completed": "완료",
}


def default_writing_spec(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec_version": 0,
        "target_format": "two_page_short_paper",
        "audience": "에이전트·소프트웨어 공학 연구자와 현업 엔지니어",
        "central_claim": "",
        "target_min_chars": 4000,
        "target_max_chars": 9000,
        "required_section_terms": ["서론", "결론"],
        "required_card_ids": list(dict.fromkeys(str(value) for value in project.get("origin_ids", []) if str(value))),
        "writing_rules": [],
        "open_decisions": [],
        "research_method": "",
        "research_method_rationale": "",
        "verification_plan": [],
        "execution_guide": [],
        "result_recording_plan": [],
        "literature_search_candidates": [],
    }


def normalize_writing_spec(raw: dict[str, Any] | None, project: dict[str, Any]) -> dict[str, Any]:
    result = default_writing_spec(project)
    result.update(dict(raw or {}))
    minimum = max(100, int(result.get("target_min_chars") or 4000))
    maximum = max(minimum, int(result.get("target_max_chars") or 9000))
    result["target_min_chars"] = minimum
    result["target_max_chars"] = maximum
    for field in ("required_section_terms", "required_card_ids", "writing_rules", "open_decisions"):
        values = result.get(field) or []
        if isinstance(values, str):
            values = values.splitlines()
        result[field] = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    for field in ("verification_plan", "result_recording_plan", "literature_search_candidates"):
        values = result.get(field) or []
        result[field] = [dict(value) for value in values if isinstance(value, dict)]
    values = result.get("execution_guide") or []
    if isinstance(values, str):
        values = values.splitlines()
    result["execution_guide"] = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    result["central_claim"] = str(result.get("central_claim") or "").strip()
    result["audience"] = str(result.get("audience") or "").strip()
    result["research_method"] = str(result.get("research_method") or "").strip()
    result["research_method_rationale"] = str(result.get("research_method_rationale") or "").strip()
    result["target_format"] = "two_page_short_paper"
    result["spec_version"] = max(0, int(result.get("spec_version") or 0))
    return result


def latest_event_payload(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    event = next((item for item in reversed(events) if item.get("event_type") == event_type), None)
    return dict((event or {}).get("payload") or {}) if event else None


def manuscript_plain_text(manuscript: dict[str, Any] | None) -> str:
    if not manuscript:
        return ""
    parts = [str(manuscript.get("title") or "")]
    for section in manuscript.get("sections") or []:
        parts.append(str(section.get("title") or ""))
        for paragraph in section.get("paragraphs") or []:
            parts.extend(str(sentence.get("text") or "") for sentence in paragraph.get("sentences") or [])
    for item in manuscript.get("appendix_claims") or []:
        parts.extend([str(item.get("title") or ""), str(item.get("claim") or "")])
    return "\n".join(value for value in parts if value.strip())


def manuscript_fingerprint(manuscript: dict[str, Any] | None) -> str:
    canonical = json.dumps(manuscript or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def spec_fingerprint(spec: dict[str, Any] | None) -> str:
    canonical = json.dumps(spec or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sentences(manuscript: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [
        sentence
        for section in (manuscript or {}).get("sections") or []
        for paragraph in section.get("paragraphs") or []
        for sentence in paragraph.get("sentences") or []
    ]


def validate_short_paper(
    manuscript: dict[str, Any] | None,
    spec: dict[str, Any],
    *,
    valid_card_ids: set[str],
    valid_reference_paper_ids: set[str],
    todos: list[dict[str, Any]],
    reference_count: int,
) -> dict[str, Any]:
    """Run reproducible structural checks without pretending to measure exact pages."""
    normalized = normalize_writing_spec(spec, {})
    text = manuscript_plain_text(manuscript)
    compact_chars = len(re.sub(r"\s+", "", text))
    sentences = _sentences(manuscript)
    sentence_ids = [str(item.get("sentence_id") or "") for item in sentences]
    section_titles = [str(item.get("title") or "") for item in (manuscript or {}).get("sections") or []]
    used_card_ids = {
        str(card_id)
        for sentence in sentences
        for card_id in sentence.get("evidence_card_ids") or []
    }
    used_card_ids.update(
        str(card_id)
        for item in (manuscript or {}).get("appendix_claims") or []
        for card_id in item.get("evidence_card_ids") or []
    )
    used_reference_paper_ids = {
        str(paper_id)
        for sentence in sentences
        for paper_id in sentence.get("reference_paper_ids") or []
    }
    unknown_reference_paper_ids = sorted(
        paper_id for paper_id in used_reference_paper_ids if paper_id not in valid_reference_paper_ids
    )
    critical_without_evidence = [
        str(item.get("sentence_id") or "")
        for item in sentences
        if str(item.get("role") or "") in {"central_claim", "evidence"}
        and not item.get("evidence_card_ids")
        and not item.get("reference_paper_ids")
    ]
    missing_sections = [
        term for term in normalized["required_section_terms"]
        if not any(term.casefold() in title.casefold() for title in section_titles)
    ]
    missing_required_cards = [
        card_id for card_id in normalized["required_card_ids"]
        if card_id not in used_card_ids
    ]
    unknown_cards = sorted(card_id for card_id in used_card_ids if card_id not in valid_card_ids)
    open_p0 = [item for item in todos if item.get("priority") == "P0" and item.get("status") != "resolved"]

    checks = [
        {"check_id": "manuscript", "label": "원고 존재", "severity": "blocker", "passed": bool(manuscript and sentences), "detail": f"문장 {len(sentences)}개"},
        {"check_id": "title", "label": "논문 제목", "severity": "blocker", "passed": bool(str((manuscript or {}).get("title") or "").strip()), "detail": str((manuscript or {}).get("title") or "제목 없음")},
        {"check_id": "required_sections", "label": "필수 섹션", "severity": "blocker", "passed": not missing_sections, "detail": "누락: " + ", ".join(missing_sections) if missing_sections else f"{len(section_titles)}개 섹션 확인"},
        {"check_id": "minimum_length", "label": "최소 분량 예산", "severity": "blocker", "passed": compact_chars >= normalized["target_min_chars"], "detail": f"{compact_chars:,}/{normalized['target_min_chars']:,}자"},
        {"check_id": "maximum_length", "label": "최대 분량 예산", "severity": "blocker", "passed": compact_chars <= normalized["target_max_chars"], "detail": f"{compact_chars:,}/{normalized['target_max_chars']:,}자"},
        {"check_id": "sentence_ids", "label": "문장 ID 무결성", "severity": "blocker", "passed": bool(sentence_ids) and all(sentence_ids) and len(sentence_ids) == len(set(sentence_ids)), "detail": f"전체 {len(sentence_ids)}개 · 고유 {len(set(sentence_ids))}개"},
        {"check_id": "card_ids", "label": "근거카드 ID 무결성", "severity": "blocker", "passed": not unknown_cards, "detail": "알 수 없는 카드: " + ", ".join(unknown_cards) if unknown_cards else f"연결 카드 {len(used_card_ids)}건"},
        {"check_id": "reference_ids", "label": "인용 논문 ID 무결성", "severity": "blocker", "passed": not unknown_reference_paper_ids, "detail": "References 미등록 논문: " + ", ".join(unknown_reference_paper_ids) if unknown_reference_paper_ids else f"문장 연결 논문 {len(used_reference_paper_ids)}편"},
        {"check_id": "required_cards", "label": "필수 카드 반영", "severity": "blocker", "passed": not missing_required_cards, "detail": "미반영: " + ", ".join(missing_required_cards) if missing_required_cards else f"필수 카드 {len(normalized['required_card_ids'])}건 반영"},
        {"check_id": "critical_evidence", "label": "핵심 주장 근거", "severity": "blocker", "passed": not critical_without_evidence, "detail": "근거 없음: " + ", ".join(critical_without_evidence) if critical_without_evidence else "핵심 주장·근거 문장 연결 완료"},
        {"check_id": "p0_todos", "label": "P0 Revision To-do", "severity": "blocker", "passed": not open_p0, "detail": f"열린 P0 {len(open_p0)}건"},
        {"check_id": "researcher_decisions", "label": "연구자 결정", "severity": "blocker", "passed": not normalized["open_decisions"], "detail": "미결정: " + " | ".join(normalized["open_decisions"]) if normalized["open_decisions"] else "열린 결정 없음"},
        {"check_id": "references", "label": "References", "severity": "warning", "passed": reference_count > 0, "detail": f"등록 문헌 {reference_count}편"},
        {"check_id": "central_claim", "label": "집필 명세의 중심 주장", "severity": "warning", "passed": bool(normalized["central_claim"]), "detail": normalized["central_claim"] or "중심 주장 미입력"},
    ]
    blockers = [item for item in checks if item["severity"] == "blocker" and not item["passed"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["passed"]]
    return {
        "passed": not blockers,
        "checks": checks,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "manuscript_version": int((manuscript or {}).get("version") or 0),
        "manuscript_hash": manuscript_fingerprint(manuscript),
        "spec_version": normalized["spec_version"],
        "spec_hash": spec_fingerprint(normalized),
        "compact_char_count": compact_chars,
    }


def frozen_milestone(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    frozen_positions = [index for index, item in enumerate(events) if item.get("event_type") == "short_paper_frozen"]
    if not frozen_positions:
        return None
    latest_frozen_position = frozen_positions[-1]
    latest_unfrozen_position = next(
        (index for index in range(len(events) - 1, -1, -1) if events[index].get("event_type") == "short_paper_unfrozen"),
        -1,
    )
    if latest_unfrozen_position > latest_frozen_position:
        return None
    return dict(events[latest_frozen_position].get("payload") or {})


def project_workflow_stage(project: dict[str, Any]) -> str:
    events = list(project.get("events") or [])
    if project.get("status") == "completed":
        return "completed"
    if project.get("stage") == "full_paper":
        return "full_paper_expanding"
    if frozen_milestone(events):
        return "short_paper_frozen"
    latest_frozen_position = next(
        (index for index in range(len(events) - 1, -1, -1) if events[index].get("event_type") == "short_paper_frozen"),
        -1,
    )
    latest_unfrozen_position = next(
        (index for index in range(len(events) - 1, -1, -1) if events[index].get("event_type") == "short_paper_unfrozen"),
        -1,
    )
    if latest_unfrozen_position > latest_frozen_position >= 0:
        return "revision"
    latest_types = [str(item.get("event_type") or "") for item in events]
    if "short_paper_validation" in latest_types:
        latest_validation = next(item for item in reversed(events) if item.get("event_type") == "short_paper_validation")
        latest_manuscript = next((item for item in reversed(events) if item.get("event_type") == "manuscript_version"), None)
        if not latest_manuscript or str(latest_validation.get("created_at") or "") >= str(latest_manuscript.get("created_at") or ""):
            return "validation"
    manuscript_events = [item for item in events if item.get("event_type") == "manuscript_version"]
    if manuscript_events:
        if any(item.get("event_type") in {"revision_review", "revision_resolution_applied", "revision_group_resolution_applied"} for item in events):
            return "revision"
        return "short_drafting"
    if project.get("origin_ids") or any(item.get("event_type") == "evidence_selection" for item in events):
        return "evidence_building"
    return "project_setup"
