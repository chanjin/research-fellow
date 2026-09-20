from pathlib import Path

from research_fellow.application.short_paper_milestone import (
    frozen_milestone,
    manuscript_fingerprint,
    normalize_writing_spec,
    project_workflow_stage,
    validate_short_paper,
)


def _manuscript() -> dict:
    return {
        "version": 3,
        "title": "검증 가능한 숏페이퍼",
        "sections": [
            {
                "section_id": "intro",
                "title": "1. 서론",
                "paragraphs": [{
                    "paragraph_id": "p-1",
                    "sentences": [{
                        "sentence_id": "s-001",
                        "text": "근거가 연결된 중심 주장이다. " * 8,
                        "role": "central_claim",
                        "evidence_card_ids": ["kc-1"],
                        "annotations": [],
                    }],
                }],
            },
            {
                "section_id": "conclusion",
                "title": "2. 결론",
                "paragraphs": [{
                    "paragraph_id": "p-2",
                    "sentences": [{
                        "sentence_id": "s-002",
                        "text": "연구질문에 대한 제한된 결론이다.",
                        "role": "conclusion",
                        "evidence_card_ids": ["kc-1"],
                        "annotations": [],
                    }],
                }],
            },
        ],
        "appendix_claims": [],
    }


def _spec() -> dict:
    return normalize_writing_spec({
        "spec_version": 2,
        "audience": "연구자",
        "central_claim": "통제된 에이전트가 더 검증 가능하다.",
        "target_min_chars": 100,
        "target_max_chars": 1000,
        "required_section_terms": ["서론", "결론"],
        "required_card_ids": ["kc-1"],
        "writing_rules": ["과도한 일반화를 피한다"],
        "open_decisions": [],
        "research_method":"비교 사례연구",
        "research_method_rationale":"동일 업무에서 통제 차이를 관찰한다.",
        "verification_plan":[{"plan_id":"p1","method":"trace 비교"}],
        "execution_guide":["시나리오를 고정한다."],
        "result_recording_plan":[{"artifact":"비교표","status":"planned"}],
    }, {})


def test_static_validation_passes_only_current_grounded_manuscript():
    manuscript = _manuscript()
    result = validate_short_paper(
        manuscript,
        _spec(),
        valid_card_ids={"kc-1"},
        valid_reference_paper_ids=set(),
        todos=[],
        reference_count=1,
    )
    assert result["passed"] is True
    assert result["blocker_count"] == 0
    assert result["manuscript_hash"] == manuscript_fingerprint(manuscript)


def test_static_validation_blocks_open_decisions_p0_and_missing_evidence():
    manuscript = _manuscript()
    manuscript["sections"][0]["paragraphs"][0]["sentences"][0]["evidence_card_ids"] = []
    manuscript["sections"][1]["paragraphs"][0]["sentences"][0]["evidence_card_ids"] = []
    spec = {**_spec(), "open_decisions": ["대조군 범위를 확정한다"]}
    result = validate_short_paper(
        manuscript,
        spec,
        valid_card_ids={"kc-1"},
        valid_reference_paper_ids=set(),
        todos=[{"todo_id": "todo-1", "priority": "P0", "status": "open"}],
        reference_count=0,
    )
    failed = {item["check_id"] for item in result["checks"] if not item["passed"]}
    assert result["passed"] is False
    assert {"required_cards", "critical_evidence", "p0_todos", "researcher_decisions", "references"} <= failed


def test_freeze_unfreeze_and_portfolio_stage_follow_event_order():
    freeze = {
        "event_type": "short_paper_frozen",
        "created_at": "2026-01-01T00:00:00+00:00",
        "payload": {"milestone_id": "kshort-1"},
    }
    unfreeze = {
        "event_type": "short_paper_unfrozen",
        "created_at": "2026-01-01T00:00:00+00:00",
        "payload": {"reason": "새 근거"},
    }
    assert frozen_milestone([freeze]) == {"milestone_id": "kshort-1"}
    assert frozen_milestone([freeze, unfreeze]) is None
    assert project_workflow_stage({"status": "active", "stage": "short_paper", "events": [freeze]}) == "short_paper_frozen"
    assert project_workflow_stage({"status": "active", "stage": "short_paper", "events": [freeze, unfreeze]}) == "revision"


def test_draft_and_review_templates_consume_writing_spec():
    prompt_dir = Path(__file__).parents[1] / "src" / "research_fellow" / "prompts"
    draft = (prompt_dir / "m2_short_paper_draft.j2").read_text(encoding="utf-8")
    review = (prompt_dir / "m2_short_paper_review.j2").read_text(encoding="utf-8")
    full_revision = (prompt_dir / "m2_short_paper_full_revision.j2").read_text(encoding="utf-8")
    for template in (draft, review, full_revision):
        assert "project.writing_spec" in template
        assert "central_claim" in template
        assert "research_method" in template
    assert "Required section terms describe required section titles" in draft
    assert "Treat each unresolved researcher decision" in draft
    assert "Return only the required JSON" in draft
