from pathlib import Path

from research_fellow.application.llm_retry import LLMRetryExhausted, call_with_retry
from research_fellow.application.research_cycle import execute_auto_research_cycle
from research_fellow.storage import Ledger


def test_call_with_retry_recovers_from_empty_response():
    calls = []
    def draft(prompt: str):
        calls.append(prompt)
        return "" if len(calls) < 3 else "VALID"
    text, attempts = call_with_retry(draft, "prompt", stage="unit", validator=lambda value: value == "VALID", sleep=lambda _: None)
    assert text == "VALID"
    assert attempts == 3
    assert "RETRY INSTRUCTION" in calls[1]


def test_call_with_retry_raises_with_recommendation_after_three_failures():
    try:
        call_with_retry(lambda prompt: "{unfinished", "prompt", stage="json_stage", sleep=lambda _: None)
        assert False, "expected retry exhaustion"
    except LLMRetryExhausted as error:
        assert error.attempts == 3
        assert error.error_type == "truncated_response"
        assert "strict" in error.recommended_action


def test_auto_cycle_records_retry_task_when_rq_generation_never_recovers(tmp_path: Path):
    ledger = Ledger(tmp_path / "local.db")
    case_id = ledger.create_case("research", "new")
    ledger.record(
        case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card",
        {"title": "new", "card_id": "kc-1"}, "kc-1", status="completed",
    )
    cards = [{
        "card_id": "kc-1", "title": "Card", "claim": "Claim", "evidence_excerpt": "evidence",
        "provenance": {"source_name": "paper"},
    }]
    result = execute_auto_research_cycle(
        ledger, cards, tmp_path / "cache",
        rq_drafter=lambda prompt: "",
        priority_drafter=lambda prompt: "",
        keyword_drafter=lambda prompt: "",
        abstract_reviewer=lambda prompt: "",
        fulltext_drafter=lambda prompt: "",
        synthesis_drafter=lambda prompt: "",
    )
    assert result["status"] == "rq_generation_failed"
    failures = ledger.auto_research_failures()
    assert len(failures) == 1
    assert failures[0]["stage"] == "rq_generation"
    assert failures[0]["attempt_count"] == 3
    # Input cards remain unreviewed so the researcher can manually retry the cycle.
    assert ledger.reviewed_knowledge_update_ids() == set()


def test_prepare_intent_for_retry_resets_failed_intent(tmp_path: Path):
    ledger = Ledger(tmp_path / "local.db")
    case_id = ledger.create_case("research", "intent")
    intent = {
        "intent_id": "intent-1", "title": "t", "purpose": "p", "question": "q",
        "research_context": "c", "labels": ["agent"], "priority": "high",
        "expected_evidence": "e", "completion_condition": "done",
    }
    ph = ledger.record(case_id, "curation_intent", "m2", ["m1"], "curation_intent", intent, subject_id="intent-1", status="ready")
    ledger.create_search_profile(intent)
    ledger.transition(ph, "ready", "failed")
    event = ledger.prepare_intent_for_retry("intent-1")
    assert event is not None
    assert ledger.phenomenon(ph)["status"] == "ready"
    assert [p for p in ledger.search_profiles(active_only=True) if p["intent_id"] == "intent-1"]
