from pathlib import Path

from research_fellow.application.llm_retry import call_with_retry, LLMRetryExhausted, failure_payload
from research_fellow.application.manual_recovery import external_recovery_prompt, validate_external_response
from research_fellow.storage import Ledger


def test_retry_failure_payload_contains_external_recovery_prompt():
    try:
        call_with_retry(lambda prompt: "", "ORIGINAL PROMPT", stage="synthesis", max_attempts=1, sleep=lambda _: None)
        assert False, "expected retry exhaustion"
    except LLMRetryExhausted as error:
        payload = failure_payload(error)
    prompt = external_recovery_prompt({"stage": payload["stage"], "context": payload["context"]})
    assert "ORIGINAL PROMPT" in prompt
    assert "RETRY INSTRUCTION" in prompt


def test_abstract_manual_response_requires_all_batch_ids():
    context = {"source_ids": ["p1", "p2"]}
    ok, _ = validate_external_response("abstract_screening", "p1 | high | direct", context)
    assert not ok
    ok, _ = validate_external_response(
        "abstract_screening",
        "p1 | high | direct relevance\np2 | medium | partial relevance",
        context,
    )
    assert ok


def test_manual_override_is_durable_in_run_checkpoint(tmp_path: Path):
    ledger = Ledger(tmp_path / "research.db")
    run_id = ledger.create_auto_research_run(stage="abstract_screening")
    ledger.set_manual_recovery_override(run_id, stage="abstract_screening", item_key="batch-4", response="p1 | high | reason")
    assert ledger.manual_recovery_override(run_id, stage="abstract_screening", item_key="batch-4") == "p1 | high | reason"
    ledger.clear_manual_recovery_override(run_id, stage="abstract_screening", item_key="batch-4")
    assert ledger.manual_recovery_override(run_id, stage="abstract_screening", item_key="batch-4") == ""
