from pathlib import Path

from research_fellow.application.thread_documents import current_state_prompt, report_snapshot_prompt
from research_fellow.storage import Ledger


def test_current_state_is_single_living_document(tmp_path: Path):
    ledger = Ledger(tmp_path / "rf.db")
    first = ledger.save_thread_current_state(
        thread_kind="sensemaking", thread_id="sm-1", current_question="Q1", body_text="state one"
    )
    second = ledger.save_thread_current_state(
        thread_kind="sensemaking", thread_id="sm-1", current_question="Q2", body_text="state two"
    )
    assert first["thread_id"] == second["thread_id"]
    loaded = ledger.thread_current_state("sensemaking", "sm-1")
    assert loaded is not None
    assert loaded["current_question"] == "Q2"
    assert loaded["body_text"] == "state two"
    with ledger.connect() as conn:
        count = conn.execute("SELECT COUNT(*) FROM thread_current_states WHERE thread_kind='sensemaking' AND thread_id='sm-1'").fetchone()[0]
    assert count == 1


def test_report_snapshots_are_versioned_and_soft_deletable(tmp_path: Path):
    ledger = Ledger(tmp_path / "rf.db")
    a = ledger.create_thread_report_snapshot(thread_kind="research_question", thread_id="rq-1", title="Report", body_text="one")
    b = ledger.create_thread_report_snapshot(thread_kind="research_question", thread_id="rq-1", title="Report", body_text="two")
    assert len(ledger.thread_report_snapshots("research_question", "rq-1")) == 2
    assert ledger.archive_thread_report_snapshot(a["report_id"], True)
    assert ledger.delete_thread_report_snapshot(b["report_id"])
    rows = ledger.thread_report_snapshots("research_question", "rq-1")
    assert len(rows) == 1
    assert rows[0]["report_id"] == a["report_id"]
    assert rows[0]["archived_at"]


def test_prompts_separate_living_state_from_snapshot_report():
    state_prompt = current_state_prompt(
        thread_kind="sensemaking", title="X", current_question="What?", prior_state="prior",
        conversation=[{"role": "user", "content": "new"}], reports=[]
    )
    assert "CURRENT state" in state_prompt
    assert "남은 불확실성" in state_prompt
    report_prompt = report_snapshot_prompt(
        thread_kind="sensemaking", title="X", current_state="state", current_question="What?"
    )
    assert "point-in-time report" in report_prompt
    assert "Executive Summary" in report_prompt
