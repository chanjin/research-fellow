from pathlib import Path

from research_fellow.application.search_profiles import screen_abstract_batches
from research_fellow.storage import Ledger


def _candidate(index: int) -> dict:
    return {
        "source_id": f"p{index}",
        "title": f"Paper {index}",
        "summary": "architect agent and non-functional requirements",
        "url": f"https://arxiv.org/abs/p{index}",
    }


def test_abstract_batches_resume_from_saved_reviews():
    candidates = [_candidate(i) for i in range(25)]
    calls = {"count": 0}
    saved = {}

    def reviewer(prompt: str) -> str:
        calls["count"] += 1
        # The first 20-paper batch succeeds; the second simulates a failure.
        if calls["count"] == 2:
            raise RuntimeError("truncated")
        ids = [line.split(" | ", 1)[0].strip() for line in prompt.splitlines() if line.startswith("p") and " | " in line]
        return "\n".join(f"{pid} | high | relevant" for pid in ids)

    def checkpoint(_batch_no, rows):
        for row in rows:
            saved[row["source_id"]] = row

    try:
        screen_abstract_batches({"title": "x", "question": "q", "context": "c", "keywords": []}, candidates, reviewer, batch_size=20, on_batch_completed=checkpoint)
    except RuntimeError:
        pass

    assert len(saved) == 20

    second_calls = {"count": 0}
    def reviewer2(prompt: str) -> str:
        second_calls["count"] += 1
        ids = [line.split(" | ", 1)[0].strip() for line in prompt.splitlines() if line.startswith("p") and " | " in line]
        return "\n".join(f"{pid} | high | relevant" for pid in ids)

    resumed = screen_abstract_batches(
        {"title": "x", "question": "q", "context": "c", "keywords": []},
        candidates,
        reviewer2,
        batch_size=20,
        existing_reviews=saved,
    )
    assert len(resumed) == 25
    assert second_calls["count"] == 1


def test_research_question_change_log_and_review_rollback(tmp_path: Path):
    ledger = Ledger(tmp_path / "local.db")
    rq = ledger.upsert_research_question({"question": "Q?", "rationale": "why", "source_card_ids": []})
    rq_id = rq["rq_id"]
    ledger.add_research_question_change(rq_id, "created", "새 질문이 생성되었습니다.")
    assert ledger.latest_research_question_change(rq_id)["summary"] == "새 질문이 생성되었습니다."

    case_id = ledger.create_case("research", "x")
    update_id = ledger.record(case_id, "knowledge_update", "m1", ["m2"], "knowledge_card", {"title": "새 지식", "card_id": "kc-1"}, subject_id="kc-1")
    update = ledger.phenomenon(update_id)
    review_id = ledger.create_research_state_review("manual", [update])
    ledger.complete_research_state_review(review_id, generated_rq_count=1)
    assert update_id in ledger.reviewed_knowledge_update_ids()
    ledger.rollback_research_state_review(review_id)
    assert update_id not in ledger.reviewed_knowledge_update_ids()
