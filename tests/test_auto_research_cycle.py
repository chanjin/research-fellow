from pathlib import Path
import re

from research_fellow.application.research_cycle import execute_auto_research_cycle
from research_fellow.storage import Ledger


def _update(ledger: Ledger, card_id: str) -> None:
    case_id = ledger.create_case("research", card_id)
    ledger.record(
        case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card",
        {"title": card_id, "card_id": card_id}, card_id, status="completed",
    )


def test_auto_cycle_starts_from_unreviewed_cards_and_selects_this_batch_top3(tmp_path: Path, monkeypatch):
    import research_fellow.application.research_cycle as cycle

    ledger = Ledger(tmp_path / "local.db")
    cards = []
    for index in range(1, 5):
        card_id = f"kc-{index}"
        _update(ledger, card_id)
        cards.append({
            "card_id": card_id, "title": f"Card {index}", "claim": f"Claim {index}",
            "evidence_excerpt": "evidence", "provenance": {"source_name": "paper"},
        })

    topics = ["아키텍처 제약", "장기 메모리", "평가 신뢰성", "도구 권한"]
    rq_output = "\n\n".join([
        f"""## RQ {index}\nQuestion: {topics[index-1]}에 대해 무엇을 확인해야 하는가?\nWhy Now: 새 카드 kc-{index}가 {topics[index-1]}의 새로운 공백을 보여준다.\nGap/Tension: {topics[index-1]}의 적용 조건이 불명확하다.\nResearch Context: 이번 배치의 질문이다.\nSource Card IDs: kc-{index}\nExploration Need: {topics[index-1]} 관련 추가 문헌"""
        for index in range(1, 5)
    ])

    def priority(prompt: str) -> str:
        ids = re.findall(r"RQ_ID:\s*(rq-[A-Za-z0-9]+)", prompt)[:3]
        return "\n\n".join(
            f"## Priority {i}\nRQ_ID: {rq_id}\nSCORE: {6-i}\nREASON: 이번 새 정보 배치에서 중요함"
            for i, rq_id in enumerate(ids, 1)
        )

    def fake_literature(ledger, event, cache_dir, **kwargs):
        ledger.transition(event["phenomenon_id"], "ready", "completed")
        return {"status": "completed", "report": "done", "papers": [], "run": {}}

    monkeypatch.setattr(cycle, "execute_auto_literature_review", fake_literature)
    result = execute_auto_research_cycle(
        ledger, cards, tmp_path / "cache",
        rq_drafter=lambda prompt: rq_output,
        priority_drafter=priority,
        keyword_drafter=lambda prompt: "",
        abstract_reviewer=lambda prompt: "",
        fulltext_drafter=lambda prompt: "",
        synthesis_drafter=lambda prompt: "",
    )

    assert result["status"] == "completed"
    assert result["source_card_count"] == 4
    assert len(result["questions"]) == 4
    assert len(result["dispatched"]) == 3
    assert all(item["status"] == "completed" for item in result["executions"])
    review = ledger.research_state_reviews(limit=1)[0]
    assert review["source_card_count"] == 4
    assert review["generated_rq_count"] == 4
    assert review["selected_rq_count"] == 3
    assert len([item for item in ledger.research_state_review_questions(review["review_id"]) if item["selected"]]) == 3

    # The cycle consumed the cards; running again does not use the old backlog as an auto trigger.
    again = execute_auto_research_cycle(
        ledger, cards, tmp_path / "cache",
        rq_drafter=lambda prompt: rq_output,
        priority_drafter=priority,
        keyword_drafter=lambda prompt: "",
        abstract_reviewer=lambda prompt: "",
        fulltext_drafter=lambda prompt: "",
        synthesis_drafter=lambda prompt: "",
    )
    assert again["status"] == "no_new_information"
