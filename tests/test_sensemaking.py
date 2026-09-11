from pathlib import Path

from research_fellow.storage import Ledger
from research_fellow.application.sensemaking import (
    parse_sensemaking_card_candidate,
    sensemaking_answer_prompt,
)


def test_sensemaking_thread_persists_turns_and_links_rq(tmp_path: Path):
    ledger = Ledger(tmp_path / "rf.db")
    thread = ledger.create_sensemaking_thread("Quick idea", "Is this meaningful?")
    ledger.add_sensemaking_turn(thread["thread_id"], "assistant", "It may matter.", evidence_card_ids=["kc-1"])
    turns = ledger.sensemaking_turns(thread["thread_id"])
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[-1]["evidence_card_ids"] == ["kc-1"]

    rq = ledger.create_research_question_thread(
        question="What does this imply?", source_type="sensemaking", rationale="promoted", source_payload={"sensemaking_thread_id": thread["thread_id"]}
    )
    ledger.link_sensemaking_to_rq(thread["thread_id"], rq["rq_id"])
    assert ledger.sensemaking_thread(thread["thread_id"])["linked_rq_id"] == rq["rq_id"]


def test_sensemaking_prompt_distinguishes_knowledge_and_quick_literature():
    prompt = sensemaking_answer_prompt(
        thread_title="T", conversation=[], question="Q",
        cards=[{"card_id":"kc-1","title":"A","claim":"Supported claim","context":"C","implication":"I","limits":"L"}],
        papers=[{"title":"Paper","published":"2026-01-01","citation_count":3,"summary":"Abstract"}],
    )
    assert "[kc-1]" in prompt
    assert "provisional" in prompt.lower()
    assert "Quick literature" in prompt


def test_parse_sensemaking_card_candidate_is_provisional_researcher_note():
    card = parse_sensemaking_card_candidate(
        "Title: Task-sensitive memory\nClaim: Task characteristics influence which memory representation is useful.\nContext: Agent memory\nImplication: Retrieval should adapt to task type.\nLabels: agent memory, retrieval\nEvidence: Sensemaking exchange\nConditions: Researcher review required\nLimits: Not yet source-verified",
        thread_title="Memory",
    )
    assert card is not None
    assert card["source_kind"] == "researcher_idea_note"
    assert "agent memory" in card["labels"]
