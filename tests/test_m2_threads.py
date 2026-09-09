from pathlib import Path

from research_fellow.application.m2_threads import (
    build_m2_thread_review_prompt,
    extract_knowledge_gaps,
    extract_refined_question,
    validate_manual_m2_report,
)
from research_fellow.storage import Ledger


def test_thread_sources_versions_and_reports(tmp_path: Path):
    ledger = Ledger(tmp_path / "test.db")
    rq = ledger.create_research_question_thread(
        question="Should an architect agent represent NFR dependencies?",
        source_type="researcher",
        rationale="researcher asked",
        research_context="software architecture",
        source_payload={"researcher_comment": "focus on working memory"},
    )
    rq_id = rq["rq_id"]
    thread = ledger.research_question_thread(rq_id)
    assert thread["source_type"] == "researcher"
    assert thread["versions"][0]["version_no"] == 1

    assert ledger.refine_research_question(rq_id, "How should an architect agent represent NFR dependency in working memory?", "narrowed")
    thread = ledger.research_question_thread(rq_id)
    assert len(thread["versions"]) == 2
    assert "working memory" in thread["question"]

    report = ledger.save_m2_report(
        rq_id=rq_id,
        report_text="### Current Answer\nYes.\n### Evidence\n[kc-1]\n### Knowledge Gaps / M1 Supplementation Need\nNeed empirical validation.",
        context_text="project context",
        evidence_card_ids=["kc-1"],
        generation_mode="manual_external_llm",
        knowledge_gaps="Need empirical validation.",
    )
    assert report["generation_mode"] == "manual_external_llm"
    assert ledger.research_question(rq_id)["exploration_need"] == "Need empirical validation."
    assert ledger.archive_m2_report(report["report_id"], True)
    assert ledger.m2_report(report["report_id"])["archived_at"]
    assert ledger.delete_m2_report(report["report_id"])
    assert ledger.m2_reports(rq_id) == []


def test_external_review_prompt_and_validation():
    cards = [{
        "card_id": "kc-1", "title": "NFR dependency", "claim": "NFRs interact.",
        "conditions": "architecture decisions", "limits": "single case", "provenance": {"source_name": "Paper A"},
    }]
    prompt = build_m2_thread_review_prompt(
        question="How should NFR dependency be represented?", context="architect agent", cards=cards,
        source_type="researcher", source_payload={},
    )
    assert "kc-1" in prompt
    good = """### Current Answer\nUse an explicit dependency representation based on the selected evidence.\n\n### Evidence\nThe interaction is supported by [kc-1].\n\n### Interpretation and Implications\nThis is an interpretation for an architect agent and requires validation.\n\n### Constraints and Unresolved Issues\nThe evidence is limited to one case.\n\n### Knowledge Gaps / M1 Supplementation Need\nNeed empirical studies of agent working memory.\n\n### Question Refinement\nHow should an architect agent represent NFR dependencies in working memory?\n"""
    ok, _ = validate_manual_m2_report(good, valid_card_ids={"kc-1"})
    assert ok
    assert "empirical studies" in extract_knowledge_gaps(good)
    assert "working memory" in extract_refined_question(good)

    bad = good.replace("[kc-1]", "[kc-999]")
    ok, message = validate_manual_m2_report(bad, valid_card_ids={"kc-1"})
    assert not ok
    assert "kc-999" in message
