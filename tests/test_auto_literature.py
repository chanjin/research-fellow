from pathlib import Path

from research_fellow.application.auto_literature import execute_auto_literature_review
from research_fellow.storage import Ledger


def test_auto_literature_runs_to_researcher_report(tmp_path: Path, monkeypatch):
    ledger = Ledger(tmp_path / "local.db")
    case_id = ledger.create_case("research", "auto")
    intent = {
        "intent_id": "intent-auto-1", "title": "자동 탐색", "purpose": "관련 근거 조사",
        "question": "Architect Agent는 복합 제약을 어떻게 처리하는가?",
        "research_context": "Architect Agent 연구", "labels": ["architect agent"], "priority": "높음",
        "expected_evidence": "관련 방법", "completion_condition": "5편 비교", "execution_mode": "auto", "created_by": "m2_auto",
    }
    ph_id = ledger.record(case_id, "curation_intent", "m2", ["m1"], "curation_intent", intent, subject_id=intent["intent_id"], status="ready")
    ledger.create_search_profile(intent)
    event = ledger.phenomenon(ph_id)

    candidates = [{
        "source_id": "1234.5678", "title": "Paper A", "published": "2024-01-01", "url": "https://arxiv.org/abs/1234.5678",
        "authors": ["A"], "summary": "abstract", "citation_count": 42,
        "relevance": {"level": "high", "rationale": "direct"}, "abstract_shortlist": True,
    }]
    monkeypatch.setattr("research_fellow.application.auto_literature.run_profile", lambda *args, **kwargs: {
        "run_id": "sr-1", "query": "q", "candidates": candidates, "status": "completed", "error": "",
    })
    monkeypatch.setattr("research_fellow.application.auto_literature.process_top_papers", lambda *args, **kwargs: [{
        **candidates[0], "full_text_status": "completed", "full_text_review": "SIMILARITY: 90\n- direct",
        "full_text_similarity": 90, "influential_citation_count": 3,
    }])
    # The mocked search run was not inserted into SQLite, so make the update a no-op for this application-level test.
    monkeypatch.setattr(ledger, "update_search_run_candidates", lambda *args, **kwargs: None)

    result = execute_auto_literature_review(
        ledger, event, tmp_path / "cache",
        keyword_drafter=lambda prompt: "architect agent",
        abstract_reviewer=lambda prompt: "",
        fulltext_drafter=lambda prompt: "",
        synthesis_drafter=lambda prompt: "## 탐색 요약\n초록 1편, 본문 1편을 비교했다.",
    )

    assert result["status"] == "completed"
    assert ledger.phenomenon(ph_id)["status"] == "completed"
    reports = [item for item in ledger.phenomena(type_="advice_report") if item["subject_type"] == "auto_literature_report"]
    assert len(reports) == 1
    assert reports[0]["payload"]["abstract_review_count"] == 1
    assert reports[0]["payload"]["top_papers"][0]["citation_count"] == 42
    updates = ledger.phenomena(type_="knowledge_update")
    assert len(updates) == 1
