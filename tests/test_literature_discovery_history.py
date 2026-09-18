from research_fellow.storage import Ledger
from research_fellow.application.search_profiles import intent_discovery_task_prompt, run_intent_discovery_plan


def test_literature_discovery_history_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "research_fellow.db")
    saved = ledger.save_literature_discovery_session(
        topic="vision-language models for industrial inspection",
        research_context="focus on domain shift",
        target_count=12,
        discovery_source="external",
        search_plan={"queries": ["industrial inspection VLM"]},
        search_summary="A compact landscape",
        results=[{"title": "Paper A", "source_url": "https://arxiv.org/abs/1234.5678"}],
    )

    assert saved["topic"] == "vision-language models for industrial inspection"
    assert saved["research_context"] == "focus on domain shift"
    assert saved["search_plan"]["queries"] == ["industrial inspection VLM"]
    assert saved["results"][0]["title"] == "Paper A"

    history = ledger.literature_discovery_sessions(limit=10)
    assert len(history) == 1
    assert history[0]["session_id"] == saved["session_id"]
    assert history[0]["discovery_source"] == "external"


def test_literature_discovery_history_can_be_updated_with_selected_details(tmp_path):
    ledger = Ledger(tmp_path / "research_fellow.db")
    saved = ledger.save_literature_discovery_session(
        topic="agent memory",
        research_context="compare semantic and episodic memory",
        target_count=2,
        discovery_source="internal",
        search_plan={"queries": ["agent memory"]},
        results=[
            {"title": "Paper A", "source_url": "https://arxiv.org/abs/1111.1111", "history_detail": False},
            {"title": "Paper B", "source_url": "https://arxiv.org/abs/2222.2222", "history_detail": False},
        ],
    )

    updated = ledger.update_literature_discovery_session_results(
        saved["session_id"],
        [
            {
                "title": "Paper A",
                "source_url": "https://arxiv.org/abs/1111.1111",
                "quick_take": "Detailed interpretation",
                "why_relevant": "Directly compares memory forms",
                "summary": "Abstract text",
                "history_detail": True,
            },
            {"title": "Paper B", "source_url": "https://arxiv.org/abs/2222.2222", "history_detail": False},
        ],
    )

    assert updated is not None
    assert updated["results"][0]["history_detail"] is True
    assert updated["results"][0]["quick_take"] == "Detailed interpretation"
    assert updated["results"][1] == {
        "title": "Paper B",
        "source_url": "https://arxiv.org/abs/2222.2222",
        "history_detail": False,
    }


def test_reference_list_roundtrip(tmp_path):
    ledger = Ledger(tmp_path / "research_fellow.db")
    saved = ledger.upsert_literature_reference(
        topic="vision AI",
        research_context="industrial inspection",
        session_id="lds-1",
        paper={"title":"Paper V", "source_id":"doi:1", "url":"https://doi.org/1", "summary":"abstract"},
        labels=["Computer Vision", "inspection"],
    )
    assert saved["topic"] == "vision AI"
    assert saved["labels"] == ["Computer Vision", "inspection"]
    refs = ledger.literature_references(topic="vision AI")
    assert len(refs) == 1
    assert refs[0]["paper"]["title"] == "Paper V"
    ledger.update_literature_reference_status(saved["reference_id"], "shelved")
    assert ledger.literature_reference(saved["reference_id"])["status"] == "shelved"


def test_unified_discovery_history_contains_card_and_intent_runs(tmp_path):
    ledger = Ledger(tmp_path / "research_fellow.db")
    session = ledger.save_literature_discovery_session(
        topic="agent memory", research_context="from approved claims",
        results=[{"title": "Card-origin paper"}], origin_card_ids=["kc-1", "kc-2"],
    )
    profile = ledger.create_search_profile({
        "intent_id": "intent-1", "title": "Watch agent memory", "question": "What changed?",
        "research_context": "M2 direction", "labels": ["agent memory"],
    })
    run_id = ledger.record_search_run(profile["profile_id"], "scheduled", "agent memory", [{"title": "Intent-origin paper"}], "completed")

    history = ledger.unified_literature_discovery_runs()
    by_id = {item["run_id"]: item for item in history}
    assert by_id[session["session_id"]]["origin_type"] == "knowledge_cards"
    assert by_id[session["session_id"]]["origin_ids"] == ["kc-1", "kc-2"]
    assert by_id[run_id]["origin_type"] == "research_intent"
    assert by_id[run_id]["execution_mode"] == "periodic"


def test_manual_intent_run_history_keeps_edited_topic_context_and_sources(tmp_path):
    ledger = Ledger(tmp_path / "research_fellow.db")
    profile = ledger.create_search_profile({
        "intent_id": "intent-manual", "title": "Original title", "question": "Original question?",
        "research_context": "Original context", "labels": [],
    })
    query = '{"queries":["revised query"],"topic":"Edited question?","research_context":"Edited context","sources":["crossref","arxiv"]}'
    run_id = ledger.record_search_run(profile["profile_id"], "manual_researcher", query, [], "completed_no_candidates")

    history = {item["run_id"]: item for item in ledger.unified_literature_discovery_runs()}
    assert history[run_id]["execution_mode"] == "manual"
    assert history[run_id]["topic"] == "Edited question?"
    assert history[run_id]["research_context"] == "Edited context"
    assert history[run_id]["sources"] == ["crossref", "arxiv"]


def test_intent_llm_task_prompt_uses_recent_runs_and_plan_executes(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "research_fellow.db")
    profile = ledger.create_search_profile({
        "intent_id": "intent-llm", "title": "Agent evaluation", "question": "How are agents evaluated?",
        "research_context": "Focus on persistent job agents", "labels": [],
    })
    ledger.record_search_run(profile["profile_id"], "scheduled", "old strategy", [{"title": "Old paper"}], "completed")
    recent = ledger.search_runs(profile["profile_id"])
    prompt = intent_discovery_task_prompt(profile, recent, 8)
    assert "Old paper" in prompt
    assert "persistent job agents" in prompt

    candidates = [{"source_id":"p1", "title":"New paper", "summary":"agent evaluation", "published":"2026", "authors":[], "url":"https://example.org/p1"}]
    monkeypatch.setattr("research_fellow.application.literature_discovery.collect_multisource_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr("research_fellow.application.literature_discovery.apply_discovery_triage", lambda items, text, count: items)
    outcome = run_intent_discovery_plan(
        ledger, profile, {"scope_summary":"new evidence", "queries":["agent evaluation benchmark"]},
        trigger="test_llm_task", triage_drafter=lambda prompt: "{}", sources=["arxiv"], max_results=8,
    )
    assert outcome["status"] == "completed"
    assert outcome["candidates"][0]["title"] == "New paper"
    assert ledger.search_profiles()[0]["is_active"] is True
