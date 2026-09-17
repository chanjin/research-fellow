from research_fellow.storage import Ledger


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
