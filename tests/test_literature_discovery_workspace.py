from research_fellow.storage import Ledger


def test_discovery_workspace_roundtrip_and_clear(tmp_path):
    ledger = Ledger(tmp_path / "research.db")
    payload = {
        "m1-discovery-topic": "Agent role emergence",
        "m1-discovery-context": "High-risk industrial systems",
        "m1-discovery-results": [{"title": "Persistent temporary result", "source_id": "p-1"}],
        "m1-discovery-plan": {"queries": ["agent role emergence safety"]},
        "m1-discovery-count": 12,
    }

    ledger.save_literature_discovery_workspace(payload)
    restored = ledger.literature_discovery_workspace()

    assert restored is not None
    assert restored["m1-discovery-topic"] == payload["m1-discovery-topic"]
    assert restored["m1-discovery-results"][0]["title"] == "Persistent temporary result"
    assert ledger.clear_literature_discovery_workspace() is True
    assert ledger.literature_discovery_workspace() is None
