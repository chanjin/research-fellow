import json

from research_fellow.infrastructure.semantic_scholar import enrich_citation_counts


class _Response:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps([{"citationCount": 123, "influentialCitationCount": 7}]).encode()


def test_semantic_scholar_citation_enrichment(monkeypatch):
    monkeypatch.setattr("research_fellow.infrastructure.semantic_scholar.urlopen", lambda *args, **kwargs: _Response())
    rows = enrich_citation_counts([{"source_id": "1234.5678", "title": "A"}])
    assert rows[0]["citation_count"] == 123
    assert rows[0]["influential_citation_count"] == 7


def test_citation_enrichment_is_best_effort(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr("research_fellow.infrastructure.semantic_scholar.urlopen", fail)
    rows = enrich_citation_counts([{"source_id": "1234.5678", "title": "A"}])
    assert rows[0]["citation_count"] is None
    assert rows[0]["title"] == "A"
