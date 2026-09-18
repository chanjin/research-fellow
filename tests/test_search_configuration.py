from research_fellow.search_configuration import configured_search_sources


def test_search_sources_default_to_all_supported_sources():
    assert configured_search_sources({}) == ["arxiv", "semantic_scholar", "crossref"]


def test_search_sources_are_validated_and_deduplicated():
    configured = configured_search_sources({
        "RESEARCH_FELLOW_SEARCH_SOURCES": "crossref, arxiv,unknown,crossref"
    })
    assert configured == ["crossref", "arxiv"]


def test_invalid_search_source_configuration_falls_back_safely():
    assert configured_search_sources({"RESEARCH_FELLOW_SEARCH_SOURCES": "unknown"}) == [
        "arxiv", "semantic_scholar", "crossref"
    ]
