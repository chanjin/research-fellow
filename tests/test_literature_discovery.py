from research_fellow.application.literature_discovery import (
    apply_discovery_triage,
    collect_arxiv_candidates,
    parse_discovery_search_plan,
    parse_external_literature_results,
)


def test_parse_discovery_search_plan_json():
    plan = parse_discovery_search_plan('''```json
    {"scope_summary":"scope","queries":["all:\\"vision agent\\" AND all:evaluation", "all:\\"multimodal agent\\""],"search_notes":["broad"]}
    ```''')
    assert len(plan["queries"]) == 2
    assert plan["scope_summary"] == "scope"


def test_collect_arxiv_candidates_deduplicates():
    def fake_search(query, limit):
        return [
            {"source_id": "1", "title": "Paper A", "summary": "A", "published": "2025", "authors": [], "url": "u1"},
            {"source_id": "2", "title": "Paper B", "summary": "B", "published": "2024", "authors": [], "url": "u2"},
        ]
    results = collect_arxiv_candidates(["q1", "q2"], 10, searcher=fake_search)
    assert [item["source_id"] for item in results] == ["1", "2"]


def test_apply_discovery_triage_maps_existing_candidates():
    candidates = [
        {"source_id": "1", "title": "A", "summary": "abs", "published": "2025", "authors": [], "url": "u"},
        {"source_id": "2", "title": "B", "summary": "abs", "published": "2024", "authors": [], "url": "u"},
    ]
    output = '{"papers":[{"source_id":"2","relevance_score":91,"quick_take":"take","why_relevant":"why","caution":""}]}'
    ranked = apply_discovery_triage(candidates, output, 10)
    assert ranked[0]["source_id"] == "2"
    assert ranked[0]["relevance_score"] == 91


def test_parse_external_results_uses_stable_fallback_id():
    raw = '''{
      "search_summary": "overview",
      "papers": [
        {"title":"Paper X","authors":["A"],"publication_year":"2026","source_url":"https://example.org/x","source_id":"","abstract_or_summary":"s","relevance_score":88,"quick_take":"q","why_relevant":"w","caution":"c"}
      ]
    }'''
    first = parse_external_literature_results(raw, 10)["papers"][0]
    second = parse_external_literature_results(raw, 10)["papers"][0]
    assert first["source_id"].startswith("external-")
    assert first["source_id"] == second["source_id"]
    assert first["url"] == "https://example.org/x"


def test_parse_external_results_repairs_unescaped_latex_math():
    raw = r'''{
      "search_summary": "The objective uses $\pi(K \mid S_t)$ and $\text{late binding}$.",
      "papers": [
        {"title":"Math Paper","authors":["A"],"publication_year":"2026","source_url":"https://example.org/math","source_id":"math-1","abstract_or_summary":"Defines $\alpha$ under $\mathcal{W}$.","relevance_score":90,"quick_take":"Uses $\pi$.","why_relevant":"Matches $S_t$.","caution":""}
      ]
    }'''

    parsed = parse_external_literature_results(raw, 10)

    assert parsed["search_summary"] == r"The objective uses $\pi(K \mid S_t)$ and $\text{late binding}$."
    assert parsed["papers"][0]["summary"] == r"Defines $\alpha$ under $\mathcal{W}$."


def test_parse_external_results_keeps_already_escaped_latex():
    raw = r'{"search_summary":"Uses $\\pi$ safely.","papers":[{"title":"P","source_url":"https://example.org/p","abstract_or_summary":"$\\alpha$","relevance_score":80}]}'

    parsed = parse_external_literature_results(raw, 10)

    assert parsed["search_summary"] == r"Uses $\pi$ safely."
    assert parsed["papers"][0]["summary"] == r"$\alpha$"


def test_parse_external_results_repairs_unescaped_prose_quotes():
    raw = '''{
      "search_summary": "Reviews the "agent-as-job" framing",
      "papers": [
        {"title":"The "Agent" Problem","source_url":"https://example.org/p","abstract_or_summary":"Calls this the "responsibility gap" in deployment.","relevance_score":80}
      ]
    }'''

    parsed = parse_external_literature_results(raw, 10)

    assert parsed["search_summary"] == 'Reviews the "agent-as-job" framing'
    assert parsed["papers"][0]["title"] == 'The "Agent" Problem'


def test_parse_external_results_repairs_missing_and_trailing_commas():
    raw = '''{
      "search_summary": "overview"
      "papers": [
        {"title":"Paper A" "authors":["A"],"source_url":"https://example.org/a","relevance_score":80,},
        {"title":"Paper B","source_url":"https://example.org/b","relevance_score":70,},
      ]
    }'''

    parsed = parse_external_literature_results(raw, 10)

    assert [paper["title"] for paper in parsed["papers"]] == ["Paper A", "Paper B"]

from research_fellow.application.literature_discovery import build_paper_labels, collect_multisource_candidates, google_scholar_url


def test_multisource_candidates_merge_duplicate_title_and_sources():
    def fake_arxiv(query, limit):
        return [{"source_id":"2401.12345","title":"Agent Vision","summary":"vision agent memory","published":"2025","authors":[],"url":"https://arxiv.org/abs/2401.12345"}]
    def fake_semantic(query, limit):
        return [{"source_id":"2401.12345","title":"Agent Vision","summary":"vision agent memory","published":"2025","authors":[],"url":"https://arxiv.org/abs/2401.12345","discovery_sources":["Semantic Scholar"]}]
    results = collect_multisource_candidates("agent vision", "", ["q"], ["arxiv","semantic_scholar"], 10, arxiv_searcher=fake_arxiv, semantic_searcher=fake_semantic, crossref_searcher=lambda q,l: [])
    assert len(results) == 1
    assert "arXiv" in results[0]["discovery_sources"]
    assert "Semantic Scholar" in results[0]["discovery_sources"]


def test_build_paper_labels_and_google_scholar_link():
    paper = {"title":"Vision Language Models for Industrial Inspection", "summary":"domain adaptation for industrial defect inspection", "subjects":["Computer Vision"]}
    labels = build_paper_labels(paper)
    assert "Computer Vision" in labels
    assert "inspection" in [x.lower() for x in labels]
    assert "scholar.google.com" in google_scholar_url(paper)
