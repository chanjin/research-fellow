from research_fellow.application.meaning_summary import attach_reports, build_fact_groups, delta_inputs
from research_fellow.application.search_profiles import query_ladder


def _card(identifier: str) -> dict[str, object]:
    return {"card_id": identifier, "title": identifier, "claim": f"Claim for {identifier}", "provenance": {"source_name": "source"}}


def test_groups_only_follow_approved_relations_and_attach_explicit_report_references() -> None:
    cards = [_card("K1"), _card("K2"), _card("K3")]
    relations = [{"relation_id": "R1", "source_card_id": "K1", "target_card_id": "K2", "relation_type": "supports", "evidence": "explicit evidence", "conditions": "condition", "confidence": "medium"}]
    reports = [
        {"payload": {"evidence_card_ids": ["K2"]}},
        {"payload": {"evidence_card_ids": []}},
    ]
    groups, unmatched = attach_reports(build_fact_groups(cards, relations), reports)
    assert [len(group["cards"]) for group in groups] == [2, 1]
    assert len(groups[0]["reports"]) == 1
    assert unmatched == [reports[1]]


def test_delta_uses_last_saved_baseline_and_keeps_related_old_card_as_context() -> None:
    cards = [_card("K1"), _card("K2")]
    relations = [{"relation_id": "R1", "source_card_id": "K1", "target_card_id": "K2", "relation_type": "supports", "evidence": "explicit evidence", "conditions": "condition", "confidence": "medium"}]
    previous = {"payload": {"baseline": {"card_ids": ["K1"], "relation_ids": [], "report_ids": []}}}
    delta = delta_inputs(cards, relations, [], previous)
    assert delta["delta"]["card_ids"] == ["K2"]
    assert delta["delta"]["relation_ids"] == ["R1"]
    assert {card["card_id"] for card in delta["cards"]} == {"K1", "K2"}


def test_keyword_plan_runs_phrases_independently_and_keeps_explicit_and_queries() -> None:
    plan = query_ladder({
        "keywords": ["large language model", "design feasibility", "evaluation"],
        "core_terms": ["LLM", "design", "feasibility", "evaluation", "benchmark"],
    })
    assert [label for label, _ in plan] == [
        "우선순위 1 · 정확 구문", "우선순위 2 · 정확 구문", "우선순위 3 · 정확 구문",
        "전체 구문 중복 제거어 AND", "LLM 선정 핵심 5개어 AND",
    ]
    assert plan[:3] == [
        ("우선순위 1 · 정확 구문", 'all:"large language model"'),
        ("우선순위 2 · 정확 구문", 'all:"design feasibility"'),
        ("우선순위 3 · 정확 구문", 'all:"evaluation"'),
    ]
    assert "all:and" not in plan[3][1]
    assert plan[-1][1] == "all:llm AND all:design AND all:feasibility AND all:evaluation AND all:benchmark"
