from research_fellow.application.search_profiles import parse_auto_search_strategy, query_ladder


def test_parse_auto_search_strategy_builds_or_groups_and_and_queries():
    draft = '''
CONCEPT_GROUPS:
Agent use case | architect agent ; software architect agent ; LLM architecture assistant
Constraint | non-functional requirements ; quality attributes ; NFR
Decision | architectural decision ; design decision dependency
CORE_TERMS:
architect
constraints
dependency
QUERY_VARIANTS:
(all:"architect agent" OR all:"software architect agent") AND (all:"non-functional requirements" OR all:"quality attributes")
(all:"LLM architecture assistant") AND (all:"architectural decision" OR all:"design decision dependency")
'''
    plan = parse_auto_search_strategy(draft)
    assert len(plan["concept_groups"]) == 3
    assert plan["concept_groups"][0]["terms"][0] == "architect agent"
    assert len(plan["queries"]) == 2
    assert " OR " in plan["queries"][0]
    assert " AND " in plan["queries"][0]

    profile = {"keywords": plan["phrases"], "core_terms": plan["core_terms"], "boolean_queries": plan["queries"]}
    ladder = query_ladder(profile)
    assert [label for label, _ in ladder] == ["자동 Boolean 1", "자동 Boolean 2"]
    assert ladder[0][1] == plan["queries"][0]


def test_auto_search_strategy_old_keyword_format_falls_back():
    plan = parse_auto_search_strategy('''
PRIORITY_PHRASES:
architect agent
non-functional requirements
architecture decision
CORE_TERMS:
architect
requirements
decision
''')
    assert plan["phrases"]
    assert plan["queries"]
    assert any(" OR " in query or " AND " in query for query in plan["queries"])
