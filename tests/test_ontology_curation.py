from research_fellow.application.ontology_curation import parse_relation_suggestions, parse_type_suggestions


def facets():
    return [{"facet_id": "of-1", "name": "Agent Usecase", "description": ""}]


def types():
    return [
        {"type_id": "ot-1", "name": "Architect Agent", "facet_id": "of-1", "facet_name": "Agent Usecase"},
        {"type_id": "ot-2", "name": "Software Architecture", "facet_id": None, "facet_name": None},
    ]


def test_existing_type_is_resolved_by_name():
    parsed = parse_type_suggestions(
        '{"recommendations":[{"action":"assign_existing","facet":"Agent Usecase","type":"Architect Agent","confidence":0.9,"reason":"fit"}]}',
        existing_facets=facets(), existing_types=types(),
    )
    item = parsed["recommendations"][0]
    assert item["type_id"] == "ot-1"
    assert item["facet_id"] == "of-1"


def test_duplicate_new_type_becomes_existing_assignment():
    parsed = parse_type_suggestions(
        '{"recommendations":[{"action":"create_new","facet":"Agent Usecase","type":"Architect Agent","confidence":0.7,"reason":"new"}]}',
        existing_facets=facets(), existing_types=types(),
    )
    item = parsed["recommendations"][0]
    assert item["action"] == "assign_existing"
    assert item["type_id"] == "ot-1"
    assert parsed["warnings"]


def test_new_type_can_have_no_facet_and_top5_comparisons():
    raw = '''{
      "summary":"candidate",
      "recommendations":[{
        "action":"create_new","facet":"","type":"Decision Reliability",
        "description":"decision quality","confidence":0.8,"reason":"distinct",
        "similar_types":[
          {"type":"A","difference":"d1"},{"type":"B","difference":"d2"},
          {"type":"C","difference":"d3"},{"type":"D","difference":"d4"},
          {"type":"E","difference":"d5"},{"type":"F","difference":"d6"}
        ]
      }]
    }'''
    parsed = parse_type_suggestions(raw, existing_facets=facets(), existing_types=types())
    item = parsed["recommendations"][0]
    assert item["facet_id"] is None
    assert len(item["similar_types"]) == 5


def test_relation_parser_resolves_names_and_drops_invalid():
    raw = '''{
      "relations":[
        {"source_type":"Architect Agent","relation_name":"operates_on","target_type":"Software Architecture","reason":"useful"},
        {"source_type":"Missing","relation_name":"x","target_type":"Software Architecture"}
      ]
    }'''
    parsed = parse_relation_suggestions(raw, existing_types=types())
    assert len(parsed["relations"]) == 1
    assert parsed["relations"][0]["source_type_id"] == "ot-1"
    assert parsed["relations"][0]["target_type_id"] == "ot-2"
    assert parsed["warnings"]
