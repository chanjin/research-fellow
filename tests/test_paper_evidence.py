from research_fellow.application.paper_evidence import (
    assemble_paper_evidence_candidates,
    paper_evidence_query,
)


def test_title_and_question_drive_bounded_ontology_expansion():
    cards = [
        {"card_id": "kc-old", "title": "Prior evidence", "claim": "Persistent jobs need memory."},
        {"card_id": "kc-seed", "title": "Agent memory", "claim": "Procedural memory supports agents."},
        {"card_id": "kc-same", "title": "Workflow memory", "claim": "Workflow memory supports iteration."},
        {"card_id": "kc-near", "title": "Human review", "claim": "Human review constrains autonomy."},
    ]
    query = paper_evidence_query("Persistent co-author", "How does memory support iterative writing?")
    result = assemble_paper_evidence_candidates(
        query=query,
        cards=cards,
        search_hits=[{"card_id": "kc-seed", "score": 0.9}],
        inherited_card_ids=["kc-old"],
        seed_type_ids={"t-memory"},
        type_names={"t-memory": "Memory", "t-review": "Human Review"},
        type_relations=[{
            "source_type_id": "t-review",
            "target_type_id": "t-memory",
            "relation_name": "constrains",
        }],
        card_ids_by_type={"t-memory": ["kc-seed", "kc-same"], "t-review": ["kc-near"]},
    )
    by_id = {item["card_id"]: item for item in result}
    assert query.startswith("Paper title: Persistent co-author")
    assert by_id["kc-old"]["source"] == "inherited_rq"
    assert by_id["kc-seed"]["source"] == "hybrid_search"
    assert by_id["kc-same"]["source"] == "ontology_same_type"
    assert by_id["kc-near"]["source"] == "ontology_related_type"
    assert "Human Review —constrains→ Memory" in by_id["kc-near"]["ontology_paths"]

