from pathlib import Path

from research_fellow.application.ontology import ontology_dot, search_cards_for_ontology
from research_fellow.infrastructure.retrieval import KnowledgeRetriever
from research_fellow.storage import Ledger


def test_ontology_crud_and_assignments(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    first = ledger.create_ontology_type("Reasoning Method", "Ways an agent reasons")
    second = ledger.create_ontology_type("Agent Capability", "Capabilities of an agent")
    ledger.assign_cards_to_ontology_type(first["type_id"], ["kc-1", "kc-2"])
    assert ledger.ontology_card_ids(first["type_id"]) == ["kc-1", "kc-2"]
    assert ledger.ontology_types_for_card("kc-1")[0]["name"] == "Reasoning Method"

    relation = ledger.create_ontology_type_relation(
        first["type_id"], second["type_id"], "improves", "Reasoning can improve capability"
    )
    assert relation["relation_name"] == "improves"
    dot = ontology_dot(ledger.ontology_types(), ledger.ontology_type_relations())
    assert "Reasoning Method" in dot
    assert "improves" in dot


def test_ontology_search_can_expand_card_relations(tmp_path: Path) -> None:
    cards = [
        {"card_id": "a", "title": "Reflection method", "claim": "Reflection revises agent action after feedback.", "labels": ["reflection"], "concepts": [], "applies_to": [], "excludes": [], "supports_question_types": [], "conditions": "", "limits": "", "evidence_excerpt": "", "status": "verified"},
        {"card_id": "b", "title": "Agent evaluation", "claim": "Evaluation measures agent task capability.", "labels": ["evaluation"], "concepts": [], "applies_to": [], "excludes": [], "supports_question_types": [], "conditions": "", "limits": "", "evidence_excerpt": "", "status": "verified"},
    ]
    relations = [{"relation_id": "r1", "source_card_id": "a", "target_card_id": "b", "relation_type": "supports", "confidence": "high"}]
    hits = search_cards_for_ontology(
        KnowledgeRetriever(tmp_path / "index.json"), cards, relations, "reflection", use_embedding=False, use_relations=True
    )
    assert [hit.result.card["card_id"] for hit in hits] == ["a", "b"]


def test_inline_card_type_assignment_replaces_previous_type(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    first = ledger.create_ontology_type("Memory Mechanism")
    second = ledger.create_ontology_type("Reasoning Method")

    ledger.set_card_ontology_type("kc-1", first["type_id"])
    assert [item["name"] for item in ledger.ontology_types_for_card("kc-1")] == ["Memory Mechanism"]

    ledger.set_card_ontology_type("kc-1", second["type_id"])
    assert [item["name"] for item in ledger.ontology_types_for_card("kc-1")] == ["Reasoning Method"]
    assert ledger.ontology_card_ids(first["type_id"]) == []

    ledger.set_card_ontology_type("kc-1", None)
    assert ledger.ontology_types_for_card("kc-1") == []


def test_context_graph_contains_only_focus_and_one_hop(tmp_path: Path) -> None:
    from research_fellow.application.ontology import ontology_context_dot

    ledger = Ledger(tmp_path / "ledger.db")
    focus = ledger.create_ontology_type("Reasoning Method")
    neighbour = ledger.create_ontology_type("Agent Capability")
    remote = ledger.create_ontology_type("Evaluation Method")
    ledger.create_ontology_type_relation(focus["type_id"], neighbour["type_id"], "improves")
    ledger.create_ontology_type_relation(neighbour["type_id"], remote["type_id"], "evaluated_by")

    dot = ontology_context_dot(ledger.ontology_types(), ledger.ontology_type_relations(), focus["type_id"])
    assert "Reasoning Method" in dot
    assert "Agent Capability" in dot
    assert "improves" in dot
    assert "Evaluation Method" not in dot
    assert "evaluated_by" not in dot


def test_facet_and_multi_type_assignment(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    requirement = ledger.create_ontology_facet("Requirement")
    agent_case = ledger.create_ontology_facet("Agent Case")
    target = ledger.create_ontology_facet("Target Domain")

    nfr = ledger.create_ontology_type("Non-functional Requirement", facet_id=requirement["facet_id"])
    architect = ledger.create_ontology_type("Architect Agent", facet_id=agent_case["facet_id"])
    architecture = ledger.create_ontology_type("Software Architecture", facet_id=target["facet_id"])

    ledger.set_card_ontology_types("card-1", [nfr["type_id"], architect["type_id"], architecture["type_id"]])
    assigned = ledger.ontology_types_for_card("card-1")
    assert {item["name"] for item in assigned} == {
        "Non-functional Requirement", "Architect Agent", "Software Architecture"
    }
    assert {item["facet_name"] for item in assigned} == {"Requirement", "Agent Case", "Target Domain"}

    ledger.unassign_card_from_ontology_type("card-1", nfr["type_id"])
    assert {item["name"] for item in ledger.ontology_types_for_card("card-1")} == {
        "Architect Agent", "Software Architecture"
    }


def test_deleting_facet_keeps_types(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db")
    facet = ledger.create_ontology_facet("Target Domain")
    ontology_type = ledger.create_ontology_type("Software Architecture", facet_id=facet["facet_id"])
    assert ledger.delete_ontology_facet(facet["facet_id"])
    refreshed = {item["type_id"]: item for item in ledger.ontology_types()}[ontology_type["type_id"]]
    assert refreshed["facet_id"] is None
