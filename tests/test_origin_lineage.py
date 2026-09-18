from research_fellow.memory import KnowledgeMemory
from research_fellow.origin_lineage import cards_for_origin, merge_origin_links, origin_labels, origin_research_context
from research_fellow.storage import Ledger


def _paper_origin():
    return {
        "origin_type": "paper_writing",
        "origin_id": "pp-1",
        "origin_sub_id": "todo-1",
        "label": "4대 통제조건 근거 보완",
        "source_card_ids": [],
    }


def test_origin_lineage_propagates_from_profile_to_paper_and_card(tmp_path):
    ledger = Ledger(tmp_path / "research.db")
    memory = KnowledgeMemory(tmp_path / "research.db")
    profile = ledger.create_search_profile({
        "intent_id": "intent-1", "title": "Find evidence", "question": "What evidence exists?",
        "origin_links": [_paper_origin()],
    })
    assert origin_labels(profile["origin_links"]) == ["논문 작성 · 4대 통제조건 근거 보완"]
    assert profile["origin_links"][0]["research_title"] == "Find evidence"
    assert profile["origin_links"][0]["research_question"] == "What evidence exists?"

    paper = ledger.upsert_shelf_paper({
        "title": "A paper", "source_id": "doi:1", "origin_links": profile["origin_links"],
    })
    card = memory.add({
        "title": "A card", "source_kind": "external_paper",
        "claim": "This is a sufficiently detailed research claim.",
        "provenance": {"source_name": paper["title"], "paper_id": paper["paper_id"]},
        "origin_links": paper["origin_links"],
    })
    assert cards_for_origin([card], ["pp-1"])[0]["card_id"] == card["card_id"]


def test_duplicate_paper_accumulates_distinct_origins(tmp_path):
    ledger = Ledger(tmp_path / "research.db")
    first = ledger.upsert_shelf_paper({"title": "Same paper", "source_id": "doi:1", "origin_links": [_paper_origin()]})
    second = ledger.upsert_shelf_paper({
        "title": "Same paper", "source_id": "doi:1",
        "origin_links": [{
            "origin_type": "researcher_question", "origin_id": "rq-1",
            "label": "Can agents safely define their jobs?", "source_card_ids": [],
        }],
    })
    assert first["paper_id"] == second["paper_id"]
    assert len(second["origin_links"]) == 2


def test_origin_merge_deduplicates_by_stable_identity():
    enriched = {**_paper_origin(), "research_question": "어떤 통제 조건이 필요한가?"}
    merged = merge_origin_links([_paper_origin()], [enriched])
    assert len(merged) == 1
    assert merged[0]["research_question"] == "어떤 통제 조건이 필요한가?"


def test_origin_research_context_preserves_title_question_and_intent():
    context = origin_research_context([{
        **_paper_origin(),
        "research_title": "산업 에이전트 통제 조건",
        "research_question": "자율 직무 창발은 어디까지 허용할 수 있는가?",
        "research_context": "고위험 제조 환경의 실패비용을 중심으로 검토한다.",
    }])
    assert "연구 제목: 산업 에이전트 통제 조건" in context
    assert "연구 질문: 자율 직무 창발은 어디까지 허용할 수 있는가?" in context
    assert "의도 맥락: 고위험 제조 환경" in context
