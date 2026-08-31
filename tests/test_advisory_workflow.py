from pathlib import Path

from research_fellow.application.advisory_workflow import collect_evidence_clusters, parse_advisory_plan
from research_fellow.application.episodic_memory import recall_act_spec, store_advisory_episode
from research_fellow.infrastructure.episodic_retrieval import EpisodicRetriever
from research_fellow.storage import Ledger
from research_fellow.infrastructure.retrieval import KnowledgeRetriever


def _card(card_id: str, claim: str, labels: list[str]) -> dict[str, object]:
    return {
        "card_id": card_id, "title": card_id, "claim": claim, "labels": labels,
        "evidence_excerpt": "source-grounded excerpt", "conditions": "stated condition", "limits": "stated limit",
        "provenance": {"source_name": f"{card_id}.pdf"},
    }


def test_relation_cluster_keeps_seed_and_limited_two_hop_context(tmp_path: Path) -> None:
    cards = [
        _card("K1", "명세 우선 설계는 검토 단위를 명시한다.", ["agent", "specification"]),
        _card("K2", "명시된 검토 단위는 승인 경계를 분명하게 한다.", ["approval"]),
        _card("K3", "승인 경계는 감사 가능성을 높인다.", ["governance"]),
        _card("K4", "연결되지 않은 주장은 포함되지 않아야 한다.", ["noise"]),
    ]
    relations = [
        {"relation_id": "R1", "source_card_id": "K1", "target_card_id": "K2", "relation_type": "supports", "confidence": "high"},
        {"relation_id": "R2", "source_card_id": "K2", "target_card_id": "K3", "relation_type": "qualifies", "confidence": "medium"},
    ]
    cluster = KnowledgeRetriever(tmp_path / "index.json").cluster(cards, relations, "agent specification", seed_limit=1)
    assert cluster.card_ids == ["K1", "K2", "K3"]
    assert cluster.members[2].distance == 2
    assert cluster.members[2].relation_ids == ("R1", "R2")


def test_plan_parser_falls_back_to_three_decision_oriented_subquestions() -> None:
    plan = parse_advisory_plan("", "research_question", "문맥 민감성을 어떻게 측정할 것인가?")
    assert len(plan.subquestions) == 3
    assert "적용 조건" in plan.subquestions[1].question


def test_collection_uses_each_subquestion_as_its_own_deterministic_query(tmp_path: Path) -> None:
    plan = parse_advisory_plan(
        "## Decision question\n측정 방법을 결정한다.\n## Subquestions\n- 평가 지표는 무엇인가?\n- 적용 한계는 무엇인가?",
        "research_question", "문맥 민감성 측정",
    )
    cards = [_card("K1", "평가 지표는 과업 맥락과 함께 정의해야 한다.", ["평가", "지표"])]
    clusters = collect_evidence_clusters(plan, KnowledgeRetriever(tmp_path / "index.json"), cards, [], context="LLM 설계 생성")
    assert len(clusters) == 2
    assert all(cluster.query.endswith("LLM 설계 생성") for _, cluster in clusters)


def test_obsolete_or_excluded_cards_cannot_be_semantic_seed(tmp_path: Path) -> None:
    allowed = _card("K1", "명세 우선 설계는 검토 단위를 명시한다.", ["agent", "specification"])
    excluded = {**_card("K2", "비슷하지만 다른 맥락의 주장", ["agent"]), "excludes": ["legacy agent"]}
    obsolete = {**_card("K3", "더 이상 사용하지 않는 주장", ["agent"]), "status": "obsolete"}
    retriever = KnowledgeRetriever(tmp_path / "index.json")
    # A deterministic stand-in for a successful embedding search, ordered so
    # the test verifies that metadata gates, not ranking alone, select seeds.
    retriever.search = lambda *_args, **_kwargs: [
        type("R", (), {"card": excluded, "score": 0.99})(),
        type("R", (), {"card": obsolete, "score": 0.98})(),
        type("R", (), {"card": allowed, "score": 0.80})(),
    ]
    cluster = retriever.cluster([allowed, excluded, obsolete], [], "legacy agent specification", seed_limit=1)
    assert cluster.card_ids == ["K1"]


def test_closed_advisory_is_stored_as_recallable_episode(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    case_id = ledger.create_case("research", "문맥 민감성 평가")
    episode = store_advisory_episode(
        ledger, case_id=case_id, episode_type="research_question",
        situation_summary="LLM 생성 설계에서 문맥 변화에 대한 평가 지표를 결정한다.",
        decision_question="문맥 민감성 측정 프로토콜을 정할 수 있는가?",
        advisory_plan=["평가 대상 정의", "조건과 한계 확인"], answer="조건부 측정 틀을 제안한다.",
        evidence_card_ids=["K1", "K2"], evidence_relation_ids=["R1"], unresolved_items=["비교 설계안 정의"],
    )
    assert ledger.episode_memories()[0]["episode_id"] == episode.episode_id
    spec = recall_act_spec(
        ledger, EpisodicRetriever(tmp_path / "episodes.json"), situation="LLM 설계 결과의 문맥 민감성을 평가하는 방법",
        active_card_ids={"K1"}, semantic=False,
    )
    assert len(spec.recalled_episodes) == 1
    assert spec.revalidation_card_ids == ("K1",)
    assert spec.response_strategy == "planned_investigation"  # provisional episodes do not shortcut review
