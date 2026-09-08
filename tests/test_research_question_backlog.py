from pathlib import Path

from research_fellow.application.advising import (
    create_exploration_intent_for_rq,
    parse_research_question_suggestions,
    store_research_question_candidates,
)
from research_fellow.storage import Ledger


def test_parse_rich_rq_candidate():
    text = """## RQ 1
Question: LLM 기반 아키텍트 에이전트는 복합 제약을 어떻게 반영해야 하는가?
Why Now: 최근 카드에서 개별 설계 질문에는 대안을 제공하지만 실제 프로젝트의 다중 제약에는 한계가 있다는 결과가 제시되었다.
Gap/Tension: 개별 결정의 품질과 상호의존적 결정 집합의 품질 사이에 공백이 있다.
Research Context: Architect Agent의 의사결정 워크플로우 설계와 직접 연결된다.
Source Card IDs: kc-1, kc-2
Exploration Need: NFR trade-off와 architecture decision dependency를 다루는 방법 및 사례가 필요하다.
"""
    items = parse_research_question_suggestions(text, valid_card_ids={"kc-1", "kc-2", "kc-3"})
    assert len(items) == 1
    assert items[0].source_card_ids == ["kc-1", "kc-2"]
    assert "복합 제약" in items[0].question
    assert "상호의존" in items[0].gap_or_tension


def test_backlog_lifecycle_and_intent_link(tmp_path: Path):
    ledger = Ledger(tmp_path / "local.db")
    text = """## RQ 1
Question: 아키텍트 에이전트는 실제 프로젝트의 복합 제약을 어떻게 처리해야 하는가?
Why Now: M1의 새 정보가 개별 질문과 프로젝트 수준 의사결정 사이의 차이를 보여준다.
Gap/Tension: 복수 제약과 의존 결정의 결합 방식이 불명확하다.
Research Context: Architect Agent 연구의 다음 탐색 축이다.
Source Card IDs: kc-1
Exploration Need: 제약 표현, NFR trade-off, decision dependency 관련 근거가 필요하다.
"""
    candidate = parse_research_question_suggestions(text, valid_card_ids={"kc-1"})[0]
    updates = [{"phenomenon_id": "ph-u1", "payload": {"card_id": "kc-1"}}]
    saved = store_research_question_candidates(ledger, [candidate], updates)
    rq = saved[0]
    assert rq["status"] == "candidate"
    assert rq["source_update_ids"] == ["ph-u1"]

    assert ledger.update_research_question_status(rq["rq_id"], "interested")
    assert ledger.research_question(rq["rq_id"])["status"] == "interested"

    intent_id, request_id = create_exploration_intent_for_rq(ledger, ledger.research_question(rq["rq_id"]))
    refreshed = ledger.research_question(rq["rq_id"])
    assert refreshed["status"] == "exploring"
    links = ledger.research_question_intents(rq["rq_id"])
    assert links[0]["intent_id"] == intent_id
    assert links[0]["request_id"] == request_id
    request = next(item for item in ledger.phenomena(type_="decision_request") if item["phenomenon_id"] == request_id)
    assert request["payload"]["intent"]["question"] == rq["question"]
    assert "Why this question emerged" in request["payload"]["intent"]["research_context"]


def test_regeneration_preserves_researcher_status(tmp_path: Path):
    ledger = Ledger(tmp_path / "local.db")
    first = {
        "question": "동일 질문인가?",
        "rationale": "첫 번째 충분한 도출 이유입니다.",
        "status": "candidate",
    }
    saved = ledger.upsert_research_question(first)
    ledger.update_research_question_status(saved["rq_id"], "interested")
    ledger.upsert_research_question({
        "question": "동일 질문인가?",
        "rationale": "새로운 M1 정보로 도출 이유가 보강되었습니다.",
        "gap_or_tension": "새로운 공백이 확인되었습니다.",
    })
    rq = ledger.research_question(saved["rq_id"])
    assert rq["status"] == "interested"
    assert "보강" in rq["rationale"]


def test_rq_backlog_syncs_between_workspaces(tmp_path: Path):
    from research_fellow.workspace_sync import WorkspaceSync

    school_db = tmp_path / "school.db"
    server_dir = tmp_path / "server"
    school = Ledger(school_db)
    rq = school.upsert_research_question({
        "question": "동기화되는 연구질문인가?",
        "rationale": "학교에서 새로 도출한 질문을 집에서도 이어서 검토해야 한다.",
        "status": "interested",
    })
    WorkspaceSync(school_db, server_dir).initialize_server_from_local()

    home_db = tmp_path / "home.db"
    Ledger(home_db)
    home_sync = WorkspaceSync(home_db, server_dir)
    preview = home_sync.preview()
    assert any(change.table == "research_questions" and change.direction == "server→local" for change in preview.changes)
    home_sync.apply()
    home = Ledger(home_db)
    copied = home.research_question(rq["rq_id"])
    assert copied is not None
    assert copied["status"] == "interested"


def test_auto_priority_and_direct_m1_dispatch(tmp_path: Path):
    from research_fellow.application.advising import (
        auto_exploration_candidates,
        dispatch_top_research_questions,
        parse_rq_priority_assessment,
    )

    ledger = Ledger(tmp_path / "local.db")
    rq1 = ledger.upsert_research_question({
        "question": "Architect Agent는 복합 제약을 어떻게 다뤄야 하는가?",
        "rationale": "새 근거가 다중 제약 처리 한계를 보여준다.",
        "gap_or_tension": "결정 의존성과 NFR trade-off가 연결되지 않았다.",
        "research_context": "Architect Agent 연구의 핵심 설계 문제다.",
        "exploration_need": "NFR trade-off 및 decision dependency 사례 탐색",
        "status": "interested",
    })
    rq2 = ledger.upsert_research_question({
        "question": "장기 실행 Agent의 메모리는 어떻게 관리해야 하는가?",
        "rationale": "장기 실행 중 상태 누적 문제가 확인되었다.",
        "status": "candidate",
    })
    rq3 = ledger.upsert_research_question({
        "question": "프롬프트 표현은 어떻게 정리해야 하는가?",
        "rationale": "표현 방식에 대한 추가 확인이 필요하다.",
        "status": "hold",
    })

    eligible = auto_exploration_candidates(ledger)
    assert {item["rq_id"] for item in eligible} == {rq1["rq_id"], rq2["rq_id"]}
    ranking_text = f"""## Priority 1
RQ_ID: {rq1['rq_id']}
SCORE: 5
REASON: 현재 연구 방향과 직접 연결되고 새 근거가 강하다.

## Priority 2
RQ_ID: {rq2['rq_id']}
SCORE: 4
REASON: 장기 실행 안정성에 중요하다.
"""
    ranked = parse_rq_priority_assessment(ranking_text, eligible, limit=3)
    dispatched = dispatch_top_research_questions(ledger, ranked, limit=3)
    assert len(dispatched) == 2

    ready = ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
    assert len(ready) == 2
    assert all(item["payload"]["execution_mode"] == "auto" for item in ready)
    assert all(item["payload"]["created_by"] == "m2_auto" for item in ready)
    assert len(ledger.search_profiles(active_only=True)) == 2
    assert ledger.research_question(rq1["rq_id"])["status"] == "exploring"
    assert ledger.research_question(rq2["rq_id"])["status"] == "exploring"

    # Already-dispatched questions are not selected again.
    assert auto_exploration_candidates(ledger) == []


def test_auto_dispatch_reuses_duplicate_intent(tmp_path: Path):
    from research_fellow.application.advising import dispatch_top_research_questions

    ledger = Ledger(tmp_path / "local.db")
    rq1 = ledger.upsert_research_question({
        "question": "Architect Agent는 복합 제약과 NFR trade-off를 어떻게 처리해야 하는가?",
        "rationale": "복합 제약 처리 한계가 확인되었다.",
        "exploration_need": "NFR trade-off와 decision dependency 조사",
        "status": "interested",
    })
    rq2 = ledger.upsert_research_question({
        "question": "Architect Agent는 NFR trade-off와 복합 제약을 어떻게 처리해야 하는가?",
        "rationale": "동일한 연구 공백에서 파생된 질문이다.",
        "exploration_need": "NFR trade-off와 decision dependency 조사",
        "status": "interested",
    })
    first = dispatch_top_research_questions(ledger, [{"rq": rq1, "score": 5, "reason": "핵심"}], limit=1)
    second = dispatch_top_research_questions(ledger, [{"rq": rq2, "score": 5, "reason": "핵심"}], limit=1)

    assert len(ledger.phenomena(type_="curation_intent")) == 1
    assert second[0]["reused"] is True
    assert first[0]["intent"].intent_id == second[0]["intent"].intent_id
    assert ledger.research_question_intents(rq2["rq_id"])[0]["intent_id"] == first[0]["intent"].intent_id


def _record_card_update(ledger: Ledger, card_id: str, title: str = "새 카드") -> str:
    case_id = ledger.create_case("research", title)
    return ledger.record(
        case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card",
        {"title": title, "card_id": card_id}, card_id, status="completed",
    )


def test_m1_new_information_is_unreviewed_card_set_and_rq_keeps_batch_sources(tmp_path: Path):
    from research_fellow.application.advising import recent_knowledge_updates

    ledger = Ledger(tmp_path / "local.db")
    first = _record_card_update(ledger, "kc-1", "첫 카드")
    second = _record_card_update(ledger, "kc-1", "첫 카드 보강")
    third = _record_card_update(ledger, "kc-2", "둘째 카드")

    updates = recent_knowledge_updates(ledger, limit=100)
    assert {item["payload"]["card_id"] for item in updates} == {"kc-1", "kc-2"}
    kc1 = next(item for item in updates if item["payload"]["card_id"] == "kc-1")
    assert set(kc1["pending_update_ids"]) == {first, second}

    review_id = ledger.create_research_state_review("manual", updates)
    text = """## RQ 1
Question: 새 카드들이 제시한 설계 한계를 어떻게 검증해야 하는가?
Why Now: 두 지식카드가 서로 다른 설계 한계를 보여주어 함께 검토할 필요가 있다.
Gap/Tension: 한계의 적용 조건이 아직 명확하지 않다.
Research Context: 현재 연구 방향을 구체화하는 질문이다.
Source Card IDs: kc-1, kc-2
Exploration Need: 적용 조건과 반례를 비교할 문헌이 필요하다.
"""
    candidate = parse_research_question_suggestions(text, valid_card_ids={"kc-1", "kc-2"})[0]
    saved = store_research_question_candidates(ledger, [candidate], updates, review_id=review_id)
    ledger.complete_research_state_review(review_id, generated_rq_count=1, summary="2건 검토")

    assert recent_knowledge_updates(ledger, limit=100) == []
    rq = saved[0]
    sources = ledger.research_question_sources(rq["rq_id"])
    assert {item["card_id"] for item in sources} == {"kc-1", "kc-2"}
    assert {item["review_id"] for item in sources} == {review_id}
    assert first in ledger.reviewed_knowledge_update_ids()
    assert second in ledger.reviewed_knowledge_update_ids()
    assert third in ledger.reviewed_knowledge_update_ids()


def test_existing_rq_accumulates_source_cards_across_review_batches(tmp_path: Path):
    from research_fellow.application.advising import recent_knowledge_updates

    ledger = Ledger(tmp_path / "local.db")
    _record_card_update(ledger, "kc-a")
    updates1 = recent_knowledge_updates(ledger)
    r1 = ledger.create_research_state_review("manual", updates1)
    c1 = parse_research_question_suggestions("""## RQ 1
Question: 동일 연구질문을 계속 보강할 수 있는가?
Why Now: 첫 번째 카드가 질문의 초기 근거를 제공한다.
Source Card IDs: kc-a
Exploration Need: 추가 근거가 필요하다.
""", valid_card_ids={"kc-a"})[0]
    saved1 = store_research_question_candidates(ledger, [c1], updates1, review_id=r1)[0]
    ledger.complete_research_state_review(r1, generated_rq_count=1)

    _record_card_update(ledger, "kc-b")
    updates2 = recent_knowledge_updates(ledger)
    r2 = ledger.create_research_state_review("manual", updates2)
    c2 = parse_research_question_suggestions("""## RQ 1
Question: 동일 연구질문을 계속 보강할 수 있는가?
Why Now: 두 번째 카드가 다른 조건의 근거를 추가한다.
Source Card IDs: kc-b
Exploration Need: 조건 비교가 필요하다.
""", valid_card_ids={"kc-b"})[0]
    saved2 = store_research_question_candidates(ledger, [c2], updates2, review_id=r2)[0]
    ledger.complete_research_state_review(r2, generated_rq_count=1)

    assert saved1["rq_id"] == saved2["rq_id"]
    rq = ledger.research_question(saved1["rq_id"])
    assert set(rq["source_card_ids"]) == {"kc-a", "kc-b"}
    assert {item["review_id"] for item in ledger.research_question_sources(rq["rq_id"])} == {r1, r2}
