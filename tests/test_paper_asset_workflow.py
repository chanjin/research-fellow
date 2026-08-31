import sqlite3
from pathlib import Path

from research_fellow.application.paper_reading import parse_reading_questions, parse_reading_summary, promote_question
from research_fellow.storage import Ledger


def test_paper_asset_keeps_reading_and_promotion_history(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    paper = ledger.upsert_shelf_paper({"title": "A paper", "intake_source": "search"})
    questions = parse_reading_questions(
        "Question: 방법의 적용 범위는 무엇인가?\n"
        "Tentative answer: 세 조건을 분리하여 결과를 비교하는 절차를 제안한다.\n"
        "Evidence: p.3 Method; p.6 Results\nUncertainty: 한 과업에서만 확인되었다."
    )
    question_id = ledger.add_paper_reading_questions(paper["paper_id"], questions)[0]
    item = ledger.paper_reading_questions(paper["paper_id"])[0]
    request_id = promote_question(ledger, paper, item, "연구자 검토")
    ledger.update_paper_reading_question(question_id, researcher_comment="연구자 검토", status="promoted", promotion_request_id=request_id)

    assert ledger.phenomena(type_="decision_request")
    assert ledger.paper_reading_questions(paper["paper_id"])[0]["status"] == "promoted"
    assert any(event["event_type"] == "reading_question_reviewed" for event in ledger.paper_asset_events(paper["paper_id"]))


def test_reading_questions_support_legacy_required_columns(tmp_path: Path) -> None:
    database = tmp_path / "legacy.db"
    with sqlite3.connect(database) as conn:
        conn.execute(
            """CREATE TABLE paper_reading_questions (
                question_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question TEXT NOT NULL,
                tentative_answer TEXT NOT NULL, evidence_json TEXT NOT NULL, uncertainty TEXT NOT NULL,
                research_relevance TEXT NOT NULL, suggested_ontology TEXT NOT NULL,
                researcher_comment TEXT NOT NULL, status TEXT NOT NULL,
                promotion_request_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )"""
        )
    ledger = Ledger(database)
    paper = ledger.upsert_shelf_paper({"title": "Legacy schema paper", "intake_source": "upload"})
    saved = ledger.add_paper_reading_questions(paper["paper_id"], [{
        "question": "무엇을 검증했는가?",
        "tentative_answer": "제한된 조건에서 절차를 검증했다.",
        "evidence": ["p.2 Method"],
        "uncertainty": "표본 범위가 제한적이다.",
    }])

    assert saved
    with sqlite3.connect(database) as conn:
        row = conn.execute(
            "SELECT research_relevance, suggested_ontology FROM paper_reading_questions WHERE question_id=?", (saved[0],)
        ).fetchone()
    assert row == ("", "")


def test_markdown_heading_questions_and_summary_are_all_preserved() -> None:
    output = """**연구 요약**
이 논문은 설계 의사결정에서 LLM의 역할을 비교한다.

---

**질문 1**
*첫 번째 연구자 판단 질문은 무엇인가?*
**Tentative answer**
첫 번째 해석은 원문 결과에 근거한 충분히 긴 설명입니다.
**Evidence**
- p.2: method
**Uncertainty**
한정된 과업에서만 검증되었다.

**질문 2**
*두 번째 연구자 판단 질문은 무엇인가?*
**Tentative answer**
두 번째 해석은 원문 결과에 근거한 충분히 긴 설명입니다.
**Evidence**
- p.6: result
**Uncertainty**
참가자 집단이 제한되었다.

**질문 3**
*세 번째 연구자 판단 질문은 무엇인가?*
**Tentative answer**
세 번째 해석은 원문 결과에 근거한 충분히 긴 설명입니다.
**Evidence**
- p.8: discussion
**Uncertainty**
실제 프로젝트에는 추가 검증이 필요하다.
"""
    assert len(parse_reading_questions(output)) == 3
    assert parse_reading_summary(output) == "이 논문은 설계 의사결정에서 LLM의 역할을 비교한다."


def test_researcher_judgment_question_labels_and_unheaded_summary_are_preserved() -> None:
    output = """코딩 에이전트의 품질 변화를 추적하는 종단적 연구

이 연구는 코딩 에이전트 하네스의 변화가 품질에 미치는 영향을 분석한다. 모델을 고정하고 하네스만
변경하여 성능과 토큰 사용량을 비교했으며, 품질 보증 부재를 주요 원인으로 논의한다.

***
**연구자 판단이 필요한 질문:**
하네스의 복잡도 증가가 기능 향상으로 이어지지 못한 원인은 무엇인가?
**잠정적 답변:**
에이전트의 비기능적 품질을 회귀 검증하는 체계가 부족하기 때문이다.
**증거:** p.5; \"absence of Agentic Quality Assurance\"
**불확실성:** 실제 현업의 다른 제약 조건은 별도 검증이 필요하다.

---
**연구자 판단이 필요한 질문:**
LLM 제공자 계층과 컨텍스트 관리가 고위험 영역인 이유는 무엇인가?
**잠정적 답변:**
두 계층은 모델과 환경 사이의 정보 전달을 직접 제어하기 때문이다.
**증거:** p.5; \"directly govern how information is passed\"
**불확실성:** 미래 아키텍처에서는 위험 양상이 달라질 수 있다.
"""
    questions = parse_reading_questions(output)
    assert len(questions) == 2
    assert questions[0]["evidence"] == ['p.5; "absence of Agentic Quality Assurance"']
    assert "코딩 에이전트의 품질" in parse_reading_summary(output)
