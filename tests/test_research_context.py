from research_fellow.application.advising import parse_research_context_mapping
from research_fellow.domain.research import ResearchState


def test_researcher_note_is_research_state_not_knowledge_card() -> None:
    state = ResearchState(
        question="대규모 언어 모델이 생성한 디자인 개념의 실현 가능성은 어떻게 평가하는가?",
        researcher_note="제조 가능성과 사용자 검증을 함께 봐야 한다.",
    )

    assert state.researcher_note.startswith("제조 가능성")
    assert "researcher_note" in state.model_dump()


def test_context_mapping_parser_keeps_researcher_confirmation_fields() -> None:
    draft = """Current hypothesis: 실현 가능성은 다차원 기준으로 평가해야 한다.
Constraints:
- 제조 자료가 제한적이다.
Unresolved issues:
- 평가 기준의 가중치를 정해야 한다.
Recent evidence changes:
- 새 M1 카드가 사용자 검증의 중요성을 제시했다.
"""

    assert parse_research_context_mapping(draft) == {
        "hypothesis": "실현 가능성은 다차원 기준으로 평가해야 한다.",
        "constraints": ["제조 자료가 제한적이다."],
        "unresolved": ["평가 기준의 가중치를 정해야 한다."],
        "changes": ["새 M1 카드가 사용자 검증의 중요성을 제시했다."],
    }
