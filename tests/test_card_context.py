from pathlib import Path
from tempfile import TemporaryDirectory

from research_fellow.application.curation import normalize_candidate_draft
from research_fellow.application.ontology_curation import build_curation_context, type_suggestion_prompt
from research_fellow.application.paper_reading import parse_reading_questions
from research_fellow.infrastructure.document_reader import ExtractedPage
from research_fellow.storage import Ledger


class DummyRetriever:
    def search(self, *args, **kwargs):
        return []


def test_card_context_fields_are_parsed_from_qualification_draft():
    source = "Before context. Exact evidence sentence about memory choice. After context explaining the task."
    result = normalize_candidate_draft(
        title="paper.pdf", source_kind="외부 논문", page=ExtractedPage(1, source), labels=[], index=1,
        source_text=source,
        text_draft="""Title: Task-dependent memory retrieval\nClaim: Abstract insight and successful trajectories contribute differently by task.\nEvidence: Exact evidence sentence about memory choice.\nContext: The paper compares reasoning-heavy and embodied interactive tasks.\nImplication: Memory retrieval should adapt to task characteristics.\nSource Excerpt: Before context. Exact evidence sentence about memory choice. After context explaining the task.\nLabels: memory, retrieval\nConditions: Task types differ.\nLimits: The comparison covers the evaluated tasks only.""",
    )
    assert "reasoning-heavy" in result.card["context"]
    assert "adapt" in result.card["implication"]
    assert "Before context" in result.card["source_excerpt"]


def test_paper_reading_parser_keeps_context_implication_and_source_excerpt():
    text = """질문: 어떤 메모리가 과업별로 유리한가?\n잠정 답변: 추론 과업과 행동 과업에서 유리한 기억 형태가 다르다.\n근거: p.2 HotpotQA result; p.3 ALFWorld result\n한계·유보: 두 과업 비교에 한정된다.\n연구 관련성: 메모리 구조 선택에 영향을 준다.\n레이블: memory\n카드 제목: 과업별 메모리 기여 차이\n핵심 개념: insight, trajectory\n적용 대상: HotpotQA, ALFWorld\n적용 조건: 비교 실험 조건\n카드 맥락: 추론 중심 과업과 환경 행동 중심 과업을 비교한다.\n설계 함의: 과업 특성에 따라 기억 인출 방식을 달리할 필요가 있다.\n주변 원문: p.2 abstract insight ... p.3 successful trajectory ..."""
    item = parse_reading_questions(text)[0]
    assert item["suggested_context"].startswith("추론 중심")
    assert "기억 인출" in item["suggested_implication"]
    assert "p.2" in item["suggested_source_excerpt"]


def test_shelf_preserves_abstract_for_ontology_context():
    with TemporaryDirectory() as d:
        ledger = Ledger(Path(d) / "research.db")
        paper = ledger.upsert_shelf_paper({"title": "Memory Paper", "source_id": "1234.5678", "abstract": "This is the abstract."})
        assert ledger.shelf_paper(paper["paper_id"])["abstract"] == "This is the abstract."
