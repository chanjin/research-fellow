from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_fellow.application.curation import _compact_title, normalize_candidate_draft
from research_fellow.application.claim_curation import parse_claim_review, parse_discovered_claims, retained_claims
from research_fellow.application.decisions import decide_request
from research_fellow.application.management import delete_knowledge_card, delete_knowledge_relation
from research_fellow.application.prompt_tasks import claim_verification_prompt, gap_search_plan_prompt, lineage_review_prompt
from research_fellow.application.advising import draft_research_direction, record_research_direction
from research_fellow.application.progressive_curation import ProgressiveCandidate, consolidate_candidates, submit_progressive_candidates
from research_fellow.domain.research import ResearchState
from research_fellow.infrastructure.prompt_templates import PROMPT_CATALOG, validate_prompt_template
from research_fellow.application.relations import create_relation_candidate, lineage_dot, propose_relation_candidates
from research_fellow.infrastructure.document_reader import ExtractedDocument, ExtractedPage, _normalize_pdf_text, _remove_extraction_boilerplate, extract_document, extracted_document_text
from research_fellow.infrastructure.retrieval import KnowledgeRetriever
from research_fellow.infrastructure.retrieval import RetrievalResult
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.services import complete_intent, create_document_candidates, request_curation_intent
from research_fellow.storage import Ledger


class DecisionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.ledger = Ledger(root / "ledger.db")
        self.memory = KnowledgeMemory(root / "knowledge.jsonl")
        self.relations = RelationMemory(root / "relations.jsonl")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_only_approved_cards_enter_semantic_memory(self) -> None:
        requests, warnings = create_document_candidates(
            self.ledger, "paper.pdf", "외부 논문",
            [ExtractedPage(1, "첫 번째 근거 문장은 충분히 긴 연구 결과입니다."), ExtractedPage(2, "두 번째 근거 문장도 충분히 긴 연구 결과입니다.")], ["test"],
        )
        self.assertTrue(warnings)
        self.assertEqual(self.memory.all(), [])
        self.assertEqual(len(requests), 2)

        self.assertTrue(decide_request(self.ledger, self.memory, requests[0], "approved"))
        self.assertTrue(decide_request(self.ledger, self.memory, requests[1], "rejected", "근거가 부족함"))
        self.assertEqual(len(self.memory.all()), 1)
        self.assertEqual(self.memory.all()[0]["evidence_pages"], [])
        self.assertFalse(decide_request(self.ledger, self.memory, requests[0], "approved"))
        self.assertEqual(len(self.memory.all()), 1)

    def test_only_approved_intent_is_exposed_to_m1(self) -> None:
        first = request_curation_intent(self.ledger, "승인 대상", "확인할 질문", ["agent"], "높음")
        second = request_curation_intent(self.ledger, "보류 대상", "다른 질문", [], "보통")
        self.assertEqual(self.ledger.phenomena(recipient="m1", type_="curation_intent", status="ready"), [])

        decide_request(self.ledger, self.memory, first, "approved")
        decide_request(self.ledger, self.memory, second, "deferred")
        queue = self.ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["payload"]["title"], "승인 대상")
        self.assertTrue(complete_intent(self.ledger, queue[0], "탐색 결과"))
        self.assertFalse(complete_intent(self.ledger, queue[0], "중복 결과"))
        updates = self.ledger.phenomena(recipient="m2", type_="knowledge_update")
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]["payload"]["finding"], "탐색 결과")

    def test_p4_question_creates_report_and_at_most_three_approval_intents(self) -> None:
        state = ResearchState(question="명세 우선 설계가 에이전트 검토 품질을 높이는가?")
        draft = draft_research_direction(state, [], [], None)
        self.assertTrue(draft.report)
        self.assertLessEqual(len(draft.intents), 3)
        _, request_ids = record_research_direction(self.ledger, state, [], [], draft)
        self.assertLessEqual(len(request_ids), 3)
        pending = self.ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
        self.assertEqual(len(pending), len(request_ids))
        self.assertTrue(decide_request(self.ledger, self.memory, request_ids[0], "approved"))
        ready = self.ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        self.assertEqual(len(ready), 1)

    def test_text_draft_is_normalized_to_page_grounded_candidate(self) -> None:
        result = normalize_candidate_draft(
            title="note.md", source_kind="연구자의 아이디어 노트",
            page=ExtractedPage(4, "## 검증 범위\n실험은 특정 조건에서만 재현 가능하다.", "검증 범위"),
            labels=["evaluation"], index=1,
            text_draft="주장: 재현성 평가는 적용 조건을 함께 기술해야 한다.\n조건: 동일 장비와 데이터셋\n한계: 다른 도메인 일반화는 미확인",
        )
        card = result.card
        self.assertEqual(card["source_kind"], "researcher_idea_note")
        self.assertEqual(card["evidence_pages"], [])
        self.assertEqual(card["citation_markers"], [])
        self.assertEqual(card["provenance"], {"source_name": "note.md"})
        self.assertIn("근거 발췌", result.warnings[0])

    def test_pdf_hyphenation_and_compact_title(self) -> None:
        self.assertEqual(_normalize_pdf_text("informa-\ntion"), "information")
        self.assertEqual(_normalize_pdf_text("informa\n-\ntion"), "information")
        self.assertTrue(_compact_title("에이전트 명세는 구현 전에 승인 가능한 판단 단위를 명확히 하고 연구자 검토 흐름까지 연결해야 한다.").endswith("…"))

    def test_publication_boilerplate_is_removed_before_llm_curation(self) -> None:
        raw = (
            "JULY/AUGUST 2026 | IEEE SOFTWARE 19\n"
            "This work is licensed under a Creative Commons Attribution 4.0 License.\n"
            "For more information, see https://creativecommons.org/licenses/by/4.0/\n"
            "The study evaluates an explicit review workflow."
        )
        self.assertEqual(
            _normalize_pdf_text(_remove_extraction_boilerplate(raw)),
            "The study evaluates an explicit review workflow.",
        )

    def test_untraceable_llm_evidence_is_replaced_by_source_excerpt(self) -> None:
        page = ExtractedPage(1, "The study evaluates review workflows under constrained context windows.")
        result = normalize_candidate_draft(
            title="paper.pdf", source_kind="외부 논문", page=page, labels=[], index=1,
            text_draft="주장: 제한된 문맥에서는 검토 워크플로우가 필요하다.\n근거: 원문에 없는 한국어 근거 문장\n쪽수: 1\n인용: p.1\n조건: 제한된 문맥\n한계: 추가 검증 필요",
        )
        self.assertEqual(result.card["evidence_excerpt"], page.text)
        self.assertTrue(any("원문 발췌" in warning for warning in result.warnings))

    def test_claim_first_flow_keeps_independent_claims_before_card_constraints(self) -> None:
        claims = parse_discovered_claims(
            "1. Explicit approval boundaries make research-agent collaboration auditable.\n"
            "2. Explicit approval boundaries make research-agent collaboration auditable.\n"
            "3. Creative Commons Attribution 4.0 License"
        )
        self.assertEqual(len(claims), 1)
        reviews = parse_claim_review("Claim: C1\nDecision: retain\nMerge with:\nReason: Independent finding", claims)
        self.assertEqual(retained_claims(reviews), claims)

    def test_minimal_claim_card_can_be_saved_after_researcher_approval(self) -> None:
        card = self.memory.add({
            "card_id": "kc-minimal", "title": "Auditable approval boundaries", "source_kind": "external_paper",
            "claim": "Explicit approval boundaries improve collaboration auditability.", "labels": ["governance"],
            "provenance": {"source_name": "paper.pdf", "grounding": "source_document_only"},
        })
        self.assertEqual(card["evidence_excerpt"], "")
        self.assertEqual(card["conditions"], "")

    def test_python_extracted_text_export_keeps_page_markers(self) -> None:
        document = ExtractedDocument(
            document_id="pdf-test", file_name="paper.pdf", title="Test paper", author="Author",
            source_format="pdf", original_page_count=2,
            pages=[ExtractedPage(1, "첫 쪽 텍스트"), ExtractedPage(2, "둘째 쪽 텍스트", "METHOD")],
            extraction_note="텍스트만 사용합니다.",
        )
        text = extracted_document_text(document)
        self.assertIn("===== Extracted segment 1 =====", text)
        self.assertIn("===== Extracted segment 2 · METHOD =====", text)
        self.assertIn("둘째 쪽 텍스트", text)

    def test_text_extraction_cache_reuses_derived_content(self) -> None:
        class Upload:
            name = "research-note.md"

            def getvalue(self) -> bytes:
                return b"# Research note\n\nA page-addressable note."

        cache = Path(self.directory.name) / "extracted_documents"
        first = extract_document(Upload(), cache_dir=cache)
        second = extract_document(Upload(), cache_dir=cache)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(second.pages[0].blocks[0].block_id, "p1-b01")
        self.assertEqual(len(list(cache.glob("*.json"))), 1)

    def test_memory_rejects_card_without_required_evidence(self) -> None:
        with self.assertRaises(Exception):
            self.memory.add({"card_id": "bad", "title": "불완전 카드", "source_kind": "external_paper", "claim": "근거 없음"})

    def test_only_approved_relation_enters_relation_memory_and_lineage_projection(self) -> None:
        source = self.memory.add({
            "card_id": "kc-source", "title": "출발 지식", "source_kind": "external_paper", "claim": "출발 지식의 검증 가능한 주장은 충분히 길다.",
            "labels": [], "evidence_excerpt": "출발 카드의 근거 발췌입니다.", "evidence_pages": [1], "citation_markers": ["p.1"],
            "conditions": "조건", "limits": "한계", "provenance": {"source_name": "a", "page_or_section": "p.1"},
        })
        target = self.memory.add({
            "card_id": "kc-target", "title": "도착 지식", "source_kind": "external_paper", "claim": "도착 지식의 검증 가능한 주장은 충분히 길다.",
            "labels": [], "evidence_excerpt": "도착 카드의 근거 발췌입니다.", "evidence_pages": [2], "citation_markers": ["p.2"],
            "conditions": "조건", "limits": "한계", "provenance": {"source_name": "b", "page_or_section": "p.2"},
        })
        request_id, _ = create_relation_candidate(
            self.ledger, source["card_id"], target["card_id"], "qualifies", "", "두 카드의 적용 조건이 다르다.", "같은 평가 기준", "medium",
        )
        self.assertEqual(self.relations.all(), [])
        self.assertTrue(decide_request(self.ledger, self.memory, request_id, "approved", relation_memory=self.relations))
        self.assertEqual(len(self.relations.all()), 1)
        dot = lineage_dot([source, target], self.relations.all())
        self.assertIn("qualifies (medium)", dot)

        self.assertTrue(delete_knowledge_relation(self.ledger, self.relations, self.relations.all()[0]["relation_id"], "잘못된 관계"))
        self.assertEqual(self.relations.all(), [])

    def test_card_tombstone_hides_connected_relation_without_erasing_audit_record(self) -> None:
        source = self.memory.add({
            "card_id": "kc-a", "title": "A", "source_kind": "external_paper", "claim": "A 카드의 검증 가능한 주장은 충분히 길다.", "labels": ["test"],
            "evidence_excerpt": "A 근거 발췌", "evidence_pages": [1], "citation_markers": ["p.1"], "conditions": "조건", "limits": "한계", "provenance": {"source_name": "a", "page_or_section": "p.1"},
        })
        target = self.memory.add({
            "card_id": "kc-b", "title": "B", "source_kind": "external_paper", "claim": "B 카드의 검증 가능한 주장은 충분히 길다.", "labels": ["test"],
            "evidence_excerpt": "B 근거 발췌", "evidence_pages": [2], "citation_markers": ["p.2"], "conditions": "조건", "limits": "한계", "provenance": {"source_name": "b", "page_or_section": "p.2"},
        })
        request_id, _ = create_relation_candidate(self.ledger, source["card_id"], target["card_id"], "supports", "", "A와 B가 같은 주장을 검토한다.", "조건", "medium")
        decide_request(self.ledger, self.memory, request_id, "approved", relation_memory=self.relations)
        self.assertTrue(delete_knowledge_card(self.ledger, self.memory, source["card_id"], "출처 철회"))
        self.assertEqual(len(self.memory.all()), 1)
        self.assertIn(source["card_id"], {card["card_id"] for card in self.memory.all(include_deleted=True)})
        self.assertEqual(self.relations.active_for_cards({card["card_id"] for card in self.memory.all()}), [])

    def test_relation_proposals_are_created_without_manual_relation_form(self) -> None:
        cards = []
        for card_id, title in (("kc-1", "첫 카드"), ("kc-2", "둘 카드")):
            cards.append(self.memory.add({
                "card_id": card_id, "title": title, "source_kind": "external_paper", "claim": f"{title}의 검증 가능한 주장은 충분히 길다.", "labels": ["shared"],
                "evidence_excerpt": f"{title} 근거 발췌", "evidence_pages": [1], "citation_markers": ["p.1"], "conditions": "조건", "limits": "한계", "provenance": {"source_name": title, "page_or_section": "p.1"},
            }))
        requests, _ = propose_relation_candidates(self.ledger, cards, [], draft_for=lambda _a, _b: None)
        self.assertEqual(len(requests), 1)
        pending = self.ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
        self.assertEqual(pending[0]["subject_type"], "knowledge_relation")

    def test_p3_lexical_retrieval_returns_card_source_and_reason(self) -> None:
        card = self.memory.add({
            "card_id": "kc-search", "title": "명세 우선 설계", "source_kind": "external_paper",
            "claim": "에이전트 명세는 구현 전에 검토되어야 한다.", "labels": ["agent", "specification"],
            "evidence_excerpt": "명세 검토는 구현 오류를 줄인다.", "evidence_pages": [3], "citation_markers": ["p.3"],
            "conditions": "초기 설계 단계", "limits": "단일 사례", "provenance": {"source_name": "paper.pdf", "page_or_section": "p.3"},
        })
        results = KnowledgeRetriever(Path(self.directory.name) / "retrieval.json").search([card], "agent specification")
        self.assertEqual(results[0].card["card_id"], "kc-search")
        self.assertEqual(results[0].method, "lexical")
        self.assertIn("키워드 일치", results[0].reason)

    def test_additional_jinja_tasks_use_only_python_assigned_references(self) -> None:
        card = {
            "card_id": "kc-prompt", "title": "근거 카드", "claim": "검증 가능한 주장입니다.",
            "evidence_excerpt": "원문 근거입니다.", "evidence_pages": [2], "labels": ["evaluation"],
            "conditions": "초기 설계", "limits": "단일 사례",
            "provenance": {"source_name": "paper.pdf", "page_or_section": "p.2"},
        }
        verification = claim_verification_prompt("명세가 중요하다", [card])
        gap_plan = gap_search_plan_prompt("비교 연구가 부족함", "평가 방법", [card])
        lineage = lineage_review_prompt("명세", [card])
        self.assertIn("[E1]", verification)
        self.assertIn("[E1]", gap_plan)
        self.assertIn("[K1]", lineage)
        self.assertNotIn("kc-prompt", gap_plan.split("Existing approved evidence:", 1)[0])

    def test_registered_prompt_assets_compile_before_the_developer_ui_saves_them(self) -> None:
        self.assertGreaterEqual(len(PROMPT_CATALOG), 10)
        for template in PROMPT_CATALOG:
            source = (Path(__file__).parents[1] / "src" / "research_fellow" / "prompts" / template.name).read_text(encoding="utf-8")
            validate_prompt_template(source)
        with self.assertRaises(ValueError):
            validate_prompt_template("{% for")

    def test_p4_direction_creates_at_most_three_approved_gated_intents(self) -> None:
        state = ResearchState(
            question="명세 우선 설계의 검증 근거는 충분한가?",
            unresolved_issues=["반대 사례", "평가 지표", "적용 조건", "네 번째 항목"],
        )
        draft = draft_research_direction(state, [], [], None)
        self.assertLessEqual(len(draft.intents), 3)
        _, requests = record_research_direction(self.ledger, state, [], [], draft)
        self.assertEqual(len(requests), 3)
        self.assertEqual(self.ledger.phenomena(recipient="m1", type_="curation_intent", status="ready"), [])
        self.assertTrue(decide_request(self.ledger, self.memory, requests[0], "approved"))
        queue = self.ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        self.assertEqual(len(queue), 1)
        self.assertTrue(queue[0]["payload"]["expected_evidence"])
        self.assertTrue(queue[0]["payload"]["completion_condition"])

    def test_progressive_llm_duplicate_judgement_keeps_independent_card_set_clean(self) -> None:
        def candidate(identifier: str, page: int) -> ProgressiveCandidate:
            return ProgressiveCandidate(identifier, {
                "card_id": identifier, "title": f"후보 {identifier}", "source_kind": "external_paper",
                "claim": "같은 검증 가능한 연구 주장을 제시합니다.", "labels": ["evaluation"],
                "evidence_excerpt": "동일한 원문 근거 발췌입니다.", "evidence_pages": [page], "citation_markers": [f"p.{page}"],
                "conditions": "같은 적용 조건", "limits": "단일 문헌",
                "provenance": {"source_name": "paper.pdf", "page_or_section": f"p.{page}"},
            }, page, [f"p{page}-b01"], [])
        first, second = candidate("kc-page-1", 1), candidate("kc-page-2", 2)
        review = "후보: kc-page-1, kc-page-2\n관계: duplicate\n처리: merge_into_first\n이유: 같은 주장을 반복한다."
        kept, decisions, _ = consolidate_candidates([first, second], review)
        self.assertEqual(len(kept), 1)
        self.assertEqual(decisions[0].relation, "duplicate")
        requests = submit_progressive_candidates(self.ledger, ExtractedDocument(
            "pdf-test", "paper.pdf", "paper", "", "pdf", 2, [], "test"
        ), kept, decisions, [])
        self.assertEqual(self.memory.all(), [])
        self.assertTrue(decide_request(self.ledger, self.memory, requests[0], "approved"))
        self.assertEqual(len(self.memory.all()), 1)


if __name__ == "__main__":
    unittest.main()
