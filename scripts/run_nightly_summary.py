"""Run and persist one local M2 activity-delta summary (for launchd or cron)."""

from __future__ import annotations

import argparse
from pathlib import Path

from research_fellow.application.meaning_summary import (
    delta_inputs, delta_meaning_summary_prompt, deterministic_delta_summary, latest_summary, record_delta_summary,
)
from research_fellow.application.search_profiles import run_profile, scheduled_profiles
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.claim_curation import submit_claim_cards
from research_fellow.llm import ollama_draft_result, set_llm_audit_logger
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.storage import Ledger


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the nightly Research Fellow activity delta summary.")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--without-ollama", action="store_true")
    args = parser.parse_args()
    ledger = Ledger(args.data_dir / "research_fellow.db")
    set_llm_audit_logger(ledger.record_llm_call)
    memory = KnowledgeMemory(args.data_dir / "knowledge_cards.jsonl")
    relations = RelationMemory(args.data_dir / "knowledge_relations.jsonl")
    for profile in scheduled_profiles(ledger):
        draft_for = lambda prompt: ollama_draft_result(prompt, args.model, not args.without_ollama, profile="abstract_triage").text
        outcome = run_profile(ledger, profile, "nightly", reviewer=draft_for)
        if outcome["status"] == "completed" and outcome["candidates"]:
            processed = process_top_papers(profile, outcome["candidates"], args.data_dir, lambda prompt: ollama_draft_result(prompt, args.model, not args.without_ollama, profile="full_text_similarity").text, make_cards=False)
            top_three = sorted(processed, key=lambda item: item.get("full_text_similarity", 0), reverse=True)[:3]
            drafted = process_top_papers(profile, top_three, args.data_dir, lambda prompt: ollama_draft_result(prompt, args.model, not args.without_ollama, profile="p1_card_draft").text, make_cards=True, review_full_text=False)
            drafted = [{**item, "auto_selected_for_cards": True} for item in drafted]
            for candidate in drafted:
                if candidate.get("candidate_cards"):
                    document = type("BatchDocument", (), {"title": candidate["title"]})()
                    submit_claim_cards(ledger, document, candidate["candidate_cards"], ["자동 야간 탐색·PDF 본문 비교를 거친 후보입니다. 원문 발췌·조건·한계를 승인 시 검토하세요."])
            merged = {item["source_id"]: item for item in outcome["candidates"]}
            merged.update({item["source_id"]: item for item in processed})
            merged.update({item["source_id"]: item for item in drafted})
            ledger.update_search_run_candidates(outcome["run_id"], list(merged.values()))
    reports = ledger.phenomena(recipient="researcher", type_="advice_report")
    delta = delta_inputs(memory.all(), relations.all(), reports, latest_summary(ledger))
    result = ollama_draft_result(delta_meaning_summary_prompt(delta), args.model, not args.without_ollama, profile="m2_report")
    summary = result.text if result.ok else deterministic_delta_summary(delta)
    record_delta_summary(ledger, delta, summary, trigger="nightly")
    print("Saved nightly Delta summary." if result.ok else "Saved deterministic nightly Delta summary (Ollama unavailable).")


if __name__ == "__main__":
    main()
