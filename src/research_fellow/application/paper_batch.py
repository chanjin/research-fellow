"""Bounded five-paper P1 intake: download, extract, compare, and draft cards."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from research_fellow.application.claim_curation import build_simple_claim_cards, discovery_prompt, parse_candidate_claims
from research_fellow.infrastructure.document_reader import extract_document

class _BytesUpload:
    def __init__(self, name: str, value: bytes): self.name, self._value = name, value
    def getvalue(self) -> bytes: return self._value

def process_top_papers(profile: dict[str, Any], candidates: list[dict[str, Any]], data_dir: Path, draft_for: Callable[[str], str | None], make_cards: bool = False, review_full_text: bool = True) -> list[dict[str, Any]]:
    order = {"high": 0, "medium": 1, "low": 2, "unreviewed": 3}
    shortlisted = [candidate for candidate in candidates if candidate.get("abstract_shortlist")]
    selected = shortlisted or sorted(candidates, key=lambda c: order.get(c.get("relevance", {}).get("level", "unreviewed"), 3))[:5]
    paper_dir = data_dir / "papers"; paper_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in selected:
        try:
            source_id = re.sub(r"[^A-Za-z0-9._-]", "_", candidate["source_id"])
            pdf_path = paper_dir / f"{source_id}.pdf"
            if not pdf_path.exists():
                pdf_url = candidate["url"].replace("/abs/", "/pdf/") + ".pdf"
                request = Request(pdf_url, headers={"User-Agent": "ResearchFellow/0.1 local-literature-discovery"})
                with urlopen(request, timeout=60) as response: pdf_path.write_bytes(response.read())
            document = extract_document(_BytesUpload(pdf_path.name, pdf_path.read_bytes()), max_pages=20, cache_dir=data_dir / "extracted_documents")
            text = "\n\n".join(page.text for page in document.pages)[:14000]
            relevance = draft_for(_relevance_prompt(profile, candidate, text)) if review_full_text else candidate.get("full_text_review", "")
            relevance = relevance or "본문 기반 적합성 초안을 만들지 못했습니다."
            similarity = _similarity_score(relevance, candidate)
            claims = parse_candidate_claims(draft_for(discovery_prompt(document, max_claims=2)) or "", limit=2) if make_cards else []
            cards = build_simple_claim_cards(document, "외부 논문", claims, profile["keywords"])
            for card in cards:
                card["provenance"] = {"source_name": candidate["title"], "source_url": candidate["url"], "grounding": "full_text_extracted_pending"}
            results.append({**candidate, "pdf_path": str(pdf_path), "full_text_status": "completed", "full_text_review": relevance, "full_text_similarity": similarity, "candidate_cards": cards, "evidence_status": "full_text_extracted_pending"})
        except Exception as error:
            results.append({**candidate, "full_text_status": "failed", "full_text_error": str(error), "candidate_cards": [], "evidence_status": "abstract_only_pending"})
    return results

def _similarity_score(review: str, candidate: dict[str, Any]) -> int:
    match = re.search(r"(?:SIMILARITY|유사도)\s*[:=]\s*(\d{1,3})", review, flags=re.IGNORECASE)
    if match:
        return min(100, int(match.group(1)))
    level = candidate.get("relevance", {}).get("level", "low")
    return {"high": 75, "medium": 50, "low": 25}.get(level, 0)


def _relevance_prompt(profile: dict[str, Any], candidate: dict[str, Any], text: str) -> str:
    return f"""You are M2. Compare the extracted full text of one candidate paper with the approved research profile. Write Korean. First line must be exactly `SIMILARITY: NN` where NN is an integer 0-100 for fit to the approved Intent. Then write 3 short bullets: direct relevance, useful method/result/limit, and what remains unconfirmed. Use only the text. This is candidate triage, not verified knowledge.\n\nApproved Intent title: {profile['title']}\nResearch question: {profile['question']}\nApproved Intent context (purpose, expected evidence, completion condition): {profile['context']}\nSearch keywords are only auxiliary: {', '.join(profile['keywords'])}\nPaper: {candidate['title']}\nExtracted text:\n{text}"""
