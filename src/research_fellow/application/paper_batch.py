"""Bounded five-paper P1 intake: download, extract, compare, and draft cards."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

from research_fellow.infrastructure.prompt_renderer import apply_review_language_policy
from research_fellow.application.claim_curation_cards import build_simple_claim_cards
from research_fellow.application.claim_curation_prompts import discovery_prompt
from research_fellow.application.claim_curation_parsers import parse_candidate_claims
from research_fellow.application.llm_retry import LLMRetryExhausted
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
                pdf_url = str(candidate.get("pdf_url") or "").strip()
                source_url = str(candidate.get("url") or "").strip()
                if not pdf_url and "arxiv.org/abs/" in source_url:
                    pdf_url = source_url.replace("/abs/", "/pdf/") + ".pdf"
                if not pdf_url:
                    raise ValueError("검증된 PDF URL이 없어 본문 비교를 자동 수행할 수 없습니다.")
                request = Request(pdf_url, headers={"User-Agent": "ResearchFellow/0.1 local-literature-discovery"})
                with urlopen(request, timeout=60) as response: pdf_path.write_bytes(response.read())
            document = extract_document(_BytesUpload(pdf_path.name, pdf_path.read_bytes()), max_pages=20, cache_dir=data_dir / "extracted_documents")
            text = "\n\n".join(page.text for page in document.pages)[:14000]
            relevance = draft_for(_relevance_prompt(profile, candidate, text)) if review_full_text else candidate.get("full_text_review", "")
            relevance = relevance or "본문 기반 적합성 초안을 만들지 못했습니다."
            similarity = _similarity_score(relevance, candidate)
            paper_summary, knowledge_candidates = _parse_review_assets(relevance)
            claims = parse_candidate_claims(draft_for(discovery_prompt(document, max_claims=2)) or "", limit=2) if make_cards else []
            cards = build_simple_claim_cards(document, "외부 논문", claims, profile["keywords"])
            for card in cards:
                card["provenance"] = {"source_name": candidate["title"], "source_url": candidate["url"], "grounding": "full_text_extracted_pending"}
                card["origin_links"] = list(candidate.get("origin_links", []))
            results.append({**candidate, "pdf_path": str(pdf_path), "full_text_status": "completed", "full_text_review": relevance, "full_text_similarity": similarity, "paper_summary": paper_summary, "knowledge_candidates": knowledge_candidates, "candidate_cards": cards, "evidence_status": "full_text_extracted_pending"})
        except LLMRetryExhausted:
            raise
        except Exception as error:
            results.append({**candidate, "full_text_status": "failed", "full_text_error": str(error), "candidate_cards": [], "evidence_status": "abstract_only_pending"})
    return results

def _similarity_score(review: str, candidate: dict[str, Any]) -> int:
    match = re.search(r"(?:SIMILARITY|유사도)\s*[:=]\s*(\d{1,3})", review, flags=re.IGNORECASE)
    if match:
        return min(100, int(match.group(1)))
    level = candidate.get("relevance", {}).get("level", "low")
    return {"high": 75, "medium": 50, "low": 25}.get(level, 0)


def _parse_review_assets(review: str) -> tuple[str, list[dict[str, str]]]:
    summary = ""
    claims: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in (review or "").splitlines():
        line = raw.strip()
        if line.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("CLAIM:"):
            if current and current.get("claim") and current.get("evidence"):
                claims.append(current)
            current = {"claim": line.split(":", 1)[1].strip(), "evidence": "", "limits": ""}
        elif current is not None and line.startswith("EVIDENCE:"):
            current["evidence"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("LIMITS:"):
            current["limits"] = line.split(":", 1)[1].strip()
    if current and current.get("claim") and current.get("evidence"):
        claims.append(current)
    return summary, claims[:3]


def _relevance_prompt(profile: dict[str, Any], candidate: dict[str, Any], text: str) -> str:
    return apply_review_language_policy(f"""You are M1 reviewing one selected paper for a research question. Use only the extracted paper text.
Return exactly this structure:
SIMILARITY: NN
SUMMARY: <3-6 sentence paper summary, source-intrinsic and useful to a researcher>
CLAIM: <one source-grounded claim useful for the research question>
EVIDENCE: <short exact-or-close evidence excerpt from the extracted text>
LIMITS: <conditions, uncertainty, or limits>
CLAIM: <optional second claim>
EVIDENCE: <...>
LIMITS: <...>

Rules:
- NN is an integer 0-100 for fit to the approved Intent.
- Produce at most 3 CLAIM blocks.
- Do not treat a claim as approved knowledge. These are candidates for researcher review.
- Do not invent facts outside the extracted text.

Approved Intent title: {profile['title']}
Research question: {profile['question']}
Approved Intent context: {profile['context']}
Paper: {candidate['title']}
Extracted text:
{text}""")
