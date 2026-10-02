"""Prompt builders for interactive literature discovery."""
from __future__ import annotations
import json
from research_fellow.infrastructure.prompt_renderer import apply_review_language_policy
from typing import Any
def discovery_search_plan_prompt(topic: str, context: str, max_results: int) -> str:
    return apply_review_language_policy(f"""You are the literature-discovery planner for a domain research fellow.
The researcher wants a QUICK exploratory search, not a full systematic review.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

TASK
1. Interpret the research topic in academic terms.
2. Produce 3-5 English arXiv Boolean search expressions that balance recall and precision.
3. Prefer conceptually distinct query variants rather than minor wording changes.
4. Do not invent paper titles, authors, or citations. This step only creates search expressions.
5. The downstream UI will inspect approximately {max_results} candidate papers.

Return ONLY valid JSON:
{{
  "scope_summary": "one short paragraph",
  "queries": [
    "all:\"...\" AND all:\"...\"",
    "..."
  ],
  "search_notes": ["...", "..."]
}}
""")


def discovery_triage_prompt(topic: str, context: str, candidates: list[dict[str, Any]], max_results: int) -> str:
    rows = []
    for index, paper in enumerate(candidates[:30], start=1):
        rows.append(
            {
                "ref": f"P{index}",
                "source_id": paper.get("source_id", ""),
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "abstract": paper.get("summary", ""),
                "url": paper.get("url", ""),
            }
        )
    return apply_review_language_policy(f"""You are helping a researcher QUICKLY understand a literature area.
Do not perform a systematic review. Rank only the retrieved papers below.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

RETRIEVED PAPERS
{json.dumps(rows, ensure_ascii=False, indent=2)}

TASK
Select up to {max(5, min(int(max_results), 20))} papers that are most useful for orientation.
For each selected paper:
- give a relevance score from 0 to 100;
- give a 1-2 sentence quick_take of what the paper appears to contribute, based only on title/abstract;
- explain why_relevant to the research topic;
- optionally state a caution if the abstract is only indirectly relevant.
Do not invent facts beyond the supplied metadata and abstract.

Return ONLY valid JSON:
{{
  "papers": [
    {{
      "source_id": "...",
      "relevance_score": 0,
      "quick_take": "...",
      "why_relevant": "...",
      "caution": "..."
    }}
  ]
}}
""")


def external_literature_discovery_prompt(topic: str, context: str, max_results: int, sources: list[str] | None = None) -> str:
    target = max(5, min(int(max_results), 20))
    source_names = ", ".join(sources or ["arXiv", "Semantic Scholar", "Crossref", "Google Scholar"])
    return apply_review_language_policy(f"""Act as a literature-discovery assistant with web access.
I am doing a QUICK exploratory literature search, not a systematic review.

RESEARCH TOPIC / QUESTION
{topic.strip()}

OPTIONAL CONTEXT
{context.strip() or '(none)'}

SEARCH SOURCES TO CONSULT
{source_names}
Use these sources as discovery aids; verify the paper itself rather than trusting a single index.

Find approximately {target} real academic papers that help me understand this topic.
Prioritize directly relevant work, then a small number of useful adjacent/foundational papers.
Verify that every paper exists. Never invent titles, authors, URLs, DOI, arXiv IDs, or publication years.
Prefer a stable source page URL (arXiv HTML/abstract page, DOI landing page, publisher page, or official paper page). The URL is for reading the source page; do NOT treat it as a PDF download URL.

For each paper give:
- title
- authors
- publication_year
- abstract_url: bibliographic/abstract/landing page (arXiv /abs, DOI landing, publisher abstract page, etc.)
- full_text_url: readable full-text HTML or official/open-access article page when available; empty if not verified. For arXiv papers, prefer the readable HTML URL (https://arxiv.org/html/<arXiv-id>) when it exists; keep the /abs page in abstract_url and the direct PDF in pdf_url.
- pdf_url: direct PDF URL only when independently verified; empty if not verified
- source_id (arXiv ID or DOI when available; otherwise a stable unique identifier or empty string)
- abstract_or_summary (brief and factual; if you do not have the abstract, label it as a summary)
- relevance_score 0-100
- quick_take: 1-2 sentences on what the paper contributes
- why_relevant: why I should read it for this research topic
- caution: any important limitation or indirectness

Return ONLY valid JSON with this schema:
If any field contains LaTeX, escape every backslash for JSON (for example, write $\\\\pi$ in the JSON source, not $\\pi$).
{{
  "search_summary": "short orientation to the literature",
  "papers": [
    {{
      "title": "...",
      "authors": ["..."],
      "publication_year": "2025",
      "abstract_url": "https://...",
      "full_text_url": "",
      "pdf_url": "",
      "source_id": "...",
      "abstract_or_summary": "...",
      "relevance_score": 90,
      "quick_take": "...",
      "why_relevant": "...",
      "caution": "..."
    }}
  ]
}}
""")
