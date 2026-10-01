from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# Researcher-review language convention.  Keep this in one place so new prompt
# assets inherit the same review-friendly output style automatically.
REVIEW_LANGUAGE_POLICY = """OUTPUT LANGUAGE POLICY (apply to the generated result):
- Write researcher-facing narrative sentences in Korean. This includes summaries, explanations, rationales, claims, evidence interpretations, limitations, conditions, answers, contexts, implications, cautions, and review notes.
- Keep short machine-oriented vocabulary in English: Label/Labels, Type, Facet, Category, Status, Relation/Relation Type, Action, Priority, Confidence, ontology names, tags, and search keywords/query expressions.
- Keep JSON keys and required enum/token values exactly as specified by the schema, normally in English.
- Keep bibliographic facts and identifiers in their original form: paper titles, author names, model/method proper names, DOI/arXiv IDs, URLs, and exact source excerpts.
- When a field is a natural-language sentence, prefer Korean even when the JSON key or field label is English. When a field is a short taxonomy/tag/type value, prefer concise English.
- Do not translate exact quotations or evidence excerpts that must remain source-grounded."""


@lru_cache(maxsize=1)
def prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def apply_review_language_policy(prompt: str) -> str:
    """Apply the global researcher-review language convention to a prompt."""
    body = str(prompt or "").strip()
    if not body:
        return REVIEW_LANGUAGE_POLICY
    if body.startswith(REVIEW_LANGUAGE_POLICY):
        return body
    return f"{REVIEW_LANGUAGE_POLICY}\n\n{body}"


def render_prompt(template_name: str, **context: Any) -> str:
    """Render a versioned prompt asset; missing inputs fail instead of becoming blank text."""
    rendered = prompt_environment().get_template(template_name).render(**context).strip()
    return apply_review_language_policy(rendered)
