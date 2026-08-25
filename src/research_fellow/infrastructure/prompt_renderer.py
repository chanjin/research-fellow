from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


PROMPT_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=1)
def prompt_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context: Any) -> str:
    """Render a versioned prompt asset; missing inputs fail instead of becoming blank text."""
    return prompt_environment().get_template(template_name).render(**context).strip()
