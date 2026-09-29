"""Compatibility facade for research-question capabilities. New code should import concern-specific modules."""
from research_fellow.application.advising_rq_parsers import *  # noqa: F401,F403
from research_fellow.application.advising_rq_store import *  # noqa: F401,F403
from research_fellow.application.advising_rq_prompts import *  # noqa: F401,F403
from research_fellow.application.advising_rq_intents import *  # noqa: F401,F403
