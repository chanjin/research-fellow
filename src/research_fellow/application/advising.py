"""Compatibility facade for M2 advising capabilities.

New code should import from advising_state, advising_research_questions, or advising_direction.
"""
from research_fellow.application.advising_state import *  # noqa: F401,F403
from research_fellow.application.advising_research_questions import *  # noqa: F401,F403
from research_fellow.application.advising_direction import *  # noqa: F401,F403
