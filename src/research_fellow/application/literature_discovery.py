"""Compatibility facade for literature-discovery capabilities.

New code should import the concern-specific modules directly.
"""
from research_fellow.application.literature_discovery_formats import *  # noqa: F401,F403
from research_fellow.application.literature_discovery_sources import *  # noqa: F401,F403
from research_fellow.application.literature_discovery_prompts import *  # noqa: F401,F403
from research_fellow.application.literature_discovery_parsers import *  # noqa: F401,F403
