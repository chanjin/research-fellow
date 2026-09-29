"""Compatibility facade for ontology-curation capabilities.

New code should import the concern-specific modules directly.
"""
from research_fellow.application.ontology_curation_context import *  # noqa: F401,F403
from research_fellow.application.ontology_curation_prompts import *  # noqa: F401,F403
from research_fellow.application.ontology_curation_parsers import *  # noqa: F401,F403
