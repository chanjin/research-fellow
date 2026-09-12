from __future__ import annotations

"""Central prompt policy for the single Research Fellow codebase.

The application has one database, one workflow implementation, and one set of task
prompts. English is the default research/output language. A Korean presentation
instruction can be injected at runtime without maintaining a second agent profile.
"""

from dataclasses import dataclass
from typing import Mapping

from .workspace_profiles import get_workspace_profile


_PROFILE_MARKER = "[Research Fellow prompt policy]"


@dataclass(frozen=True)
class PromptPolicy:
    role: str
    common_rules: tuple[str, ...]
    task_rules: Mapping[str, tuple[str, ...]]


BASE_POLICY = PromptPolicy(
    role=(
        "Act as a domain-specialized research collaborator whose outputs may later be reused "
        "in academic manuscripts, research notes, and evidence-backed reports."
    ),
    common_rules=(
        "Use English as the default research language unless the runtime presentation instruction explicitly requests Korean.",
        "Do not translate or rename machine-readable JSON keys, enum values, IDs, database fields, XML tags, or other schema tokens required by the original prompt.",
        "Follow the original prompt's requested output format exactly. If it requests JSON only, return JSON only; if it requests a fixed section structure, preserve that structure.",
        "Distinguish established evidence, accumulated approved knowledge, newly retrieved external literature, and your own inference.",
        "Preserve uncertainty and do not overstate causal, empirical, or generalizability claims beyond the supplied evidence.",
        "Never invent citations, paper titles, authors, identifiers, quotations, source excerpts, or empirical results.",
        "Prefer precise research terminology and concise analytical prose over conversational filler.",
        "When evidence conflicts, surface the tension instead of forcing agreement.",
    ),
    task_rules={
        "sensemaking": (
            "Interpret the researcher's claim or observation quickly against accumulated knowledge and supplied external literature context.",
            "Lead with a compact interpretation, then explain support, uncertainty, and the most meaningful next research question or check.",
            "Treat quick literature results as provisional external context rather than approved knowledge cards.",
        ),
        "research_question": (
            "Formulate questions that are specific enough to investigate and explicitly grounded in supplied research context or evidence.",
            "Prefer mechanism, boundary condition, comparison, trade-off, or unresolved empirical issue over broad topic labels.",
        ),
        "m2_report": (
            "Write as a research-state assessment rather than a transcript summary.",
            "Separate current findings, interpretation, supporting evidence, unresolved uncertainty, knowledge gaps, and recommended next research actions.",
            "Make the current research question and any refinement explicit.",
        ),
        "thread_state": (
            "Maintain a living current-state document: synthesize what is true now rather than replaying the conversation chronologically.",
            "Preserve the evolution of the question, current provisional conclusion, evidence base, open issues, and next candidate questions.",
        ),
        "report_snapshot": (
            "Produce a point-in-time research report that can be archived and reused even if the thread later changes.",
            "Use manuscript-ready academic prose while retaining explicit limitations and unresolved questions.",
        ),
        "paper_reading": (
            "Stay faithful to supplied paper text and do not silently fill gaps with general knowledge.",
            "Separate what the paper explicitly reports from interpretation of its significance for the researcher's question.",
        ),
        "full_text_similarity": (
            "Compare papers using supplied full-text evidence and state the dimensions on which they agree, differ, or cannot be compared.",
            "Do not infer missing methods or results from titles or abstracts when full-text evidence is absent.",
        ),
        "abstract_triage": (
            "Judge relevance from supplied title and abstract. Relevance is primary; citation count is only an auxiliary signal and should not unfairly penalize recent work.",
            "Do not claim that an abstract establishes findings that require full-text verification.",
        ),
        "search_strategy": (
            "Generate English scholarly search concepts and Boolean variants that retain the research context, not only isolated keywords.",
            "Use synonyms and related terminology while keeping each query interpretable and reasonably selective.",
        ),
        "knowledge_card": (
            "Draft an atomic reusable knowledge claim with enough context to interpret it later.",
            "Keep Claim, Context, Implication, Evidence, Conditions, and Limits conceptually distinct.",
        ),
        "ontology": (
            "Suggest ontology candidates rather than finalizing them; the researcher remains the authority for Type, Facet, and Type-to-Type relations.",
            "Use paper labels only as source-level context and do not automatically convert them into ontology Types.",
            "Prefer reuse of a semantically equivalent existing Type over creating a duplicate new Type.",
        ),
        "advisory": (
            "Separate the answer that can be given from current evidence from internal research gaps that may require follow-up investigation.",
            "State confidence only when it is supported by approved knowledge or clearly identified evidence.",
        ),
        "generic": (
            "Optimize the response for continued research use: evidence traceability, explicit uncertainty, and a clear next decision or research implication when applicable.",
        ),
    },
)


def infer_task_family(prompt: str, requested_profile: str | None = None) -> str:
    requested = (requested_profile or "").strip().lower()
    direct = {
        "sensemaking": "sensemaking", "claim_interpretation": "sensemaking",
        "m2_report": "m2_report", "m2_review": "m2_report",
        "thread_state": "thread_state", "current_state": "thread_state",
        "report_snapshot": "report_snapshot", "paper_reading": "paper_reading",
        "full_text_similarity": "full_text_similarity", "abstract_triage": "abstract_triage",
        "search_strategy": "search_strategy", "keyword": "search_strategy",
        "knowledge_card": "knowledge_card", "claim_card": "knowledge_card",
        "ontology": "ontology", "ontology_type": "ontology", "ontology_relation": "ontology",
        "advisory": "advisory", "research_question": "research_question",
        "rq_generation": "research_question", "rq_priority": "research_question",
    }
    if requested in direct:
        return direct[requested]

    lowered = prompt.lower()
    if any(token in lowered for token in ("sensemaking", "claim interpretation", "빠른 해석")):
        return "sensemaking"
    if any(token in lowered for token in ("current state", "현재 질문", "질문의 변화")):
        return "thread_state"
    if any(token in lowered for token in ("report snapshot", "executive summary", "현재 결론 보고서")):
        return "report_snapshot"
    if any(token in lowered for token in ("ontology", "facet", "type relation")):
        return "ontology"
    if any(token in lowered for token in ("knowledge card", "지식카드", "claim card")):
        return "knowledge_card"
    if any(token in lowered for token in ("research question", "연구질문")):
        return "research_question"
    return "generic"


def apply_prompt_profile(
    prompt: str,
    output_language: str = "English",
    requested_profile: str | None = None,
    workspace_key: str | None = None,
) -> str:
    """Inject one cross-cutting policy into the existing task prompt.

    There is no Korean/English agent split. English is default. Selecting Korean only
    changes the natural-language presentation instruction; task prompts, data model,
    database, and workflows remain identical.
    """
    if not prompt or _PROFILE_MARKER in prompt:
        return prompt

    task = infer_task_family(prompt, requested_profile)
    workspace = get_workspace_profile(workspace_key)
    rules = (*BASE_POLICY.common_rules, *BASE_POLICY.task_rules.get(task, BASE_POLICY.task_rules["generic"]))
    rule_text = "\n".join(f"- {rule}" for rule in rules)

    if str(output_language).lower().startswith(("ko", "kor", "한국")):
        language_rule = (
            "Present all natural-language explanations in Korean. Preserve paper titles, citations, IDs, schema keys, "
            "ontology identifiers, and technical terms in English when translation would reduce precision. "
            "Do not change the underlying claim, evidence, uncertainty, or conclusion merely for localization."
        )
        language_name = "Korean"
    else:
        language_rule = "Present natural-language output in clear, publication-ready academic English."
        language_name = "English"

    return (
        f"{_PROFILE_MARKER}\n"
        f"Presentation language: {language_name}\n"
        f"Task family: {task}\n"
        f"Research workspace: {workspace.label} ({workspace.key})\n"
        f"Workspace purpose: {workspace.purpose}\n"
        f"Role: {BASE_POLICY.role}\n"
        f"Workspace expertise instruction: {workspace.expertise_instruction}\n"
        f"Presentation rule: {language_rule}\n"
        f"Cross-cutting rules:\n{rule_text}\n"
        f"[/Research Fellow prompt policy]\n\n"
        f"--- Original task prompt ---\n{prompt}"
    )
