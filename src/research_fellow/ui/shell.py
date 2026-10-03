"""Job-centred operational UI shell for the Research Fellow."""
from __future__ import annotations
from typing import Any, Callable, Mapping

from research_fellow.ui.attention import render_attention_queue
from research_fellow.ui.activity import render_activity_feed
from research_fellow.ui.research import render_research_workspace
from research_fellow.ui.knowledge import render_knowledge_workspace
from research_fellow.ui.system import render_system_workspace


def render_operating_desk(
    st: Any,
    *,
    workspace_label: str,
    workspace_purpose: str,
    attention: Mapping[str, Any],
    running: Mapping[str, Any],
    activity: Mapping[str, Any],
    research: Mapping[str, Any] | None = None,
    knowledge: Mapping[str, Any] | None = None,
    system: Mapping[str, Any] | None = None,
    developer_mode: bool = False,
    research_submit_question: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    research_prepare_external_advisory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    research_submit_external_advisory: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    research_request_additional_literature: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    research_request_initial_answer: Callable[[str], Mapping[str, Any]] | None = None,
    research_request_answer_update: Callable[[str], Mapping[str, Any]] | None = None,
    attention_interaction_inputs: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    attention_submit_response: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
    attention_candidate_action: Callable[[Mapping[str, Any], Mapping[str, Any], str], Any] | None = None,
    attention_dismiss_round: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    attention_dismiss_item: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    knowledge_update_paper_metadata: Callable[[str, list[str] | str, str, str | None], Mapping[str, Any]] | None = None,
    knowledge_attach_paper_pdf: Callable[[str, str, bytes], Mapping[str, Any]] | None = None,
    knowledge_enqueue_ontology_work: Callable[[], Mapping[str, Any]] | None = None,
    knowledge_manage_ontology: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    english: bool = True,
) -> None:
    st.header("Research Fellow" if english else "연구위원")
    st.markdown(f"**{workspace_label}**")
    if workspace_purpose:
        st.caption(workspace_purpose)
    st.caption(
        "The agent works continuously; intervene only when attention is required."
        if english else
        "에이전트가 지속적으로 일하고, 사람은 필요한 순간에만 개입합니다."
    )
    runtime_flash = st.session_state.pop("persistent-runtime-flash", "")
    if runtime_flash:
        st.info(("Agent continued: " if english else "Agent가 다음 작업을 이어서 수행했습니다: ") + str(runtime_flash))
    runtime_error = st.session_state.pop("persistent-runtime-error", "")
    if runtime_error:
        st.warning(("Automatic continuation needs attention: " if english else "자동 실행 연계에 확인이 필요합니다: ") + str(runtime_error))

    attention_count = int(attention.get("total") or 0)
    running_counts = dict(running.get("counts") or {})
    active_count = int(running_counts.get("running", 0)) + int(running_counts.get("queued", 0)) + int(running_counts.get("waiting", 0))
    knowledge_counts = dict((knowledge or {}).get("counts") or {})
    research_total = int((research or {}).get("total", 0))
    cols = st.columns(4)
    cols[0].metric("Research" if english else "연구", research_total)
    cols[1].metric("Attention" if english else "확인 필요", attention_count)
    cols[2].metric("Knowledge" if english else "지식카드", int(knowledge_counts.get("cards", 0)))
    cols[3].metric("Active work" if english else "진행 중", active_count)

    view_keys: list[str] = []
    if research is not None:
        view_keys.append("research")
    view_keys.append("attention")
    if knowledge is not None:
        view_keys.append("knowledge")
    view_keys.append("activity")
    if developer_mode and system is not None:
        view_keys.append("system")
    labels = {
        "research": "Research" if english else "연구",
        "attention": "Attention" if english else "확인 필요",
        "knowledge": "Knowledge" if english else "지식",
        "activity": "Activity" if english else "활동",
        "system": "System" if english else "시스템",
    }

    # A tab cannot be selected programmatically in Streamlit.  Use a compact
    # horizontal workspace selector so Research can send the user directly to
    # the owning RQ's Attention work without losing context.
    requested_view = str(st.session_state.pop("operating-desk-navigate", "") or "").strip()
    if requested_view in view_keys:
        st.session_state["operating-desk-view"] = requested_view
    current_view = str(st.session_state.get("operating-desk-view") or "")
    if current_view not in view_keys:
        st.session_state["operating-desk-view"] = "research" if "research" in view_keys else view_keys[0]
    selected_view = st.radio(
        "Workspace" if english else "업무 화면",
        view_keys,
        format_func=lambda key: labels.get(key, key),
        horizontal=True,
        key="operating-desk-view",
        label_visibility="collapsed",
    )

    if selected_view == "research" and research is not None:
        render_research_workspace(
            st, research, english=english,
            submit_research_question=research_submit_question,
            prepare_external_advisory=research_prepare_external_advisory,
            submit_external_advisory=research_submit_external_advisory,
            request_additional_literature=research_request_additional_literature,
            request_initial_answer=research_request_initial_answer,
            request_answer_update=research_request_answer_update,
        )
    elif selected_view == "attention":
        render_attention_queue(
            st, attention, english=english,
            interaction_inputs=attention_interaction_inputs,
            submit_response=attention_submit_response,
            candidate_action=attention_candidate_action,
            dismiss_round=attention_dismiss_round,
            dismiss_item=attention_dismiss_item,
        )
    elif selected_view == "knowledge" and knowledge is not None:
        render_knowledge_workspace(
            st, knowledge, english=english,
            update_paper_metadata=knowledge_update_paper_metadata,
            attach_paper_pdf=knowledge_attach_paper_pdf,
            enqueue_ontology_work=knowledge_enqueue_ontology_work,
            manage_ontology=knowledge_manage_ontology,
        )
    elif selected_view == "activity":
        render_activity_feed(st, activity, english=english)
        st.caption(
            "This workspace is reserved for M2 work and deliverables; for now it shows recent agent activity."
            if english else
            "이 화면은 M2의 보고서·논문·자문 수행과 산출물 공간으로 확장합니다. 현재는 최근 Agent 활동을 표시합니다."
        )
    elif selected_view == "system" and developer_mode and system is not None:
        render_system_workspace(st, system, english=english)
