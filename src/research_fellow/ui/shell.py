"""Job-centred operational UI shell for the Research Fellow."""
from __future__ import annotations
from typing import Any, Callable, Mapping

from research_fellow.ui.attention import render_attention_queue
from research_fellow.ui.running import render_running_work
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
    recent_count = int(activity.get("total") or 0)
    cols = st.columns(3)
    cols[0].metric("Attention" if english else "확인 필요", attention_count)
    cols[1].metric("Active work" if english else "진행 중", int(running_counts.get("running", 0)) + int(running_counts.get("queued", 0)) + int(running_counts.get("waiting", 0)))
    cols[2].metric("Recent activity" if english else "최근 활동", recent_count)

    labels = [
        "Attention" if english else "확인 필요",
        "Running" if english else "진행 중",
        "Activity" if english else "활동",
    ]
    if research is not None:
        labels.append("Research" if english else "연구")
    if knowledge is not None:
        labels.append("Knowledge" if english else "지식")
    if developer_mode and system is not None:
        labels.append("System" if english else "시스템")
    tabs = st.tabs(labels)
    attention_tab, running_tab, activity_tab = tabs[:3]
    with attention_tab:
        render_attention_queue(
            st, attention, english=english,
            interaction_inputs=attention_interaction_inputs,
            submit_response=attention_submit_response,
            candidate_action=attention_candidate_action,
            dismiss_round=attention_dismiss_round,
            dismiss_item=attention_dismiss_item,
        )
    with running_tab:
        render_running_work(st, running, english=english)
    with activity_tab:
        render_activity_feed(st, activity, english=english)
    next_tab = 3
    if research is not None:
        with tabs[next_tab]:
            render_research_workspace(
                st, research, english=english,
                submit_research_question=research_submit_question,
                prepare_external_advisory=research_prepare_external_advisory,
                submit_external_advisory=research_submit_external_advisory,
                request_additional_literature=research_request_additional_literature,
                request_initial_answer=research_request_initial_answer,
                request_answer_update=research_request_answer_update,
            )
        next_tab += 1
    if knowledge is not None:
        with tabs[next_tab]:
            render_knowledge_workspace(
                st, knowledge, english=english,
                update_paper_metadata=knowledge_update_paper_metadata,
                attach_paper_pdf=knowledge_attach_paper_pdf,
                enqueue_ontology_work=knowledge_enqueue_ontology_work,
            )
        next_tab += 1
    if developer_mode and system is not None:
        with tabs[next_tab]:
            render_system_workspace(st, system, english=english)
