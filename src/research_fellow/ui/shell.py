"""Job-centred operational UI shell for the Research Fellow."""
from __future__ import annotations
from typing import Any, Mapping

from research_fellow.ui.attention import render_attention_queue
from research_fellow.ui.running import render_running_work
from research_fellow.ui.activity import render_activity_feed
from research_fellow.ui.research import render_research_workspace
from research_fellow.ui.knowledge import render_knowledge_workspace


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

    attention_count = int(attention.get("total") or 0)
    running_counts = dict(running.get("counts") or {})
    recent_count = int(activity.get("total") or 0)
    cols = st.columns(3)
    cols[0].metric("Attention" if english else "확인 필요", attention_count)
    cols[1].metric("Active work" if english else "진행 중", int(running_counts.get("running", 0)) + int(running_counts.get("waiting", 0)))
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
    tabs = st.tabs(labels)
    attention_tab, running_tab, activity_tab = tabs[:3]
    with attention_tab:
        render_attention_queue(st, attention, english=english)
    with running_tab:
        render_running_work(st, running, english=english)
    with activity_tab:
        render_activity_feed(st, activity, english=english)
    next_tab = 3
    if research is not None:
        with tabs[next_tab]:
            render_research_workspace(st, research, english=english)
        next_tab += 1
    if knowledge is not None:
        with tabs[next_tab]:
            render_knowledge_workspace(st, knowledge, english=english)
