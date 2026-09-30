"""Streamlit renderer for the job-centred Research Workspace."""
from __future__ import annotations
from typing import Any, Mapping


def _short(text: str, n: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_research_workspace(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    st.subheader("Research" if english else "연구")
    st.caption(
        "Follow the job from research questions through evidence acquisition and knowledge change."
        if english else
        "연구질문에서 근거 확보와 지식 변화까지 직무의 흐름을 따라봅니다."
    )

    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(4)
    cols[0].metric("Active questions" if english else "활성 연구질문", int(counts.get("active_questions", 0)))
    cols[1].metric("Exploring" if english else "탐색 중", int(counts.get("exploring_questions", 0)))
    cols[2].metric("New knowledge" if english else "새 지식", int(counts.get("new_knowledge", 0)))
    cols[3].metric("Next actions" if english else "다음 행동", int(counts.get("next_actions", 0)))

    st.markdown("### " + ("Active Research Questions" if english else "활성 연구질문"))
    questions = list(snapshot.get("questions") or [])
    if not questions:
        st.caption("No active research questions." if english else "활성 연구질문이 없습니다.")
    for item in questions:
        status = str(item.get("status") or "candidate")
        with st.expander(f"[{status}] {item.get('question', '')}", expanded=status == "exploring"):
            if item.get("rationale"):
                st.write(_short(item.get("rationale", ""), 500))
            meta = []
            if item.get("source_type"):
                meta.append(f"source: {item['source_type']}")
            meta.append(f"intents: {int(item.get('intent_count') or 0)}")
            meta.append(f"evidence: {int(item.get('evidence_source_count') or 0)}")
            st.caption(" · ".join(meta))
            if item.get("exploration_need"):
                st.caption(("Exploration need: " if english else "탐색 필요: ") + _short(item["exploration_need"], 350))

    left, right = st.columns(2)
    with left:
        st.markdown("### " + ("Evidence Acquisition" if english else "근거 확보"))
        evidence = list(snapshot.get("evidence") or [])
        if not evidence:
            st.caption("No active evidence work." if english else "진행 중인 근거 확보 작업이 없습니다.")
        for item in evidence[:12]:
            kind = "Discovery" if item.get("kind") == "literature_discovery" else "Paper"
            st.markdown(f"**{kind} · {item.get('status', '')}** — {_short(item.get('title', ''), 120)}")
            bits = []
            if item.get("count"):
                bits.append(f"{item['count']} results")
            if item.get("detail"):
                bits.append(_short(item.get("detail", ""), 160))
            if bits:
                st.caption(" · ".join(bits))

    with right:
        st.markdown("### " + ("Knowledge Progress" if english else "지식 진척"))
        updates = list(snapshot.get("knowledge_progress") or [])
        if not updates:
            st.caption("No recent knowledge updates." if english else "최근 지식 업데이트가 없습니다.")
        for item in updates[:12]:
            marker = "NEW" if item.get("status") == "new" else "reviewed"
            st.markdown(f"**{marker}** — {_short(item.get('title', ''), 140)}")
            if item.get("detail"):
                st.caption(_short(item.get("detail", ""), 180))

    advisory = list(snapshot.get("advisory") or [])
    if advisory:
        st.markdown("### " + ("Advisory / Research Requests" if english else "자문·연구 요청"))
        for item in advisory[:10]:
            requester = f" · {item.get('requester')}" if item.get("requester") else ""
            st.markdown(f"- **{item.get('status', '')}** · {_short(item.get('question', ''), 160)}{requester}")

    st.markdown("### " + ("Next Actions" if english else "다음 행동"))
    actions = list(snapshot.get("next_actions") or [])
    if not actions:
        st.success("No research action currently requires prioritisation." if english else "현재 우선 처리할 연구 행동이 없습니다.")
    else:
        for item in actions:
            priority = str(item.get("priority") or "medium").upper()
            st.markdown(f"**{priority} · {_short(item.get('title', ''), 180)}**")
            st.caption(_short(item.get("reason", ""), 300))
