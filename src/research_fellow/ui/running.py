"""Streamlit renderer for the operational Running workspace."""
from __future__ import annotations
from typing import Any, Mapping


def render_running_work(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    st.subheader("Running Work" if english else "진행 중인 작업")
    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(4)
    cols[0].metric("Running" if english else "실행 중", int(counts.get("running", 0)))
    cols[1].metric("Queued" if english else "대기 작업", int(counts.get("queued", 0)))
    cols[2].metric("Waiting" if english else "사람 응답 대기", int(counts.get("waiting", 0)))
    cols[3].metric("Needs attention" if english else "실행 예외", int(counts.get("attention", 0)))
    items = list(snapshot.get("items") or [])
    if not items:
        st.info("No active or suspended work." if english else "현재 실행 중이거나 대기 중인 작업이 없습니다.")
        return
    for item in items:
        status = str(item.get("status") or "")
        workflow = str(item.get("workflow_id") or "workflow")
        step = str(item.get("current_step") or "")
        st.markdown(f"**{workflow}** · `{status}`")
        detail = " · ".join(x for x in [step, str(item.get("waiting_interaction") or ""), str(item.get("summary") or "")] if x)
        if detail:
            st.caption(detail)
