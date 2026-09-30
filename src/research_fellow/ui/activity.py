"""Streamlit renderer for the operational Activity workspace."""
from __future__ import annotations
from typing import Any, Mapping


def render_activity_feed(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    st.subheader("Recent Activity" if english else "최근 활동")
    items = list(snapshot.get("items") or [])
    if not items:
        st.info("No recent activity." if english else "최근 활동이 없습니다.")
        return
    for item in items[:30]:
        actor = str(item.get("actor") or "Agent")
        action = str(item.get("action") or "")
        title = str(item.get("title") or "")
        created = str(item.get("created_at") or "")
        st.markdown(f"**{actor}** · {action}")
        st.caption(" · ".join(x for x in [created, title] if x))
