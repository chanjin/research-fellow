"""Streamlit summary renderer for the unified human Attention Queue."""
from __future__ import annotations
from typing import Any, Mapping

_CATEGORY_LABELS = {
    "decisions": ("결정", "Decisions"),
    "reviews": ("검토", "Reviews"),
    "inputs": ("입력", "Inputs"),
    "exceptions": ("예외", "Exceptions"),
}


def render_attention_queue(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    title = "Attention Needed" if english else "확인이 필요한 작업"
    st.subheader(title)
    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(4)
    for index, category in enumerate(("decisions", "reviews", "inputs", "exceptions")):
        ko, en = _CATEGORY_LABELS[category]
        cols[index].metric(en if english else ko, int(counts.get(category, 0)))

    items = list(snapshot.get("items") or [])
    if not items:
        st.success("No work currently requires human attention." if english else "현재 사람의 확인이 필요한 작업이 없습니다.")
        return
    for item in items[:12]:
        category = str(item.get("category") or "")
        ko, en = _CATEGORY_LABELS.get(category, (category, category))
        level = str(item.get("autonomy_level") or "")
        target = str(item.get("target_autonomy_level") or level)
        label = en if english else ko
        st.markdown(f"**[{label}] {item.get('title') or item.get('interaction_id')}**")
        summary = str(item.get("summary") or "").strip()
        suffix = f"{level} → {target}" if target != level else level
        st.caption(" · ".join(x for x in [summary, suffix, str(item.get("interaction_id") or "")] if x))
