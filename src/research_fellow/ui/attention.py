"""Streamlit renderer for the unified human Attention Queue.

The queue itself is read-only.  When callbacks are supplied, each projected item
is realized through its existing Interaction Contract and submitted responses are
returned to the application layer for durable handling.
"""
from __future__ import annotations
from typing import Any, Callable, Mapping

from research_fellow.ui.interaction import render_interaction

_CATEGORY_LABELS = {
    "decisions": ("결정", "Decisions"),
    "reviews": ("검토", "Reviews"),
    "inputs": ("입력", "Inputs"),
    "exceptions": ("예외", "Exceptions"),
}


def render_attention_queue(
    st: Any,
    snapshot: Mapping[str, Any],
    *,
    english: bool = True,
    interaction_inputs: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    submit_response: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
) -> None:
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

    for index, item in enumerate(items[:20]):
        category = str(item.get("category") or "")
        ko, en = _CATEGORY_LABELS.get(category, (category, category))
        level = str(item.get("autonomy_level") or "")
        target = str(item.get("target_autonomy_level") or level)
        label = en if english else ko
        summary = str(item.get("summary") or "").strip()
        suffix = f"{level} → {target}" if target != level else level
        heading = f"[{label}] {item.get('title') or item.get('interaction_id')}"

        with st.expander(heading, expanded=index == 0):
            st.caption(" · ".join(x for x in [summary, suffix, str(item.get("interaction_id") or "")] if x))
            if category == "exceptions" or not str(item.get("interaction_id") or ""):
                st.info(
                    "This item requires recovery/diagnostic handling in Runtime or Developer Tools."
                    if english else
                    "이 항목은 Runtime 또는 Developer Tools의 복구·진단 경로에서 처리합니다."
                )
                continue
            if interaction_inputs is None or submit_response is None:
                st.caption(
                    "Interaction handling is not connected in this view."
                    if english else "이 화면에는 Interaction 처리 경로가 연결되어 있지 않습니다."
                )
                continue
            try:
                inputs = interaction_inputs(item)
                if inputs is None:
                    st.info(
                        "No Interaction adapter is registered for this Attention item."
                        if english else "이 Attention 항목에 연결된 Interaction adapter가 없습니다."
                    )
                    continue
                result = render_interaction(
                    st,
                    str(item.get("interaction_id") or ""),
                    inputs,
                    key=f"attention-{item.get('attention_id') or index}",
                )
                if result.submitted:
                    outcome = submit_response(item, result.values)
                    status = str(getattr(outcome, "status", "completed"))
                    st.success(
                        f"Interaction processed: {status}."
                        if english else f"Interaction을 처리했습니다: {status}."
                    )
                    st.rerun()
            except (ValueError, TypeError, KeyError) as error:
                st.error(str(error))
