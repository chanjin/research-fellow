"""Streamlit renderer for the unified human Attention Queue.

The queue itself is read-only.  When callbacks are supplied, each projected item
is realized through its existing Interaction Contract and submitted responses are
returned to the application layer for durable handling.
"""
from __future__ import annotations
import hashlib
from typing import Any, Callable, Mapping

from research_fellow.ui.interaction import render_interaction

_CATEGORY_LABELS = {
    "decisions": ("결정", "Decisions"),
    "reviews": ("검토", "Reviews"),
    "inputs": ("입력", "Inputs"),
    "exceptions": ("예외", "Exceptions"),
}


def _stable_interaction_key(item: Mapping[str, Any]) -> str:
    """Return a stable UI key across Attention projection/source transitions.

    Attention rows can be reconstructed from a failure, a recovered run, or a
    workflow checkpoint.  Using ``attention_id`` directly makes Streamlit widget
    state disappear when the projection source changes after submission.  The
    literature round + interaction/stage identity remains stable across those
    transitions and is therefore the right UI-state key.
    """
    payload = dict(item.get("payload") or {})
    parts = [
        str(item.get("round_id") or payload.get("intent_id") or ""),
        str(item.get("subject_id") or payload.get("research_question_id") or ""),
        str(item.get("interaction_id") or ""),
        str(payload.get("stage") or payload.get("current_stage") or item.get("phase_label") or ""),
        str(payload.get("item_key") or ""),
    ]
    identity = "|".join(parts).strip("|") or str(item.get("attention_id") or "attention-item")
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def _attention_error_key(item: Mapping[str, Any]) -> str:
    return f"attention-submit-error::{_stable_interaction_key(item)}"


def _submission_error_message(error: Exception, *, english: bool) -> str:
    message = str(error).strip() or error.__class__.__name__
    if english:
        return f"The response was not applied. {message} The pasted response is preserved; edit it and submit again."
    return f"응답을 반영하지 못했습니다. {message} 붙여넣은 응답은 그대로 유지되므로 수정 후 다시 제출할 수 있습니다."


def _render_attention_item_body(
    st: Any,
    item: Mapping[str, Any],
    *,
    english: bool,
    interaction_inputs: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None,
    submit_response: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None,
    candidate_action: Callable[[Mapping[str, Any], Mapping[str, Any], str], Any] | None,
    key_suffix: str,
) -> None:
    category = str(item.get("category") or "")
    ko, en = _CATEGORY_LABELS.get(category, (category, category))
    level = str(item.get("autonomy_level") or "")
    target = str(item.get("target_autonomy_level") or level)
    suffix = f"{level} → {target}" if target != level else level
    phase = str(item.get("phase_label") or "").strip()
    summary = str(item.get("summary") or "").strip()
    badge = en if english else ko
    st.markdown(f"**[{badge}] {phase or item.get('title') or item.get('interaction_id')}**")
    st.caption(" · ".join(x for x in [summary, suffix, str(item.get("interaction_id") or "")] if x))
    if not str(item.get("interaction_id") or ""):
        st.info(
            "This item requires recovery/diagnostic handling in Runtime or Developer Tools."
            if english else
            "이 항목은 Runtime 또는 Developer Tools의 복구·진단 경로에서 처리합니다."
        )
        return
    if interaction_inputs is None or submit_response is None:
        st.caption(
            "Interaction handling is not connected in this view."
            if english else "이 화면에는 Interaction 처리 경로가 연결되어 있지 않습니다."
        )
        return
    inputs = interaction_inputs(item)
    if inputs is None:
        st.info(
            "No Interaction adapter is registered for this Attention item."
            if english else "이 Attention 항목에 연결된 Interaction adapter가 없습니다."
        )
        return
    stable_key = _stable_interaction_key(item)
    error_key = _attention_error_key(item)
    prior_error = str(st.session_state.get(error_key) or "").strip()
    if prior_error:
        st.error(prior_error)
    result = render_interaction(
        st,
        str(item.get("interaction_id") or ""),
        inputs,
        key=f"attention-{stable_key}-{key_suffix}",
        item_action=(lambda candidate, decision: candidate_action(item, candidate, decision)) if candidate_action is not None else None,
    )
    if result.submitted:
        try:
            outcome = submit_response(item, result.values)
        except Exception as error:  # submission/validation/runtime errors must remain visible in-place
            message = _submission_error_message(error, english=english)
            st.session_state[error_key] = message
            st.error(message)
            return
        st.session_state.pop(error_key, None)
        status = str(getattr(outcome, "status", "completed"))
        st.success(
            f"Interaction processed: {status}."
            if english else f"Interaction을 처리했습니다: {status}."
        )
        st.rerun()


def render_attention_queue(
    st: Any,
    snapshot: Mapping[str, Any],
    *,
    english: bool = True,
    interaction_inputs: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    submit_response: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
    candidate_action: Callable[[Mapping[str, Any], Mapping[str, Any], str], Any] | None = None,
) -> None:
    title = "Attention Needed" if english else "확인이 필요한 작업"
    st.subheader(title)
    continuation_flash = st.session_state.pop("attention-continuation-flash", "")
    if continuation_flash:
        st.info(("Agent continued after your action: " if english else "사람의 판단 이후 Agent가 다음 작업을 이어서 수행했습니다: ") + str(continuation_flash))
    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(4)
    for index, category in enumerate(("decisions", "reviews", "inputs", "exceptions")):
        ko, en = _CATEGORY_LABELS[category]
        cols[index].metric(en if english else ko, int(counts.get(category, 0)))

    items = list(snapshot.get("items") or [])
    if not items:
        st.success("No work currently requires human attention." if english else "현재 사람의 확인이 필요한 작업이 없습니다.")
        return

    def _render_item_set(view_items: list[Mapping[str, Any]], *, view_key: str) -> None:
        if not view_items:
            st.caption("No items in this category." if english else "이 범주에는 현재 작업이 없습니다.")
            return
        # Literature work is job-centric: one Curation Intent is one literature round.
        round_groups: dict[str, list[Mapping[str, Any]]] = {}
        standalone: list[Mapping[str, Any]] = []
        round_order: list[str] = []
        for item in view_items[:40]:
            round_id = str(item.get("round_id") or "").strip()
            if not round_id:
                standalone.append(item)
                continue
            if round_id not in round_groups:
                round_groups[round_id] = []
                round_order.append(round_id)
            round_groups[round_id].append(item)

        card_index = 0
        for round_id in round_order:
            group = round_groups[round_id]
            first = group[0]
            subject = str(first.get("subject_title") or first.get("title") or "Research Question")
            round_label = str(first.get("round_label") or "문헌조사")
            phases = [str(x.get("phase_label") or "").strip() for x in group if str(x.get("phase_label") or "").strip()]
            current = " / ".join(dict.fromkeys(phases))
            heading = f"{subject} · {round_label}" + (f" · {current}" if current else "")
            with st.expander(heading, expanded=card_index == 0):
                st.caption(
                    "One literature round continues through its current human boundaries (Input → Review → Input → Decision)."
                    if english else
                    "하나의 문헌조사 라운드 안에서 필요한 사람 작업이 Input → Review → Input → Decision 순으로 이어집니다."
                )
                for idx, item in enumerate(group):
                    if idx:
                        st.divider()
                    try:
                        _render_attention_item_body(
                            st, item, english=english, interaction_inputs=interaction_inputs,
                            submit_response=submit_response, candidate_action=candidate_action, key_suffix=f"{view_key}-round-{round_id}-{idx}",
                        )
                    except (ValueError, TypeError, KeyError) as error:
                        st.error(str(error))
            card_index += 1

        for index, item in enumerate(standalone):
            category = str(item.get("category") or "")
            ko, en = _CATEGORY_LABELS.get(category, (category, category))
            label = en if english else ko
            heading = f"[{label}] {item.get('title') or item.get('interaction_id')}"
            with st.expander(heading, expanded=card_index == 0 and index == 0):
                try:
                    _render_attention_item_body(
                        st, item, english=english, interaction_inputs=interaction_inputs,
                        submit_response=submit_response, candidate_action=candidate_action, key_suffix=f"{view_key}-standalone-{index}",
                    )
                except (ValueError, TypeError, KeyError) as error:
                    st.error(str(error))

    groups = dict(snapshot.get("groups") or {})
    category_order = ("inputs", "reviews", "decisions", "exceptions")
    tab_labels = []
    for category in category_order:
        ko, en = _CATEGORY_LABELS[category]
        label = en if english else ko
        tab_labels.append(f"{label} ({int(counts.get(category, 0))})")
    category_tabs = st.tabs(tab_labels)
    for tab, category in zip(category_tabs, category_order):
        with tab:
            _render_item_set(list(groups.get(category) or []), view_key=category)

