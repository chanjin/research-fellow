"""Streamlit renderer for the unified human Attention Queue.

The queue itself is read-only.  When callbacks are supplied, each projected item
is realized through its existing Interaction Contract and submitted responses are
returned to the application layer for durable handling.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from research_fellow.ui.interaction import render_interaction

_CATEGORY_LABELS = {
    "decisions": ("결정", "Decisions"),
    "reviews": ("검토", "Reviews"),
    "inputs": ("입력", "Inputs"),
    "exceptions": ("예외", "Exceptions"),
}

_WORK_KIND_LABELS = {
    "research": ("연구질문", "Research"),
    "literature": ("문헌조사", "Literature"),
    "knowledge": ("지식카드", "Knowledge"),
    "ontology": ("온톨로지", "Ontology"),
    "system": ("시스템", "System"),
}


def _attention_work_kind(item: Mapping[str, Any]) -> str:
    """Classify what the Attention item is about, independently of what action is required."""
    source_type = str(item.get("source_type") or "")
    interaction_id = str(item.get("interaction_id") or "")
    payload = dict(item.get("payload") or {}) if isinstance(item.get("payload"), Mapping) else {}
    subject_type = str(payload.get("subject_type") or "")
    nested = dict(payload.get("payload") or {}) if isinstance(payload.get("payload"), Mapping) else {}
    nested_subject = str(nested.get("subject_type") or "")
    combined = " ".join([source_type, interaction_id, subject_type, nested_subject, str(item.get("title") or ""), str(item.get("phase_label") or "")]).lower()
    if "ontology" in combined or "facet" in combined or "type_suggestion" in combined:
        return "ontology"
    if "knowledge_card" in combined or "knowledge relation" in combined or "knowledge_relation" in combined:
        return "knowledge"
    if str(item.get("round_id") or "").strip() or "literature" in combined or "paper" in combined:
        return "literature"
    if "research_question" in combined or "research question" in combined or "advisory" in combined:
        return "research"
    return "system"


def _attention_work_label(item: Mapping[str, Any], *, english: bool) -> str:
    kind = _attention_work_kind(item)
    ko, en = _WORK_KIND_LABELS.get(kind, (kind, kind))
    return en if english else ko



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




def _knowledge_review_header(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if str(item.get("category") or "") != "decisions" or str(item.get("source_type") or "") != "decision_request":
        return "", "", "", ""
    request = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    if str(request.get("subject_type") or "") != "knowledge_card":
        return "", "", "", ""
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    return (
        str(payload.get("paper_id") or "").strip(),
        str(payload.get("review_paper_title") or "").strip(),
        str(payload.get("research_question") or "").strip(),
        str(payload.get("review_summary") or "").strip(),
    )



def _compact_summary_markdown(text: str) -> str:
    """Render legacy/new 1-page summaries without large Markdown headings."""
    lines = []
    for raw in str(text or "").splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("### "):
            prefix = raw[: len(raw) - len(stripped)]
            lines.append(prefix + "**" + stripped[4:].strip() + "**")
        elif stripped.startswith("## "):
            prefix = raw[: len(raw) - len(stripped)]
            lines.append(prefix + "**" + stripped[3:].strip() + "**")
        elif stripped.startswith("# "):
            prefix = raw[: len(raw) - len(stripped)]
            lines.append(prefix + "**" + stripped[2:].strip() + "**")
        else:
            lines.append(raw)
    return "\n".join(lines).strip()


def _knowledge_review_links(item: Mapping[str, Any]) -> list[tuple[str, str]]:
    request = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    links = [
        ("초록 / 서지", str(payload.get("paper_abstract_url") or "").strip()),
        ("원문", str(payload.get("paper_full_text_url") or "").strip()),
        ("PDF", str(payload.get("paper_pdf_url") or "").strip()),
    ]
    return [(label, url) for label, url in links if url.startswith(("http://", "https://"))]



def _knowledge_review_local_pdf(item: Mapping[str, Any]) -> str:
    request = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    return str(payload.get("paper_pdf_path") or "").strip()

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
    work_label = _attention_work_label(item, english=english)
    st.markdown(f"**[{badge} · {work_label}] {phase or item.get('title') or item.get('interaction_id')}**")
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
        outcome_payload = dict(getattr(outcome, "outcome", {}) or {})
        source_type = str(item.get("source_type") or "")
        source_payload = dict(item.get("payload") or {})
        ontology_subject = str(source_payload.get("subject_type") or "")
        if (
            source_type == "research_task"
            and ontology_subject in {
                "ontology_type_suggestion",
                "ontology_relation_suggestion",
                "ontology_facet_suggestion",
            }
            and str(outcome_payload.get("review_id") or "").strip()
        ):
            st.session_state["attention-focus-category"] = "reviews"
            st.session_state["attention-continuation-flash"] = (
                "Ontology proposal created. Review it first; it is not published to Knowledge until researcher approval."
                if english else
                "Ontology 제안이 생성되었습니다. 먼저 Reviews에서 검토하세요. 연구자 승인 전에는 Knowledge에 반영되지 않습니다."
            )
        elif source_type == "ontology_change_review" and str(item.get("interaction_id") or "") == "review_ontology_change":
            st.session_state["attention-focus-category"] = "decisions"
            st.session_state["attention-continuation-flash"] = (
                "Ontology review completed. Make the final publish decision in Decisions."
                if english else
                "Ontology 검토가 완료되었습니다. Decisions에서 최종 반영 여부를 결정하세요."
            )
        elif source_type == "ontology_change_review" and str(item.get("interaction_id") or "") == "resolve_ontology_change_reviews":
            st.session_state["attention-focus-category"] = "inputs"
            st.session_state["attention-continuation-flash"] = (
                "Ontology decision processed. Approved changes are now reflected in Knowledge."
                if english else
                "Ontology 최종 결정이 처리되었습니다. 승인된 변경은 Knowledge에 반영되었습니다."
            )
        elif category == "decisions":
            st.session_state["attention-focus-category"] = "decisions"
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
    dismiss_round: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    dismiss_item: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
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

    st.caption(
        "Tabs describe what you need to do now (Input / Review / Decision / Exception). The work badge on each card shows what the task is about (Research / Literature / Knowledge / Ontology)."
        if english else
        "탭은 지금 해야 할 행동(Input / Review / Decision / Exception)을 뜻하고, 각 카드의 업무 badge는 그 일이 무엇에 관한 것인지(연구질문 / 문헌조사 / 지식카드 / 온톨로지)를 뜻합니다."
    )
    kind_counts: dict[str, int] = {}
    for item in items:
        kind = _attention_work_kind(item)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    work_summary = []
    for kind in ("research", "literature", "knowledge", "ontology", "system"):
        count = int(kind_counts.get(kind, 0))
        if not count:
            continue
        ko_label, en_label = _WORK_KIND_LABELS[kind]
        work_summary.append(f"{en_label if english else ko_label} {count}")
    if work_summary:
        st.caption(("Work types: " if english else "업무 종류: ") + " · ".join(work_summary))

    available_kinds = [kind for kind in ("research", "literature", "knowledge", "ontology", "system") if kind_counts.get(kind)]
    if len(available_kinds) > 1:
        selected_kinds = st.multiselect(
            "Work type filter" if english else "업무 종류 필터",
            options=available_kinds,
            default=available_kinds,
            format_func=lambda kind: (_WORK_KIND_LABELS.get(kind, (kind, kind))[1] if english else _WORK_KIND_LABELS.get(kind, (kind, kind))[0]),
            key="attention-work-kind-filter",
        )
        selected_set = set(selected_kinds)
        items = [item for item in items if _attention_work_kind(item) in selected_set]

    def _render_item_set(view_items: list[Mapping[str, Any]], *, view_key: str) -> None:
        if not view_items:
            st.caption("No items in this category." if english else "이 범주에는 현재 작업이 없습니다.")
            return

        # Bulk cleanup is intentionally expressed as "keep selected, delete the rest".
        # Attention itself is only a projection; dismiss_item closes each durable
        # source so removed cards do not simply reappear after refresh.
        if dismiss_item is not None and len(view_items) > 1:
            option_ids = [str(item.get("attention_id") or f"attention-{idx}") for idx, item in enumerate(view_items)]
            by_id = {str(item.get("attention_id") or f"attention-{idx}"): item for idx, item in enumerate(view_items)}

            def _bulk_label(attention_id: str) -> str:
                row = by_id.get(attention_id, {})
                phase = str(row.get("phase_label") or "").strip()
                title = str(row.get("subject_title") or row.get("title") or row.get("interaction_id") or attention_id).strip()
                return f"{title} · {phase}" if phase else title

            keep_key = f"attention-bulk-keep::{view_key}"
            kept_ids = st.multiselect(
                "Keep these Attention cards" if english else "보존할 Attention 카드",
                options=option_ids,
                format_func=_bulk_label,
                key=keep_key,
                help=(
                    "Select cards to keep. The bulk action deletes every other card in this category."
                    if english else
                    "남겨둘 카드만 선택하세요. 일괄 삭제를 실행하면 이 범주의 나머지 카드가 삭제됩니다."
                ),
            )
            kept = set(str(x) for x in kept_ids)
            delete_items = [item for attention_id, item in by_id.items() if attention_id not in kept]
            delete_count = len(delete_items)
            confirm_key = f"attention-bulk-confirm::{view_key}"
            if not st.session_state.get(confirm_key):
                if st.button(
                    (f"Delete all except selected ({delete_count})" if english else f"선택 제외 {delete_count}개 일괄 삭제"),
                    key=f"attention-bulk-delete::{view_key}",
                    disabled=delete_count == 0,
                    use_container_width=False,
                ):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.warning(
                    (f"{delete_count} Attention cards will be closed at their durable sources." if english else
                     f"선택하지 않은 Attention {delete_count}개의 원본 작업 상태를 종료합니다.")
                )
                confirm_cols = st.columns(2)
                if confirm_cols[0].button(
                    "Confirm bulk delete" if english else "일괄 삭제 확인",
                    key=f"attention-bulk-confirm-delete::{view_key}",
                    type="primary",
                    use_container_width=True,
                ):
                    kept_rounds = {
                        str(item.get("round_id") or "").strip()
                        for attention_id, item in by_id.items()
                        if attention_id in kept and str(item.get("round_id") or "").strip()
                    }
                    deleted = 0
                    skipped: list[str] = []
                    for item in delete_items:
                        round_id = str(item.get("round_id") or "").strip()
                        source_type = str(item.get("source_type") or "")
                        # A workflow-interaction dismissal cancels its whole round.
                        # Never let that delete a selected card from the same round.
                        if round_id and round_id in kept_rounds and source_type == "workflow_interaction":
                            skipped.append(_bulk_label(str(item.get("attention_id") or "")) + " (selected card in same round)")
                            continue
                        try:
                            dismiss_item(item)
                            deleted += 1
                        except Exception as error:
                            skipped.append(f"{_bulk_label(str(item.get('attention_id') or ''))}: {error}")
                    st.session_state.pop(confirm_key, None)
                    if skipped:
                        st.session_state[f"attention-bulk-delete-result::{view_key}"] = {"deleted": deleted, "skipped": skipped}
                    else:
                        st.session_state[f"attention-bulk-delete-result::{view_key}"] = {"deleted": deleted, "skipped": []}
                    st.rerun()
                if confirm_cols[1].button(
                    "Cancel" if english else "취소",
                    key=f"attention-bulk-cancel::{view_key}",
                    use_container_width=True,
                ):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

            bulk_result = st.session_state.pop(f"attention-bulk-delete-result::{view_key}", None)
            if isinstance(bulk_result, Mapping):
                deleted = int(bulk_result.get("deleted") or 0)
                skipped = list(bulk_result.get("skipped") or [])
                if deleted:
                    st.success((f"Deleted {deleted} Attention cards." if english else f"Attention {deleted}개를 삭제했습니다."))
                if skipped:
                    prefix = (
                        "Some cards were kept because they could not be safely deleted:\n"
                        if english else
                        "안전하게 삭제할 수 없어 남긴 카드가 있습니다:\n"
                    )
                    st.warning(prefix + "\n".join(f"- {x}" for x in skipped))
            st.divider()

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
            work_label = _attention_work_label(first, english=english)
            heading = f"[{work_label}] {subject} · {round_label}" + (f" · {current}" if current else "")
            with st.expander(heading, expanded=card_index == 0):
                st.caption(
                    "One literature round continues through its current human boundaries (Input → Review → Input → Decision)."
                    if english else
                    "하나의 문헌조사 라운드 안에서 필요한 사람 작업이 Input → Review → Input → Decision 순으로 이어집니다."
                )
                if dismiss_round is not None:
                    delete_key = f"attention-delete-confirm::{round_id}"
                    if not st.session_state.get(delete_key):
                        if st.button(
                            "Delete unfinished Attention" if english else "진행 중 Attention 삭제",
                            key=f"attention-delete-{round_id}-{view_key}",
                            use_container_width=False,
                        ):
                            st.session_state[delete_key] = True
                            st.rerun()
                    else:
                        st.warning(
                            "This cancels the current literature round and places the research question on hold."
                            if english else
                            "현재 문헌조사 라운드를 취소하고 연구질문을 Hold 상태로 전환합니다."
                        )
                        confirm_cols = st.columns(2)
                        if confirm_cols[0].button(
                            "Confirm delete" if english else "삭제 확인",
                            key=f"attention-delete-confirm-{round_id}-{view_key}",
                            type="primary",
                            use_container_width=True,
                        ):
                            try:
                                dismiss_round(first)
                                st.session_state.pop(delete_key, None)
                                st.success("Cancelled." if english else "진행 중 Attention을 삭제했습니다.")
                                st.rerun()
                            except Exception as error:
                                st.error(str(error))
                        if confirm_cols[1].button(
                            "Keep" if english else "취소",
                            key=f"attention-delete-cancel-{round_id}-{view_key}",
                            use_container_width=True,
                        ):
                            st.session_state.pop(delete_key, None)
                            st.rerun()
                shown_review_keys: set[str] = set()
                for idx, item in enumerate(group):
                    paper_id, paper_title, rq_text, review_summary = _knowledge_review_header(item)
                    review_key = paper_id or f"item-{idx}"
                    if review_summary and review_key not in shown_review_keys:
                        if idx:
                            st.divider()
                        st.markdown("**연구질문 관점 논문 1-Page Summary**")
                        if paper_title:
                            st.caption(paper_title)
                        if rq_text:
                            st.caption(f"Research Question · {rq_text}")
                        links = _knowledge_review_links(item)
                        local_pdf_path = _knowledge_review_local_pdf(item)
                        if links or local_pdf_path:
                            st.caption("논문 원문 확인")
                            if links:
                                link_cols = st.columns(len(links))
                                for col, (label, url) in zip(link_cols, links):
                                    col.link_button(label, url, use_container_width=True)
                            if local_pdf_path:
                                path = Path(local_pdf_path).expanduser()
                                if path.is_file():
                                    try:
                                        st.download_button(
                                            "로컬 PDF 원문 열기",
                                            data=path.read_bytes(),
                                            file_name=path.name,
                                            mime="application/pdf",
                                            key=f"attention-local-pdf-{paper_id}-{idx}",
                                            use_container_width=True,
                                        )
                                    except OSError:
                                        pass
                        st.markdown(_compact_summary_markdown(review_summary))
                        shown_review_keys.add(review_key)
                        st.divider()
                    elif idx:
                        st.divider()
                    if str(item.get("category") or "") == "inputs" and dismiss_item is not None:
                        if st.button(
                            "Delete this Input" if english else "이 Input 삭제",
                            key=f"attention-delete-input-{_stable_interaction_key(item)}-{view_key}-{idx}",
                            use_container_width=False,
                        ):
                            try:
                                dismiss_item(item)
                                st.success("Input deleted." if english else "Input Attention을 삭제했습니다.")
                                st.rerun()
                            except Exception as error:
                                st.error(str(error))
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
            work_label = _attention_work_label(item, english=english)
            heading = f"[{label} · {work_label}] {item.get('title') or item.get('interaction_id')}"
            with st.expander(heading, expanded=card_index == 0 and index == 0):
                if str(item.get("category") or "") == "inputs" and dismiss_item is not None:
                    if st.button(
                        "Delete this Input" if english else "이 Input 삭제",
                        key=f"attention-delete-input-{_stable_interaction_key(item)}-{view_key}-standalone-{index}",
                        use_container_width=False,
                    ):
                        try:
                            dismiss_item(item)
                            st.success("Input deleted." if english else "Input Attention을 삭제했습니다.")
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))
                try:
                    _render_attention_item_body(
                        st, item, english=english, interaction_inputs=interaction_inputs,
                        submit_response=submit_response, candidate_action=candidate_action, key_suffix=f"{view_key}-standalone-{index}",
                    )
                except (ValueError, TypeError, KeyError) as error:
                    st.error(str(error))

    groups = {category: [] for category in ("inputs", "reviews", "decisions", "exceptions")}
    for item in items:
        category = str(item.get("category") or "")
        if category in groups:
            groups[category].append(item)
    filtered_counts = {category: len(groups.get(category) or []) for category in groups}
    category_order = ("inputs", "reviews", "decisions", "exceptions")
    focus_category = str(st.session_state.pop("attention-focus-category", "") or "").strip()
    if focus_category in category_order:
        category_order = (focus_category,) + tuple(category for category in category_order if category != focus_category)
    tab_labels = []
    for category in category_order:
        ko, en = _CATEGORY_LABELS[category]
        label = en if english else ko
        tab_labels.append(f"{label} ({int(filtered_counts.get(category, 0))})")
    category_tabs = st.tabs(tab_labels)
    for tab, category in zip(category_tabs, category_order):
        with tab:
            _render_item_set(list(groups.get(category) or []), view_key=category)

