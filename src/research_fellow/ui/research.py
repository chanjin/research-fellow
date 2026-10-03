"""Streamlit renderer for the job-centred Research Workspace."""
from __future__ import annotations
from typing import Any, Callable, Mapping

from research_fellow.ui.interaction import render_interaction


def _short(text: str, n: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"




RESEARCH_INTAKE_INTERACTIONS = (
    "submit_research_question",
    "submit_external_advisory_request",
)


def _render_new_work(
    st: Any,
    *,
    english: bool,
    submit_research_question: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    prepare_external_advisory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    submit_external_advisory: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None,
) -> None:
    """Render the two and only two general-user Research work entry points."""
    if not (submit_research_question and prepare_external_advisory and submit_external_advisory):
        return

    st.markdown("### " + ("New Work" if english else "새 업무"))
    st.caption(
        "Delegate a research question, or pass an external advisory request to the Research Fellow."
        if english else
        "연구위원에게 연구질문을 맡기거나, 외부에서 들어온 자문 요청을 전달합니다."
    )

    flash = st.session_state.pop("research-intake-flash", "")
    if flash:
        st.success(str(flash))

    active_key = "research-intake-active"
    pending_key = "research-intake-external-pending"
    left, right = st.columns(2)
    if left.button(
        "+ Research Question" if english else "+ 연구질문 제시",
        use_container_width=True,
        type="primary" if st.session_state.get(active_key) == "research_question" else "secondary",
        key="research-intake-open-question",
    ):
        st.session_state[active_key] = "research_question"
        st.session_state.pop(pending_key, None)
    if right.button(
        "+ External Advisory Request" if english else "+ 외부 자문 요청",
        use_container_width=True,
        type="primary" if st.session_state.get(active_key) == "external_advisory" else "secondary",
        key="research-intake-open-advisory",
    ):
        st.session_state[active_key] = "external_advisory"

    active = str(st.session_state.get(active_key) or "")
    if not active:
        return

    if st.button("Close" if english else "닫기", key="research-intake-close"):
        st.session_state.pop(active_key, None)
        st.session_state.pop(pending_key, None)
        st.rerun()

    if active == "research_question":
        result = render_interaction(
            st, "submit_research_question", {}, key="research-work-intake-question"
        )
        if result.submitted:
            try:
                created = submit_research_question(result.values["research_question_input"])
                st.session_state.pop(active_key, None)
                progression = dict(created.get("progression") or {})
                next_action = str(progression.get("next_action") or "")
                execution = dict(progression.get("execution") or {})
                execution_summary = str(execution.get("summary") or "").strip()
                if execution_summary:
                    st.session_state["research-intake-flash"] = (
                        f"Research question accepted. A fresh first literature round was started from the question/context only. {execution_summary}"
                        if english else
                        f"연구질문을 접수했습니다. 기존 지식·문헌을 참조하지 않고 질문과 연구자 맥락만으로 첫 문헌 탐색을 시작했습니다. {execution_summary}"
                    )
                else:
                    st.session_state["research-intake-flash"] = (
                        f"Research question accepted. A fresh first literature round was started from the question/context only; next: {next_action or 'evidence acquisition'}."
                        if english else
                        f"연구질문을 접수했습니다. 기존 지식·문헌 없이 첫 문헌 탐색을 시작했고, 다음 단계는 {next_action or '근거 확보'}입니다."
                    )
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        return

    pending = st.session_state.get(pending_key)
    if not isinstance(pending, Mapping):
        result = render_interaction(
            st, "submit_external_advisory_request", {}, key="research-work-intake-advisory"
        )
        if result.submitted:
            try:
                prepared = dict(prepare_external_advisory(result.values["external_advisory_request"]))
                st.session_state[pending_key] = prepared
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        return

    review = render_interaction(
        st,
        "review_external_advisory_interpretation",
        {
            "external_advisory_request": dict(pending.get("external_advisory_request") or {}),
            "proposed_interpretation": dict(pending.get("proposed_interpretation") or {}),
        },
        key="research-work-intake-advisory-review",
    )
    if review.submitted:
        try:
            created = submit_external_advisory(
                dict(pending.get("external_advisory_request") or {}),
                review.values["reviewed_advisory_interpretation"],
            )
            st.session_state.pop(active_key, None)
            st.session_state.pop(pending_key, None)
            progression = dict(created.get("progression") or {})
            next_action = str(progression.get("next_action") or "")
            execution = dict(progression.get("execution") or {})
            execution_summary = str(execution.get("summary") or "").strip()
            if execution_summary:
                st.session_state["research-intake-flash"] = (
                    f"Advisory request accepted as a research question. A fresh first literature round was started from the interpreted question/context only. {execution_summary}"
                    if english else
                    f"자문 요청을 연구질문으로 접수했습니다. 기존 지식·문헌을 참조하지 않고 해석된 질문과 맥락만으로 첫 문헌 탐색을 시작했습니다. {execution_summary}"
                )
            else:
                st.session_state["research-intake-flash"] = (
                    f"Advisory request accepted as a research question. A fresh first literature round was started from the interpreted question/context only; next: {next_action or 'evidence acquisition'}."
                    if english else
                    f"자문 요청을 연구질문으로 접수했습니다. 기존 지식·문헌 없이 첫 문헌 탐색을 시작했고, 다음 단계는 {next_action or '근거 확보'}입니다."
                )
            st.rerun()
        except ValueError as error:
            st.error(str(error))


def render_research_workspace(
    st: Any,
    snapshot: Mapping[str, Any],
    *,
    submit_research_question: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    prepare_external_advisory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    submit_external_advisory: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    request_additional_literature: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    request_initial_answer: Callable[[str], Mapping[str, Any]] | None = None,
    request_answer_update: Callable[[str], Mapping[str, Any]] | None = None,
    english: bool = True,
) -> None:
    st.subheader("Research" if english else "연구")
    st.caption(
        "Follow the job from research questions through evidence acquisition and knowledge change."
        if english else
        "연구질문에서 근거 확보와 지식 변화까지 직무의 흐름을 따라봅니다."
    )

    action_flash = st.session_state.pop("research-action-flash", "")
    if action_flash:
        st.success(str(action_flash))

    _render_new_work(
        st,
        english=english,
        submit_research_question=submit_research_question,
        prepare_external_advisory=prepare_external_advisory,
        submit_external_advisory=submit_external_advisory,
    )

    previous_work = list(snapshot.get("previous_work") or [])
    if previous_work:
        st.markdown("### " + ("Previous Research Work" if english else "이전 연구 작업"))
        st.caption(
            "Research work preserved from earlier workspace versions. It is shown as a read-only compatibility view and is not duplicated into new research-question state."
            if english else
            "이전 버전에서 보존된 연구 작업입니다. 새 연구질문 상태로 복제하지 않고 기존 데이터를 읽기 전용으로 보여줍니다."
        )
        for item in previous_work[:12]:
            status = str(item.get("status") or "completed")
            title = _short(item.get("title", ""), 150)
            with st.expander(f"[{status}] {title}", expanded=False):
                if item.get("research_context"):
                    st.write(_short(item.get("research_context", ""), 500))
                st.caption(
                    (f"Preserved results: {int(item.get('result_count') or 0)}" if english else f"보존된 탐색 결과: {int(item.get('result_count') or 0)}편")
                    + (f" · {item.get('updated_at')}" if item.get("updated_at") else "")
                )

    counts = dict(snapshot.get("counts") or {})
    questions = list(snapshot.get("questions") or [])
    needs_attention = sum(1 for item in questions if bool(item.get("human_attention_required")))

    phase_counts = {}
    for item in questions:
        phase = str(item.get("lifecycle_phase") or "starting")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    cols = st.columns(6)
    cols[0].metric("All" if english else "전체 질문", len(questions))
    cols[1].metric("Starting" if english else "시작 전", phase_counts.get("starting", 0))
    cols[2].metric("First literature" if english else "최초 문헌조사", phase_counts.get("initial_literature", 0))
    cols[3].metric("Follow-up" if english else "추가 문헌조사", phase_counts.get("followup_literature", 0))
    cols[4].metric("Completed" if english else "완료", phase_counts.get("completed", 0))
    cols[5].metric("Needs attention" if english else "사람 작업 필요", needs_attention)
    st.caption(
        "Research shows the full lifecycle of every research question, including completed work."
        if english else
        "Research는 현재 탐색 중인 질문뿐 아니라 시작 전·추가 문헌조사·완료된 질문까지 전체 생애주기를 보여줍니다."
    )

    st.markdown("### " + ("Research Questions" if english else "연구질문"))
    if not questions:
        st.caption("No research questions yet." if english else "아직 연구질문이 없습니다.")

    phase_labels_ko = {
        "starting": "시작 전",
        "initial_literature": "최초 문헌조사",
        "followup_literature": "추가 문헌조사",
        "completed": "완료",
        "hold": "보류",
        "closed": "종료/제외",
    }
    phase_labels_en = {
        "starting": "Starting",
        "initial_literature": "First literature",
        "followup_literature": "Follow-up literature",
        "completed": "Completed",
        "hold": "On hold",
        "closed": "Closed",
    }
    phase_labels = phase_labels_en if english else phase_labels_ko
    filter_options = ["all", "starting", "initial_literature", "followup_literature", "completed", "hold", "closed"]
    selected_phase = st.selectbox(
        "Lifecycle" if english else "연구 단계",
        filter_options,
        index=0,
        format_func=lambda value: ("All" if english else "전체") if value == "all" else phase_labels.get(value, value),
        key="research-lifecycle-filter",
    )
    visible_questions = questions if selected_phase == "all" else [
        item for item in questions if str(item.get("lifecycle_phase") or "starting") == selected_phase
    ]
    status_labels_ko = {"exploring": "탐색 중", "interested": "관심", "candidate": "후보", "hold": "보류", "resolved": "완료", "rejected": "종료/제외"}
    status_labels_en = {"exploring": "Exploring", "interested": "Interested", "candidate": "Candidate", "hold": "On hold", "resolved": "Completed", "rejected": "Closed"}
    for item in visible_questions:
        status = str(item.get("status") or "candidate")
        stage = str(item.get("current_stage") or "")
        human_needed = bool(item.get("human_attention_required"))
        status_label = (status_labels_en if english else status_labels_ko).get(status, status)
        lifecycle_phase = str(item.get("lifecycle_phase") or "starting")
        lifecycle_label = phase_labels.get(lifecycle_phase, lifecycle_phase)
        attention_marker = " · Attention" if human_needed else ""
        stage_marker = f" · {stage}" if stage else ""
        with st.expander(f"[{lifecycle_label} · {status_label}{attention_marker}{stage_marker}] {item.get('question', '')}", expanded=lifecycle_phase in {"starting", "initial_literature", "followup_literature"} or human_needed):
            if item.get("rationale"):
                st.write(_short(item.get("rationale", ""), 500))
            if item.get("researcher_comment"):
                st.markdown("**Researcher comment**" if english else "**연구자 코멘트**")
                st.write(_short(item.get("researcher_comment", ""), 700))
            meta = []
            if item.get("source_type"):
                meta.append(f"source: {item['source_type']}")
            meta.append(f"intents: {int(item.get('intent_count') or 0)}")
            meta.append(f"evidence: {int(item.get('evidence_source_count') or 0)}")
            st.caption(" · ".join(meta))
            task_cols = st.columns(3)
            task_cols[0].metric("Attention" if english else "Attention", int(item.get("attention_item_count") or 0))
            task_cols[1].metric("Paper review pending" if english else "논문 리뷰 대기", int(item.get("paper_review_pending") or 0))
            task_cols[2].metric("Paper review done" if english else "논문 리뷰 완료", int(item.get("paper_review_completed") or 0))
            next_action = str(item.get("next_agent_action") or "")
            if stage:
                st.markdown(
                    f"**Current stage:** `{stage}`" if english else f"**현재 단계:** `{stage}`"
                )
            if next_action:
                st.write(("**Next agent action:** " if english else "**다음 Agent 행동:** ") + next_action)
            execution_status = str(item.get("execution_status") or "")
            execution_detail = str(item.get("execution_detail") or "").strip()
            if human_needed:
                attention_category = str(item.get("attention_category") or "")
                attention_phase = str(item.get("attention_phase") or "").strip()
                attention_round = str(item.get("attention_round_label") or "").strip()
                category_label = {"inputs": "Inputs", "reviews": "Reviews", "decisions": "Decisions", "exceptions": "Exceptions"}.get(attention_category, "Attention")
                detail = " · ".join(x for x in [attention_round, attention_phase] if x)
                st.warning(
                    ((f"Researcher attention is required: Attention > {category_label}" + (f" · {detail}" if detail else "")) if english else
                     (f"연구자 작업이 필요합니다: Attention > {category_label}" + (f" · {detail}" if detail else "")))
                )
            elif execution_status == "attention_projection_missing":
                st.error(execution_detail or ("M1 is waiting for human input, but no Attention item was projected." if english else "M1은 사람 입력을 기다리지만 Attention 카드가 생성되지 않았습니다."))
            elif execution_status == "dispatch_pending":
                st.info(execution_detail or ("M1 dispatch is pending." if english else "M1 실행 시작을 기다리고 있습니다."))
            elif execution_status == "agent_running":
                st.info(execution_detail or ("M1 is running." if english else "M1이 실행 중입니다."))
            elif execution_detail:
                st.caption(execution_detail)
            else:
                st.caption(
                    "No researcher action is required right now." if english else "현재 연구자 개입은 필요하지 않습니다."
                )
            if item.get("progress_detail"):
                st.caption(_short(item.get("progress_detail", ""), 350))
            st.markdown("**Research flow**" if english else "**연구 진행 흐름**")
            flow = [
                ("Literature exploration" if english else "문헌 탐구", bool(item.get("intent_count"))),
                ("Paper / knowledge review" if english else "논문·지식 검토", bool(item.get("literature_round_completed"))),
                ("Knowledge registered" if english else "지식 등록", int(item.get("approved_knowledge_count") or 0) > 0),
                ("Answer draft" if english else "답변 초안", bool(item.get("answer_draft"))),
            ]
            st.caption("  →  ".join(("✓ " if done else "○ ") + label for label, done in flow))
            st.markdown("**Evidence for this question**" if english else "**이 연구질문의 근거**")
            direct_cols = st.columns(2)
            direct_cols[0].metric("Direct papers" if english else "직접 탐색 논문", int(item.get("direct_evidence_papers") or 0))
            direct_cols[1].metric("Direct knowledge" if english else "직접 생성 지식", int(item.get("direct_evidence_cards") or 0))
            direct_titles = [x for x in item.get("direct_evidence_titles") or [] if x]
            if direct_titles:
                st.caption(("Primary lineage: " if english else "1차 근거 계보: ") + " · ".join(_short(x, 80) for x in direct_titles[:5]))
            st.markdown("**Supporting prior research assets**" if english else "**기존 연구자산의 보강 근거**")
            support_cols = st.columns(2)
            support_cols[0].metric("Existing knowledge" if english else "기존 지식카드", int(item.get("supporting_knowledge_cards") or 0))
            support_cols[1].metric("Existing papers" if english else "기존 논문", int(item.get("supporting_papers") or 0))
            support_titles = [x for x in item.get("supporting_evidence_titles") or [] if x]
            if support_titles:
                st.caption(("Supporting: " if english else "보강: ") + " · ".join(_short(x, 80) for x in support_titles[:5]))
            if int(item.get("pending_knowledge_reviews") or 0):
                st.info((f"{int(item.get('pending_knowledge_reviews') or 0)} knowledge-card candidate(s) await researcher review." if english else f"논문 기반 지식카드 후보 {int(item.get('pending_knowledge_reviews') or 0)}건이 연구자 검토를 기다립니다."))
            if item.get("answer_confirmation_pending"):
                st.info("Knowledge has been updated. Confirm draft generation in Attention." if english else "현재 연구질문에 대한 지식이 업데이트되었습니다. Attention에서 답변 초안 작성 여부를 확인하세요.")

            action_left, action_mid, action_right = st.columns(3)
            rq_id = str(item.get("rq_id") or "")
            followup_open_key = f"rq-followup-direction-open-{rq_id}"
            if request_additional_literature is not None:
                if action_left.button(
                    "Additional literature" if english else "추가 문헌 조사",
                    key=f"rq-additional-literature-{rq_id}",
                    use_container_width=True,
                ):
                    st.session_state[followup_open_key] = True
                    st.rerun()
            if request_initial_answer is not None:
                initial_ready = bool(
                    int(item.get("approved_knowledge_count") or 0) > 0
                    and not item.get("answer_draft")
                    and int(item.get("pending_knowledge_reviews") or 0) == 0
                )
                if action_mid.button(
                    "Draft answer" if english else "답변 초안 작성",
                    key=f"rq-answer-draft-{rq_id}",
                    use_container_width=True,
                    disabled=not initial_ready,
                    help=(
                        "Start from accumulated approved knowledge. Attention is created only if an external LLM step is needed." if english else
                        "누적 승인 지식을 바탕으로 초안 작성을 시작합니다. 외부 LLM 단계가 필요한 경우에만 이후 Attention이 생성됩니다."
                    ),
                ):
                    try:
                        request_initial_answer(rq_id)
                        st.session_state["research-action-flash"] = (
                            "Answer drafting started." if english else
                            "현재까지 축적된 근거를 바탕으로 답변 초안 작성을 시작했습니다."
                        )
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
            if request_answer_update is not None:
                update_ready = bool(item.get("answer_update_available"))
                if action_right.button(
                    "Update answer" if english else "답변 업데이트",
                    key=f"rq-answer-update-{item.get('rq_id')}",
                    use_container_width=True,
                    disabled=not update_ready,
                    help=(
                        "Enabled when literature or knowledge changed after the latest answer." if english else
                        "최신 답변 이후 문헌 조사 또는 지식카드 변화가 있을 때 활성화됩니다."
                    ),
                ):
                    try:
                        request_answer_update(str(item.get("rq_id") or ""))
                        st.session_state["research-action-flash"] = (
                            "Answer update started." if english else
                            "변경된 근거를 반영한 답변 업데이트를 시작했습니다."
                        )
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

            if request_additional_literature is not None and st.session_state.get(followup_open_key):
                st.divider()
                st.caption(
                    "Specify what this follow-up round should investigate beyond the evidence already collected for this question."
                    if english else
                    "이번 추가 문헌 라운드에서 기존 근거를 넘어 무엇을 더 확인할지 방향을 지정하세요."
                )
                current_titles = [str(x) for x in item.get("direct_evidence_titles") or [] if str(x).strip()]
                if current_titles:
                    st.caption(("Already collected for this RQ: " if english else "현재 RQ에서 이미 확보한 근거: ") + " · ".join(_short(x, 75) for x in current_titles[:6]))
                if item.get("answer_draft"):
                    with st.expander("Current answer context" if english else "현재 연구결과 초안", expanded=False):
                        st.write(str(item.get("answer_draft") or ""))
                st.info(
                    "Submitting this form is the researcher approval boundary. Only after approval is a new M1 literature round created; external-LLM Attention comes after that."
                    if english else
                    "이 입력의 제출이 연구자 승인 경계입니다. 승인 후에만 새 M1 문헌조사 라운드가 생성되고, 필요한 외부 LLM Attention도 그 이후에 만들어집니다."
                )
                followup_result = render_interaction(
                    st,
                    "request_followup_literature_direction",
                    {},
                    key=f"rq-followup-literature-direction-{rq_id}",
                )
                cancel_col, _ = st.columns([1, 3])
                if cancel_col.button(
                    "Cancel" if english else "취소",
                    key=f"rq-followup-literature-cancel-{rq_id}",
                    use_container_width=True,
                ):
                    st.session_state.pop(followup_open_key, None)
                    st.rerun()
                if followup_result.submitted:
                    try:
                        outcome = dict(request_additional_literature(
                            rq_id,
                            dict(followup_result.values.get("followup_literature_direction") or {}),
                        ))
                        st.session_state.pop(followup_open_key, None)
                        direction = str(outcome.get("direction") or "").strip()
                        st.session_state["research-action-flash"] = (
                            f"Additional literature exploration started: {direction}" if english else
                            f"추가 문헌 조사 방향을 반영해 새 M1 탐색 라운드를 시작했습니다: {direction}"
                        )
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

            if item.get("answer_draft"):
                version = int(item.get("answer_version") or 1)
                st.markdown((f"**Research answer v{version}**" if english else f"**연구질문 답변 v{version}**"))
                if item.get("answer_first_document"):
                    with st.expander("1st document · direct RQ evidence" if english else "1차 문서 · 이번 연구질문의 직접 근거", expanded=False):
                        st.write(str(item.get("answer_first_document") or ""))
                st.markdown("**2nd / final document**" if english else "**2차 · 최종 문서**")
                st.write(str(item.get("answer_draft") or ""))
                if item.get("answer_update_available"):
                    st.info(
                        "New literature or knowledge is available after this answer. You can update it." if english else
                        "이 답변 이후 새 문헌 또는 지식 변화가 있습니다. 답변 업데이트가 가능합니다."
                    )
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
