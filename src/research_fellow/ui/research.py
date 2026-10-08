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
    prepare_from_local_papers: Callable[[Mapping[str, Any], list[tuple[str, bytes]]], Mapping[str, Any]] | None,
    start_from_local_papers: Callable[[Mapping[str, Any], list[tuple[str, bytes]], str], Mapping[str, Any]] | None,
    extract_local_paper_metadata: Callable[[str, bytes], Mapping[str, Any]] | None,
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
    left, middle, right = st.columns(3)
    if left.button(
        "+ Research Question" if english else "+ 연구질문 제시",
        use_container_width=True,
        type="primary" if st.session_state.get(active_key) == "research_question" else "secondary",
        key="research-intake-open-question",
    ):
        st.session_state[active_key] = "research_question"
        st.session_state.pop(pending_key, None)
    if middle.button(
        "+ Start from Local Paper" if english else "+ 로컬 논문에서 시작",
        use_container_width=True,
        type="primary" if st.session_state.get(active_key) == "local_paper" else "secondary",
        key="research-intake-open-local-paper",
    ):
        st.session_state[active_key] = "local_paper"
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

    if active == "local_paper":
        st.caption(
            "Upload one or more seed papers. Their full text is used to refine the question/context before the first related-literature search."
            if english else
            "하나 이상의 seed 논문 PDF를 올립니다. 원문을 바탕으로 연구질문과 맥락을 보강한 뒤 관련 문헌 탐색을 시작합니다."
        )
        uploaded_files = st.file_uploader(
            "Seed paper PDFs" if english else "Seed 논문 PDF",
            type=["pdf"], accept_multiple_files=True, key="research-intake-local-paper-files"
        )
        question = st.text_area(
            "Initial Research Question" if english else "초기 연구질문",
            height=90, key="research-intake-local-paper-question"
        )
        context = st.text_area(
            "Research Context" if english else "연구 맥락",
            height=90, key="research-intake-local-paper-context"
        )
        files = [(str(getattr(f, "name", "paper.pdf")), bytes(f.getvalue())) for f in (uploaded_files or [])]
        if files and extract_local_paper_metadata is not None:
            with st.expander((f"Seed papers ({len(files)})" if english else f"Seed 논문 ({len(files)})"), expanded=False):
                for filename, payload in files:
                    try:
                        meta = dict(extract_local_paper_metadata(filename, payload))
                        label = str(meta.get("title") or filename)
                        details = " · ".join(x for x in [str(meta.get("publication_year") or ""), ", ".join(meta.get("authors") or [])] if x)
                        st.markdown(f"**{label}**" + (f"  \n{details}" if details else ""))
                    except ValueError as error:
                        st.warning(f"{filename}: {error}")

        pending_local_key = "research-intake-local-papers-pending"
        pending_local = st.session_state.get(pending_local_key)
        if not isinstance(pending_local, Mapping):
            if st.button(
                "Prepare RQ Refinement Prompt" if english else "연구질문 보강 프롬프트 생성",
                type="primary", use_container_width=True, key="research-intake-local-paper-prepare",
                disabled=(not files or not str(question or "").strip() or prepare_from_local_papers is None),
            ):
                try:
                    prepared = dict(prepare_from_local_papers({"question": question, "context": context}, files))
                    prepared["files"] = files
                    st.session_state[pending_local_key] = prepared
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
            return

        st.markdown("#### " + ("Refine Question with Seed Papers" if english else "Seed 논문으로 연구질문 보강"))
        st.caption(
            "Copy the prompt to your LLM, then paste the JSON response below. The confirmed question/context will become the Research input and start the first related-literature round."
            if english else
            "프롬프트를 외부 LLM에 전달하고 JSON 응답을 붙여넣으세요. 확정된 질문/맥락이 Research Input이 되고 첫 관련 문헌탐색을 시작합니다."
        )
        with st.expander("LLM Prompt" if english else "LLM 프롬프트", expanded=False):
            st.code(str(pending_local.get("prompt") or ""), language=None)
        response = st.text_area(
            "LLM JSON response" if english else "LLM JSON 응답", height=260,
            key="research-intake-local-paper-refinement-response",
        )
        left_action, right_action = st.columns([2, 1])
        if left_action.button(
            "Confirm RQ · Start Literature Search" if english else "연구질문 확정 · 문헌탐색 시작",
            type="primary", use_container_width=True, key="research-intake-local-paper-finalize",
            disabled=(not str(response or "").strip() or start_from_local_papers is None),
        ):
            try:
                created = start_from_local_papers(
                    {"question": pending_local.get("initial_question", question), "context": pending_local.get("initial_context", context)},
                    list(pending_local.get("files") or files), response,
                )
                st.session_state.pop(pending_local_key, None)
                st.session_state.pop(active_key, None)
                refined = dict(created.get("refined") or {})
                st.session_state["research-intake-flash"] = (
                    f"Research question refined from {len(created.get('seed_papers') or [])} seed papers and the first related-literature search was started."
                    if english else
                    f"Seed 논문 {len(created.get('seed_papers') or [])}편으로 연구질문/맥락을 보강하고 첫 관련 문헌탐색을 시작했습니다. {refined.get('refinement_note') or ''}"
                )
                st.rerun()
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
        if right_action.button("Back" if english else "이전", use_container_width=True, key="research-intake-local-paper-back"):
            st.session_state.pop(pending_local_key, None)
            st.rerun()
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
    prepare_from_local_papers: Callable[[Mapping[str, Any], list[tuple[str, bytes]]], Mapping[str, Any]] | None = None,
    start_from_local_papers: Callable[[Mapping[str, Any], list[tuple[str, bytes]], str], Mapping[str, Any]] | None = None,
    extract_local_paper_metadata: Callable[[str, bytes], Mapping[str, Any]] | None = None,
    add_local_paper_to_rq: Callable[[str, Mapping[str, Any], str, bytes], Mapping[str, Any]] | None = None,
    submit_local_paper_review: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
    prepare_external_advisory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    submit_external_advisory: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
    request_additional_literature: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    request_initial_answer: Callable[[str], Mapping[str, Any]] | None = None,
    request_answer_update: Callable[[str], Mapping[str, Any]] | None = None,
    review_answer_draft: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
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
        prepare_from_local_papers=prepare_from_local_papers,
        start_from_local_papers=start_from_local_papers,
        extract_local_paper_metadata=extract_local_paper_metadata,
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
    focused_rq = str(st.session_state.get("research-focus-rq") or "").strip()
    if focused_rq and any(str(item.get("rq_id") or "") == focused_rq for item in questions):
        focused = [item for item in questions if str(item.get("rq_id") or "") == focused_rq]
        focus_cols = st.columns([4, 1])
        focus_cols[0].info(
            ("Focused research question" if english else "선택한 연구질문") + " · " + str(focused[0].get("question") or focused_rq)
        )
        if focus_cols[1].button(
            "Show all" if english else "전체 보기",
            key="research-clear-focus-rq",
            use_container_width=True,
        ):
            st.session_state.pop("research-focus-rq", None)
            st.rerun()
        visible_questions = focused
    status_labels_ko = {"exploring": "탐색 중", "interested": "관심", "candidate": "후보", "hold": "보류", "resolved": "완료", "rejected": "종료/제외"}
    status_labels_en = {"exploring": "Exploring", "interested": "Interested", "candidate": "Candidate", "hold": "On hold", "resolved": "Completed", "rejected": "Closed"}
    for item in visible_questions:
        status = str(item.get("status") or "candidate")
        stage = str(item.get("current_stage") or "")
        human_needed = bool(item.get("human_attention_required"))
        status_label = (status_labels_en if english else status_labels_ko).get(status, status)
        lifecycle_phase = str(item.get("lifecycle_phase") or "starting")
        lifecycle_label = phase_labels.get(lifecycle_phase, lifecycle_phase)
        rq_id = str(item.get("rq_id") or "")

        # The default RQ surface is deliberately compact.  Detailed execution,
        # evidence, and lineage information remains available on demand below.
        answer_work_status = str(item.get("answer_work_status") or "").strip()
        if item.get("answer_work_in_progress") and answer_work_status:
            state_summary = answer_work_status
        elif human_needed:
            phase = str(item.get("attention_phase") or "").strip()
            state_summary = (
                (f"Researcher action needed · {phase}" if phase else "Researcher action needed")
                if english else
                (f"사람 작업 대기 · {phase}" if phase else "사람 작업 대기")
            )
        elif int(item.get("pending_knowledge_reviews") or 0):
            state_summary = "Knowledge Card review pending" if english else "지식카드 검토 대기"
        elif int(item.get("paper_review_pending") or 0):
            state_summary = "Paper review in progress" if english else "논문 리뷰 진행 중"
        elif item.get("answer_update_available"):
            state_summary = "Answer update available" if english else "새 지식 반영 · 답변 업데이트 가능"
        elif item.get("answer_draft"):
            version = int(item.get("answer_version") or 1)
            state_summary = f"Answer v{version} ready" if english else f"답변 v{version} 준비됨"
        elif int(item.get("approved_knowledge_count") or 0) > 0:
            state_summary = "Knowledge ready · answer can be drafted" if english else "지식 업데이트 완료 · 답변 작성 가능"
        elif int(item.get("intent_count") or 0) > 0:
            state_summary = "Literature work in progress" if english else "문헌 탐색 진행 중"
        else:
            state_summary = lifecycle_label

        rounds = int(item.get("intent_count") or 0)
        direct_papers = int(item.get("direct_evidence_papers") or 0)
        review_done = int(item.get("paper_review_completed") or 0)
        review_pending = int(item.get("paper_review_pending") or 0)
        review_total = review_done + review_pending
        knowledge_count = int(item.get("approved_knowledge_count") or 0)
        progress_bits = [
            (f"Literature {rounds}R" if english else f"문헌 {rounds}R"),
            (f"Papers {direct_papers}" if english else f"논문 {direct_papers}"),
            (f"Reviewed {review_done}/{review_total}" if english else f"리뷰 {review_done}/{review_total}"),
            (f"KC {knowledge_count}"),
        ]
        if item.get("answer_work_in_progress"):
            progress_bits.append((f"Answer {item.get('answer_work_stage') or 'running'}" if english else f"답변 {item.get('answer_work_stage') or '진행 중'}"))
        elif item.get("answer_draft"):
            progress_bits.append(f"Answer v{int(item.get('answer_version') or 1)}" if english else f"답변 v{int(item.get('answer_version') or 1)}")
        else:
            progress_bits.append("Answer -" if english else "답변 -")

        with st.container(border=True):
            rq_version = int(item.get("question_version") or 1)
            title_suffix = f" · v{rq_version}" if rq_version > 1 else ""
            st.markdown(f"#### {item.get('question', '')}{title_suffix}")
            st.markdown(("**Current:** " if english else "**현재 상태:** ") + state_summary)
            st.caption(" · ".join(progress_bits))
            next_action_visible = str(item.get("next_agent_action") or "").strip()
            if next_action_visible:
                st.info(("Next: " if english else "다음 할 일: ") + next_action_visible)

            # A revised question is already the current research question. Do not
            # repeat it as another next-round Input card. Keep the version history
            # available only as a reference.
            question_versions = [dict(x) for x in (item.get("question_versions") or [])]
            if rq_version <= 1:
                st.markdown("**Inputs**")
                with st.container(border=True):
                    st.caption("Research Question" if english else "연구질문")
                    st.write(str(item.get("question") or ""))
            if len(question_versions) > 1:
                with st.expander("Research-question history" if english else "이전 연구질문 보기", expanded=False):
                    for version_item in reversed(question_versions[:-1]):
                        version_no = int(version_item.get("version_no") or 1)
                        st.markdown(f"**v{version_no}** · {str(version_item.get('question') or '')}")
                        reason = str(version_item.get("change_reason") or "").strip()
                        if reason:
                            st.caption(reason)

            attention_count = int(item.get("attention_item_count") or 0)
            if attention_count:
                category_bits = []
                for key, ko, en in (
                    ("attention_inputs", "Input", "Input"),
                    ("attention_reviews", "Review", "Review"),
                    ("attention_decisions", "Decision", "Decision"),
                    ("attention_exceptions", "Exception", "Exception"),
                ):
                    count = int(item.get(key) or 0)
                    if count:
                        category_bits.append(f"{en if english else ko} {count}")
                attention_cols = st.columns([3, 1])
                attention_cols[0].warning(
                    (f"Attention {attention_count}: " if english else f"해야 할 일 {attention_count}건: ")
                    + " · ".join(category_bits)
                )
                if attention_cols[1].button(
                    "Open Attention" if english else "해야 할 일 보기",
                    key=f"rq-open-attention-{rq_id}",
                    use_container_width=True,
                ):
                    st.session_state["attention-focus-rq"] = rq_id
                    st.session_state["operating-desk-navigate"] = "attention"
                    st.rerun()
            elif str(item.get("execution_status") or "") == "attention_projection_missing":
                st.error(str(item.get("execution_detail") or ("Attention projection is missing." if english else "사람 입력 대기 상태이지만 Attention 연결이 없습니다.")))
            else:
                st.caption("No researcher action is required right now." if english else "현재 바로 처리할 사람 작업은 없습니다.")

            action_left, action_mid, action_right = st.columns(3)
            followup_open_key = f"rq-followup-direction-open-{rq_id}"
            if request_additional_literature is not None and rq_version <= 1:
                followup_ready = bool(
                    item.get("answer_draft_approved")
                    and int(item.get("pending_knowledge_reviews") or 0) == 0
                    and not item.get("answer_work_in_progress")
                    and int(item.get("paper_review_pending") or 0) == 0
                )
                if action_left.button(
                    "Revise Research Question" if english else "연구질문 수정",
                    key=f"rq-additional-literature-{rq_id}",
                    use_container_width=True,
                    disabled=not followup_ready,
                    help=(
                        "Enabled after the current literature round is complete and the latest Research Answer draft is researcher-approved." if english else
                        "현재 문헌 라운드의 논문 해석·Knowledge Card 결정을 마치고 최신 Research Answer Draft를 승인한 뒤 활성화됩니다."
                    ),
                ):
                    st.session_state[followup_open_key] = True
                    st.rerun()
            if request_initial_answer is not None:
                initial_ready = bool(
                    knowledge_count > 0
                    and not item.get("answer_draft")
                    and not item.get("answer_work_in_progress")
                    and not item.get("answer_confirmation_pending")
                    and int(item.get("pending_knowledge_reviews") or 0) == 0
                )
                if action_mid.button(
                    "Draft answer" if english else "답변 초안 작성",
                    key=f"rq-answer-draft-{rq_id}",
                    use_container_width=True,
                    disabled=not initial_ready,
                    help=(
                        "Start from accumulated approved knowledge. Attention is created only if an external LLM step is needed." if english else
                        "누적 승인 지식을 바탕으로 초안 작성을 시작합니다. 외부 LLM 단계가 필요한 경우 이후 Attention이 생성됩니다."
                    ),
                ):
                    try:
                        request_initial_answer(rq_id)
                        st.session_state["research-action-flash"] = (
                            "Answer drafting started. Next: complete the RQ answer Input in Attention." if english else
                            "답변 초안 작성을 시작했습니다. 다음 작업: Attention에 생성된 RQ 답변 Input을 수행하세요."
                        )
                        st.session_state["attention-focus-rq"] = rq_id
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
                if answer_work_status:
                    action_mid.caption(("Current: " if english else "현재: ") + answer_work_status)
            if request_answer_update is not None:
                update_ready = bool(item.get("answer_update_available"))
                if action_right.button(
                    "Update answer" if english else "답변 업데이트",
                    key=f"rq-answer-update-{rq_id}",
                    use_container_width=True,
                    disabled=not update_ready,
                    help=(
                        "Enabled when literature or knowledge changed after the latest answer." if english else
                        "최신 답변 이후 문헌 조사 또는 지식카드 변화가 있을 때 활성화됩니다."
                    ),
                ):
                    try:
                        request_answer_update(rq_id)
                        st.session_state["research-action-flash"] = (
                            "Answer update started. Next: complete the RQ answer Input in Attention." if english else
                            "변경된 근거를 반영한 답변 업데이트를 시작했습니다. 다음 작업: Attention에 생성된 RQ 답변 Input을 수행하세요."
                        )
                        st.session_state["attention-focus-rq"] = rq_id
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

            if item.get("answer_draft") and review_answer_draft is not None:
                st.divider()
                if rq_version > 1:
                    st.caption("이전 라운드 결과 · 참고")
                version = int(item.get("answer_version") or 1)
                approved = bool(item.get("answer_draft_approved"))
                status_text = ("Approved" if english else "승인됨") if approved else ("Review required" if english else "승인 필요")
                st.markdown((f"### Research Answer v{version} · {status_text}" if english else f"### 연구질문 답변 v{version} · {status_text}"))
                if approved:
                    st.success(
                        "The latest draft is approved. You can start the next literature round." if english else
                        "최신 Draft가 승인되었습니다. 다음 문헌 라운드를 시작할 수 있습니다."
                    )
                else:
                    st.warning(
                        "Review and approve the latest draft before starting Round 2." if english else
                        "Round 2로 넘어가기 전에 최신 Draft를 검토하고 승인하세요."
                    )
                with st.expander(
                    f"Latest Research Answer v{version}" if english else f"최신 연구질문 답변 v{version} 보기",
                    expanded=False,
                ):
                    st.markdown(str(item.get("answer_draft") or ""))
                    from research_fellow.application.research_answer_export import answer_markdown_bytes, answer_html_bytes, answer_pdf_bytes
                    export_title = str(item.get("question") or f"Research Answer v{version}")
                    base_name = f"research-answer-{rq_id}-v{version}"
                    export_cols = st.columns(3)
                    export_cols[0].download_button(
                        "Markdown" if english else "Markdown 내보내기",
                        data=answer_markdown_bytes(str(item.get("answer_draft") or "")),
                        file_name=f"{base_name}.md", mime="text/markdown; charset=utf-8",
                        key=f"rq-answer-export-md-{rq_id}-{version}", use_container_width=True,
                    )
                    export_cols[1].download_button(
                        "HTML" if english else "HTML 내보내기",
                        data=answer_html_bytes(str(item.get("answer_draft") or ""), title=export_title),
                        file_name=f"{base_name}.html", mime="text/html; charset=utf-8",
                        key=f"rq-answer-export-html-{rq_id}-{version}", use_container_width=True,
                    )
                    try:
                        pdf_data = answer_pdf_bytes(str(item.get("answer_draft") or ""), title=export_title)
                    except Exception as error:
                        export_cols[2].caption(("PDF export error: " if english else "PDF 생성 오류: ") + str(error))
                    else:
                        export_cols[2].download_button(
                            "PDF" if english else "PDF 내보내기", data=pdf_data,
                            file_name=f"{base_name}.pdf", mime="application/pdf",
                            key=f"rq-answer-export-pdf-{rq_id}-{version}", use_container_width=True,
                        )

                reopen_key = f"rq-answer-reopen-{rq_id}-{version}"
                show_review_editor = not approved
                if approved:
                    if st.button(
                        "Edit approved answer" if english else "승인본 수정하기",
                        key=f"rq-answer-reopen-button-{rq_id}-{version}",
                        type="secondary",
                    ):
                        st.session_state[reopen_key] = True
                    show_review_editor = bool(st.session_state.get(reopen_key))
                    if show_review_editor:
                        st.info(
                            "Saving an edit creates a new draft version that must be approved again." if english else
                            "수정 내용을 저장하면 새 Draft 버전이 생성되며 다시 승인이 필요합니다."
                        )

                if show_review_editor:
                    with st.expander(
                        "Edit / approve draft" if english else "Draft 수정 · 승인",
                        expanded=True,
                    ):
                        draft_review = render_interaction(
                            st,
                            "review_research_answer_draft",
                            {
                                "research_answer_review_context": {
                                    "rq_id": rq_id,
                                    "question": str(item.get("question") or ""),
                                    "markdown": str(item.get("answer_draft") or ""),
                                    "answer_version": version,
                                    "approved": approved,
                                }
                            },
                            key=f"rq-answer-review-{rq_id}-{version}",
                        )
                        if draft_review.submitted:
                            try:
                                resolution = dict(draft_review.values.get("research_answer_review_resolution") or {})
                                outcome = dict(review_answer_draft(rq_id, resolution))
                                action = str(resolution.get("action") or "")
                                st.session_state.pop(reopen_key, None)
                                if action == "approve":
                                    st.session_state["research-action-flash"] = (
                                        "Research Answer approved. The next literature round is now available." if english else
                                        "Research Answer Draft를 승인했습니다. 이제 다음 문헌 라운드를 시작할 수 있습니다."
                                    )
                                else:
                                    st.session_state["research-action-flash"] = (
                                        "Edited Markdown draft saved." if english else "수정한 Markdown Draft를 저장했습니다."
                                    )
                                st.rerun()
                            except ValueError as error:
                                st.error(str(error))

            if request_additional_literature is not None and st.session_state.get(followup_open_key):
                st.divider()
                st.caption(
                    "Define the deeper research question for Round 2."
                    if english else
                    "2차 문헌탐색에서 더 깊이 확인할 연구질문을 입력하세요."
                )
                st.info(
                    "Submitting the deep research question starts Round 2 literature discovery."
                    if english else
                    "심화 연구질문을 제출하면 2차 문헌탐색 라운드가 시작됩니다."
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
                            f"Round 2 literature exploration started: {direction}." if english else
                            f"심화 연구질문을 반영해 2차 문헌탐색을 시작했습니다: {direction}."
                        )
                        st.session_state["attention-focus-rq"] = rq_id
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

            if add_local_paper_to_rq is not None:
                with st.expander("+ Add Local PDF Paper" if english else "+ PDF 논문 추가", expanded=False):
                    st.caption(
                        "Attach a local paper directly to this research question and review it without running literature discovery."
                        if english else
                        "이 연구질문에 로컬 논문을 직접 추가하고 문헌 탐색 없이 바로 리뷰합니다."
                    )
                    local_pdf = st.file_uploader(
                        "Paper PDF" if english else "논문 PDF", type=["pdf"],
                        key=f"rq-local-paper-file-{rq_id}"
                    )
                    extracted_meta: Mapping[str, Any] = {}
                    pdf_fingerprint = "none"
                    if local_pdf is not None:
                        local_bytes = bytes(local_pdf.getvalue())
                        import hashlib
                        pdf_fingerprint = hashlib.sha256(local_bytes).hexdigest()[:10]
                        if extract_local_paper_metadata is not None:
                            try:
                                extracted_meta = dict(extract_local_paper_metadata(str(getattr(local_pdf, "name", "paper.pdf")), local_bytes))
                            except ValueError as error:
                                st.warning(str(error))
                    default_local_title = str(extracted_meta.get("title") or (str(getattr(local_pdf, "name", "")).rsplit(".", 1)[0] if local_pdf is not None else ""))
                    default_local_year = str(extracted_meta.get("publication_year") or "")
                    default_local_authors = ", ".join(str(x) for x in (extracted_meta.get("authors") or []))
                    if local_pdf is not None:
                        st.caption("PDF metadata extracted automatically; edit if needed." if english else "PDF 메타데이터를 자동 추출했습니다. 필요하면 수정하세요.")
                    local_title = st.text_input(
                        "Paper title" if english else "논문 제목", value=default_local_title,
                        key=f"rq-local-paper-title-{rq_id}-{pdf_fingerprint}"
                    )
                    local_meta_left, local_meta_right = st.columns(2)
                    local_year = local_meta_left.text_input("Year" if english else "발행 연도", value=default_local_year, key=f"rq-local-paper-year-{rq_id}-{pdf_fingerprint}")
                    local_authors = local_meta_right.text_input("Authors" if english else "저자", value=default_local_authors, key=f"rq-local-paper-authors-{rq_id}-{pdf_fingerprint}")
                    result_key = f"rq-local-paper-result-{rq_id}"
                    recent_local = st.session_state.get(result_key)
                    if isinstance(recent_local, Mapping):
                        meta = dict(recent_local.get("metadata") or {})
                        round_note = ("Added to the current unfinished literature round." if recent_local.get("reused_current_round") else "Added as a manual paper-review context.") if english else ("현재 완료되지 않은 문헌조사 라운드에 추가했습니다." if recent_local.get("reused_current_round") else "독립 논문 리뷰 맥락으로 추가했습니다.")
                        st.success(f"{round_note} {meta.get('title') or ''}")
                        if recent_local.get("full_text_extraction_note"):
                            st.caption(str(recent_local.get("full_text_extraction_note")))
                        prompt = str(recent_local.get("review_prompt") or "")
                        if prompt:
                            with st.expander("Paper Review prompt" if english else "논문 리뷰 프롬프트", expanded=True):
                                st.code(prompt, language=None)
                                st.caption("Copy this prompt to the external LLM, then paste the response below." if english else "이 프롬프트를 외부 LLM에 복사한 뒤, 응답을 아래에 붙여넣으세요.")
                        if submit_local_paper_review is not None:
                            review_response = st.text_area(
                                "Paper Review response" if english else "논문 리뷰 응답 붙여넣기",
                                key=f"rq-local-paper-review-response-{rq_id}",
                                height=240,
                                placeholder="Paste the JSON response from the external LLM." if english else "외부 LLM의 JSON 응답을 붙여넣으세요.",
                            )
                            if st.button(
                                "Apply Review" if english else "리뷰 결과 반영",
                                key=f"rq-local-paper-review-apply-{rq_id}",
                                use_container_width=True,
                                disabled=not str(review_response or "").strip(),
                            ):
                                try:
                                    applied = dict(submit_local_paper_review(recent_local, review_response))
                                    st.session_state[f"rq-local-paper-applied-{rq_id}"] = applied
                                    request_ids = [str(x) for x in (applied.get("knowledge_request_ids") or []) if str(x)]
                                    st.session_state["research-action-flash"] = (
                                        (f"Paper Review saved. Moving to Knowledge Card Decision ({len(request_ids)} candidate(s))." if request_ids else "Paper Review saved.")
                                        if english else
                                        (f"논문 리뷰를 저장했습니다. 지식카드 Decision {len(request_ids)}건으로 이동합니다." if request_ids else "논문 리뷰를 저장했습니다.")
                                    )
                                    # The canonical review write path already created Knowledge Card
                                    # decision_request phenomena.  Move directly to the owning RQ's
                                    # Decision queue instead of leaving the researcher on the review form.
                                    if request_ids:
                                        st.session_state["attention-focus-rq"] = rq_id
                                        st.session_state["operating-desk-navigate"] = "attention"
                                        st.session_state.pop(result_key, None)
                                        st.session_state.pop(f"rq-local-paper-review-response-{rq_id}", None)
                                    st.rerun()
                                except ValueError as error:
                                    st.error(str(error))
                        applied = st.session_state.get(f"rq-local-paper-applied-{rq_id}")
                        if isinstance(applied, Mapping):
                            summary = str(applied.get("summary") or "").strip()
                            if summary:
                                with st.expander("Review result" if english else "논문 리뷰 결과", expanded=True):
                                    st.markdown(summary)
                            cards = list(applied.get("knowledge_cards") or [])
                            if cards:
                                st.caption((f"Knowledge Card candidates: {len(cards)}" if english else f"지식카드 후보 {len(cards)}건이 생성되었습니다."))
                    if st.button(
                        "Add and Review" if english else "추가하고 리뷰", type="primary",
                        key=f"rq-local-paper-submit-{rq_id}", use_container_width=True,
                        disabled=local_pdf is None,
                    ):
                        try:
                            outcome = dict(add_local_paper_to_rq(
                                rq_id,
                                {"title": local_title, "publication_year": local_year, "authors": local_authors},
                                str(getattr(local_pdf, "name", "paper.pdf")), bytes(local_pdf.getvalue()),
                            ))
                            st.session_state[result_key] = outcome
                            st.session_state["research-action-flash"] = (
                                "Local paper added to the current review work. The Paper Review prompt is ready below." if english else
                                "로컬 논문을 현재 리뷰 작업에 추가했습니다. 아래에 논문 리뷰 프롬프트를 준비했습니다."
                            )
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

            with st.expander("Progress details" if english else "세부 진행상황", expanded=False):
                st.caption(f"{lifecycle_label} · {status_label}")
                if item.get("rationale"):
                    st.write(_short(item.get("rationale", ""), 500))
                if item.get("researcher_comment"):
                    st.markdown("**Researcher comment**" if english else "**연구자 코멘트**")
                    st.write(_short(item.get("researcher_comment", ""), 700))
                detail_cols = st.columns(4)
                detail_cols[0].metric("Rounds" if english else "문헌 라운드", rounds)
                detail_cols[1].metric("Review pending" if english else "리뷰 대기", review_pending)
                detail_cols[2].metric("Review done" if english else "리뷰 완료", review_done)
                detail_cols[3].metric("Approved KC" if english else "승인 KC", knowledge_count)
                next_action = str(item.get("next_agent_action") or "").strip()
                if next_action:
                    st.write(("**Next agent action:** " if english else "**다음 Agent 행동:** ") + next_action)
                execution_detail = str(item.get("execution_detail") or "").strip()
                if execution_detail:
                    st.caption(execution_detail)
                if item.get("progress_detail"):
                    st.caption(_short(item.get("progress_detail", ""), 350))

                st.markdown("**Research flow**" if english else "**연구 진행 흐름**")
                flow = [
                    ("Literature exploration" if english else "문헌 탐구", bool(rounds)),
                    ("Paper / knowledge review" if english else "논문·지식 검토", bool(item.get("literature_round_completed"))),
                    ("Knowledge registered" if english else "지식 등록", knowledge_count > 0),
                    ("Answer draft" if english else "답변 초안", bool(item.get("answer_draft"))),
                ]
                st.caption("  →  ".join(("✓ " if done else "○ ") + label for label, done in flow))

                st.markdown("**Evidence for this question**" if english else "**이 연구질문의 근거**")
                direct_cols = st.columns(2)
                direct_cols[0].metric("Direct papers" if english else "직접 탐색 논문", direct_papers)
                direct_cols[1].metric("Direct knowledge" if english else "직접 생성 지식", int(item.get("direct_evidence_cards") or 0))
                direct_titles = [x for x in item.get("direct_evidence_titles") or [] if x]
                if direct_titles:
                    st.caption(("Primary lineage: " if english else "1차 근거 계보: ") + " · ".join(_short(x, 80) for x in direct_titles[:5]))
                support_cols = st.columns(2)
                support_cols[0].metric("Existing knowledge" if english else "기존 지식카드", int(item.get("supporting_knowledge_cards") or 0))
                support_cols[1].metric("Existing papers" if english else "기존 논문", int(item.get("supporting_papers") or 0))
                support_titles = [x for x in item.get("supporting_evidence_titles") or [] if x]
                if support_titles:
                    st.caption(("Supporting: " if english else "보강: ") + " · ".join(_short(x, 80) for x in support_titles[:5]))

                if item.get("answer_draft"):
                    version = int(item.get("answer_version") or 1)
                    approval_label = (("approved" if english else "승인됨") if item.get("answer_draft_approved") else ("pending" if english else "승인 필요"))
                    st.caption((f"Research answer v{version} · Approval: {approval_label}" if english else f"연구질문 답변 v{version} · 승인 상태: {approval_label}"))
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
