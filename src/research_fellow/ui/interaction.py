"""Streamlit realization of Interaction Contracts through UI Bindings."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from research_fellow.application.dsl.interaction import interaction_contract
from research_fellow.application.dsl.interaction_binding import interaction_binding
@dataclass(frozen=True)
class InteractionRenderResult:
    interaction_id: str
    submitted: bool
    values: dict[str, Any]
def _field_value(item: Any, field: str) -> Any:
    value = item
    for part in str(field).split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            return None
    return value

def _item_label(item: Any, field: str) -> str:
    value = _field_value(item, field)
    if value is not None:
        return str(value)
    return str(item)
def _render_multi_select(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id); binding = interaction_binding(interaction_id, profile); renderer = binding.renderer
    input_name = contract.required_inputs[0]; output_name = contract.outputs[0]
    items = list(inputs.get(input_name) or []); option_by_key = {str(index): item for index, item in enumerate(items)}
    with st.form(f"interaction-{key}"):
        selected_keys = st.multiselect(str(renderer["title"]), list(option_by_key), format_func=lambda value: _item_label(option_by_key[value], str(renderer["item_label"])), help=str(renderer.get("help") or "") or None, key=f"interaction-{key}-selection")
        for selected_key in selected_keys:
            item = option_by_key[selected_key]
            if isinstance(item, Mapping):
                values = [str(item.get(field) or "").strip() for field in list(renderer.get("item_detail") or [])]
                values = [value for value in values if value]
                if values: st.caption(" · ".join(values))
        submitted = st.form_submit_button(str(renderer.get("submit_label") or "Submit"), type="primary")
    selected = [option_by_key[value] for value in selected_keys]
    if submitted and not selected and renderer.get("empty_label"): st.info(str(renderer["empty_label"]))
    return InteractionRenderResult(interaction_id, submitted, {output_name: selected})

def _render_confirm(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    with st.form(f"interaction-{key}"):
        st.markdown(f"**{renderer['title']}**")
        help_text = str(renderer.get("help") or "").strip()
        if help_text:
            st.caption(help_text)
        plan = inputs.get(contract.required_inputs[0]) if contract.required_inputs else None
        if plan is not None:
            decision_question = getattr(plan, "decision_question", None)
            if decision_question:
                st.write(f"**핵심 판단:** {decision_question}")
            subquestions = list(getattr(plan, "subquestions", ()) or ())
            for index, item in enumerate(subquestions, start=1):
                question = getattr(item, "question", None) or str(item)
                st.write(f"{index}. {question}")
        confirmed = st.form_submit_button(str(renderer.get("confirm_label") or "Confirm"), type="primary")
        cancelled = st.form_submit_button(str(renderer.get("cancel_label") or "Cancel"))
    submitted = bool(confirmed or cancelled)
    output_name = contract.outputs[0]
    return InteractionRenderResult(interaction_id, submitted, {output_name: bool(confirmed)})



def _render_approval(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    input_name = contract.required_inputs[0]
    output_name = contract.outputs[0]
    items = list(inputs.get(input_name) or [])
    id_field = str(renderer["item_id"])
    label_field = str(renderer["item_label"])
    item_by_id = {str(_field_value(item, id_field)): item for item in items}
    option_ids = [item_id for item_id in item_by_id if item_id and item_id != "None"]
    with st.form(f"interaction-{key}"):
        st.markdown(f"**{renderer['title']}**")
        help_text = str(renderer.get("help") or "").strip()
        if help_text:
            st.caption(help_text)
        selected_ids = st.multiselect(
            str(renderer.get("select_all_label") or "검토할 안건"),
            option_ids,
            default=option_ids if bool(renderer.get("default_all")) else [],
            format_func=lambda value: _item_label(item_by_id[value], label_field),
            key=f"interaction-{key}-selection",
        )
        note = st.text_input(str(renderer["comment_label"]), key=f"interaction-{key}-comment")
        approved = st.form_submit_button(str(renderer["approve_label"]), type="primary")
        deferred = st.form_submit_button(str(renderer["defer_label"]))
        rejected = st.form_submit_button(str(renderer["reject_label"]))
    submitted = bool(approved or deferred or rejected)
    if approved:
        decision = "approved"
    elif deferred:
        decision = "deferred"
    elif rejected:
        decision = "rejected"
    else:
        decision = ""
    output_id_field = str(renderer.get("output_id_field") or "request_id")
    resolutions = [
        {output_id_field: item_id, "decision": decision, "note": note}
        for item_id in selected_ids
    ] if submitted else []
    if submitted and not selected_ids and renderer.get("empty_label"):
        st.info(str(renderer["empty_label"]))
    return InteractionRenderResult(interaction_id, submitted and bool(selected_ids), {output_name: resolutions})



def _render_review_form(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    variant = str(renderer.get("variant") or "")
    output_name = contract.outputs[0]
    if variant == "ontology_change":
        review = dict(inputs.get("ontology_change_review") or {})
        types = [dict(x) for x in (inputs.get("ontology_types") or [])]
        relations = [dict(x) for x in (inputs.get("ontology_relations") or [])]
        type_ids = [str(x.get("type_id") or "") for x in types if x.get("type_id")]
        type_by_id = {str(x.get("type_id")): x for x in types}
        relation_ids = [str(x.get("relation_id") or "") for x in relations if x.get("relation_id")]
        relation_by_id = {str(x.get("relation_id")): x for x in relations}
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"): st.caption(str(renderer["help"]))
            kinds = ["선택 안 함", "타입 이름·설명 수정", "관계 이름·설명 수정", "관계 추가", "관계 삭제"]
            opinion_kind = st.selectbox("구조화 의견", kinds, key=f"interaction-{key}-kind")
            structured = ""
            if opinion_kind == "타입 이름·설명 수정" and type_ids:
                type_id = st.selectbox("수정할 타입", type_ids, format_func=lambda v: str(type_by_id[v].get("name") or v), key=f"interaction-{key}-type")
                name = st.text_input("새 타입 이름", value=str(type_by_id[type_id].get("name") or ""), key=f"interaction-{key}-type-name")
                desc = st.text_area("새 타입 설명", value=str(type_by_id[type_id].get("description") or ""), key=f"interaction-{key}-type-desc")
                structured = f"타입 수정: type_id={type_id}, 이름='{name}', 설명='{desc}'."
            elif opinion_kind in {"관계 이름·설명 수정", "관계 삭제"} and relation_ids:
                rid = st.selectbox("대상 관계", relation_ids, key=f"interaction-{key}-relation")
                if opinion_kind == "관계 삭제":
                    structured = f"관계 삭제: relation_id={rid}."
                else:
                    rel = relation_by_id[rid]
                    name = st.text_input("새 관계 이름", value=str(rel.get("relation_name") or ""), key=f"interaction-{key}-relation-name")
                    desc = st.text_area("새 관계 설명", value=str(rel.get("description") or ""), key=f"interaction-{key}-relation-desc")
                    structured = f"관계 수정: relation_id={rid}, 이름='{name}', 설명='{desc}'."
            elif opinion_kind == "관계 추가" and len(type_ids) >= 2:
                source = st.selectbox("출발 타입", type_ids, format_func=lambda v: str(type_by_id[v].get("name") or v), key=f"interaction-{key}-source")
                target = st.selectbox("도착 타입", type_ids, index=1, format_func=lambda v: str(type_by_id[v].get("name") or v), key=f"interaction-{key}-target")
                name = st.text_input("관계 이름", key=f"interaction-{key}-new-relation-name")
                desc = st.text_area("관계 설명", key=f"interaction-{key}-new-relation-desc")
                structured = f"관계 추가: source_type_id={source}, target_type_id={target}, 이름='{name}', 설명='{desc}'."
            comment = st.text_area("자유 수정·보완 요청", value=str(review.get("researcher_comment") or ""), key=f"interaction-{key}-comment")
            saved = st.form_submit_button(str(renderer.get("submit_label") or "검토 결과 저장"))
            regenerate = st.form_submit_button(str(renderer.get("regenerate_label") or "수정안 재생성"), type="primary")
        combined = "\n".join(x for x in [structured, comment.strip()] if x)
        feedback = {"review_id": str(review.get("review_id") or ""), "structured_comment": structured, "comment": comment.strip(), "combined_comment": combined, "regenerate_requested": bool(regenerate)}
        return InteractionRenderResult(interaction_id, bool(saved or regenerate), {output_name: feedback})
    if variant == "revision_reconciliation":
        proposal = dict(inputs.get("reconciliation_result") or {})
        existing = [dict(x) for x in (inputs.get("existing_todos") or [])]
        labels={"resolved":"해결","retained":"유지","modified":"수정","obsolete":"불필요","merged":"통합","split":"분리","researcher_review":"연구자 판단"}
        statuses=list(labels)
        reviewed=[]; selected_new=[]
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"): st.caption(str(renderer["help"]))
            for item in proposal.get("existing_todo_assessments") or []:
                todo_id=str(item.get("todo_id") or "")
                include=st.checkbox(f"`{todo_id}` 평가 반영", value=True, key=f"interaction-{key}-include-{todo_id}")
                default=str(item.get("status") or "retained")
                status=st.selectbox("상태", statuses, index=statuses.index(default) if default in statuses else 1, format_func=lambda v: labels[v], key=f"interaction-{key}-status-{todo_id}")
                reason=st.text_area("판단 사유", value=str(item.get("reason") or ""), key=f"interaction-{key}-reason-{todo_id}")
                problem=st.text_area("다음 작업 내용", value=str(item.get("updated_problem") or ""), key=f"interaction-{key}-problem-{todo_id}")
                criterion=st.text_area("완료 기준", value=str(item.get("updated_completion_criteria") or ""), key=f"interaction-{key}-criterion-{todo_id}")
                if include: reviewed.append({**item,"status":status,"reason":reason.strip(),"updated_problem":problem.strip(),"updated_completion_criteria":criterion.strip()})
            for item in proposal.get("new_todos") or []:
                todo_id=str(item.get("todo_id") or "")
                if st.checkbox(f"[{item.get('priority','P1')}] {item.get('label','신규')} · {item.get('problem','')}", value=True, key=f"interaction-{key}-new-{todo_id}"):
                    selected_new.append(dict(item))
            submitted = st.form_submit_button(str(renderer.get("submit_label") or "검토 결과 반영"), type="primary")
        result={"summary":proposal.get("summary", ""),"revision_achievements":proposal.get("revision_achievements") or [],"next_revision_recommendations":proposal.get("next_revision_recommendations") or [],"assessments":reviewed,"new_todos":selected_new,"existing_todos":existing}
        return InteractionRenderResult(interaction_id, bool(submitted), {output_name: result})
    if variant == "research_question_triage":
        rq = dict(inputs.get("research_question") or {})
        labels = {"interested":"관심", "exploring":"탐색중", "hold":"보류", "completed":"완료", "rejected":"제외"}
        statuses = list(labels)
        current = str(rq.get("status") or "interested")
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"): st.caption(str(renderer["help"]))
            st.write(str(rq.get("question") or ""))
            status = st.radio("상태", statuses, index=statuses.index(current) if current in statuses else 0, horizontal=True, format_func=lambda v: labels[v], key=f"interaction-{key}-status")
            note = st.text_input("판단 메모 (선택)", key=f"interaction-{key}-note")
            submitted = st.form_submit_button(str(renderer.get("submit_label") or "상태 반영"), type="primary")
        return InteractionRenderResult(interaction_id, bool(submitted), {output_name: {"rq_id": str(rq.get("rq_id") or ""), "status": status, "note": note.strip()}})

    if variant == "external_advisory_interpretation":
        request = dict(inputs.get("external_advisory_request") or {})
        proposed = dict(inputs.get("proposed_interpretation") or {})
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"): st.caption(str(renderer["help"]))
            st.caption(f"요청자: {request.get('requester') or '미지정'}")
            st.write(str(request.get("request") or ""))
            interpretation = st.text_area("M2 해석", value=str(proposed.get("interpretation") or proposed.get("question") or ""), height=130, key=f"interaction-{key}-interpretation")
            question = st.text_area("Thread에서 관리할 질문", value=str(proposed.get("question") or ""), height=110, key=f"interaction-{key}-question")
            submitted = st.form_submit_button(str(renderer.get("submit_label") or "Thread 시작"), type="primary")
        if submitted and not question.strip():
            st.info("Thread에서 관리할 질문을 입력하세요.")
            submitted = False
        return InteractionRenderResult(interaction_id, bool(submitted), {output_name: {"interpreted_question": question.strip(), "interpretation": interpretation.strip()}})

    if variant == "paper_reading_claim":
        paper = dict(inputs.get("paper") or {})
        item = dict(inputs.get("reading_question") or {})
        duplicates = [dict(x) for x in (inputs.get("duplicate_candidates") or [])]
        context = str(inputs.get("research_context") or "")
        evidence_levels = {
            "empirical":"실증 — 논문의 데이터·실험·사례가 직접 뒷받침",
            "theoretical":"이론 — 개념적·논리적 논증이 중심",
            "review":"문헌 종합 — 여러 선행 연구를 검토·종합",
            "provisional":"잠정 — 연구자의 해석이거나 추가 검증 필요",
        }
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"): st.caption(str(renderer["help"]))
            decision_options=["register","defer","irrelevant"]
            existing = "defer" if item.get("status") == "deferred" else "irrelevant" if item.get("status") == "irrelevant" else "register"
            decision=st.radio("연구자 결정", decision_options, index=decision_options.index(existing), horizontal=True, format_func={"register":"지식카드 등록","defer":"보류","irrelevant":"무관"}.get, key=f"interaction-{key}-decision")
            comment=st.text_area("근거 해석·첨삭", value=str(item.get("researcher_comment") or ""), key=f"interaction-{key}-comment")
            evidence_text=st.text_area("원문 근거 (한 줄에 하나 · 최대 5개)", value="\n".join(list(item.get("evidence") or [])[:5]), key=f"interaction-{key}-evidence")
            card_title=st.text_input("카드 제목", value=str(item.get("suggested_title") or ""), key=f"interaction-{key}-title")
            card_claim=st.text_area("주장 (Claim)", value=str(item.get("tentative_answer") or ""), key=f"interaction-{key}-claim")
            card_context=st.text_area("지식 맥락", value=context, height=150, key=f"interaction-{key}-context")
            card_labels=st.text_input("레이블 (쉼표 구분)", value=str(item.get("suggested_labels") or ""), key=f"interaction-{key}-labels")
            card_concepts=st.text_input("핵심 개념 (쉼표 구분)", value=str(item.get("suggested_concepts") or ""), key=f"interaction-{key}-concepts")
            card_applies_to=st.text_input("적용 대상 (쉼표 구분)", value=str(item.get("suggested_applies_to") or ""), key=f"interaction-{key}-applies")
            card_conditions=st.text_area("적용 조건", value=str(item.get("suggested_conditions") or ""), key=f"interaction-{key}-conditions")
            card_limits=st.text_area("한계·유보", value=str(item.get("suggested_limits") or item.get("uncertainty") or ""), key=f"interaction-{key}-limits")
            card_evidence_level=st.selectbox("근거 수준", list(evidence_levels), format_func=evidence_levels.get, key=f"interaction-{key}-level")
            duplicate_mode="separate"; duplicate_target_id=""
            if duplicates:
                duplicate_mode=st.radio("유사 카드 처리", ["enrich","separate","defer"], horizontal=True, format_func={"enrich":"기존 카드 근거 보강","separate":"별도 카드 등록","defer":"보류"}.get, key=f"interaction-{key}-duplicate-mode")
                options={str(x.get("card_id")): f"{x.get('title','')} · {str(x.get('claim',''))[:60]}" for x in duplicates if x.get("card_id")}
                if options:
                    duplicate_target_id=st.selectbox("근거를 보강할 기존 카드", list(options), format_func=lambda v: options[v], key=f"interaction-{key}-duplicate-target")
            submitted=st.form_submit_button(str(renderer.get("submit_label") or "판단 저장"), type="primary")
        evidence=[line.strip(" -•") for line in evidence_text.splitlines() if line.strip(" -•")]
        review={"decision":decision,"comment":comment.strip(),"evidence":evidence,"card_title":card_title.strip(),"card_claim":card_claim.strip(),"card_context":card_context.strip(),"card_labels":card_labels.strip(),"card_concepts":card_concepts.strip(),"card_applies_to":card_applies_to.strip(),"card_conditions":card_conditions.strip(),"card_limits":card_limits.strip(),"card_evidence_level":card_evidence_level,"duplicate_mode":duplicate_mode,"duplicate_target_id":duplicate_target_id}
        return InteractionRenderResult(interaction_id, bool(submitted), {output_name: review})

    raise NotImplementedError(f"Unknown review_form variant {variant!r}: {interaction_id}")



def _render_text_input(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    renderer = interaction_binding(interaction_id, profile).renderer
    output_name = contract.outputs[0]
    values: dict[str, Any] = {}
    with st.form(f"interaction-{key}"):
        st.markdown(f"**{renderer['title']}**")
        help_text = str(renderer.get("help") or "").strip()
        if help_text:
            st.caption(help_text)
        for field in list(renderer.get("fields") or []):
            name = str(field["name"]); label = str(field["label"]); widget = str(field.get("widget") or "text_input")
            kwargs = {"key": f"interaction-{key}-{name}"}
            if field.get("placeholder"):
                kwargs["placeholder"] = str(field["placeholder"])
            if field.get("help"):
                kwargs["help"] = str(field["help"])
            if widget == "text_area":
                kwargs["height"] = int(field.get("height") or 100)
                value = st.text_area(label, **kwargs)
            else:
                value = st.text_input(label, **kwargs)
            values[name] = value
        submitted = st.form_submit_button(str(renderer.get("submit_label") or "Submit"), type="primary")
    if submitted:
        missing = [str(f["label"]) for f in list(renderer.get("fields") or []) if bool(f.get("required")) and not str(values.get(str(f["name"])) or "").strip()]
        if missing:
            st.info("필수 입력을 확인하세요: " + ", ".join(missing))
            submitted = False
    return InteractionRenderResult(interaction_id, submitted, {output_name: values})

def _render_message(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    renderer = interaction_binding(interaction_id, profile).renderer
    input_name = contract.required_inputs[0]
    raw = inputs.get(input_name)
    items = list(raw or []) if isinstance(raw, (list, tuple)) else ([] if raw is None else [raw])
    st.markdown(f"**{renderer['title']}**")
    help_text = str(renderer.get("help") or "").strip()
    if help_text:
        st.caption(help_text)
    if not items and renderer.get("empty_label"):
        st.caption(str(renderer["empty_label"]))
    for item in items:
        st.write(_item_label(item, str(renderer["item_label"])))
        details = [str(_field_value(item, field) or "").strip() for field in list(renderer.get("item_detail") or [])]
        details = [value for value in details if value]
        if details:
            st.caption(" · ".join(details))
    return InteractionRenderResult(interaction_id, False, {})

def render_interaction(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str = "default") -> InteractionRenderResult:
    contract = interaction_contract(interaction_id); binding = interaction_binding(interaction_id, profile)
    missing = [name for name in contract.required_inputs if name not in inputs]
    if missing: raise ValueError(f"Missing interaction inputs {missing}: {interaction_id}")
    if binding.renderer_type == "message": return _render_message(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "text_input": return _render_text_input(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "multi_select": return _render_multi_select(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "confirm": return _render_confirm(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "approval": return _render_approval(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "review_form": return _render_review_form(st, interaction_id, inputs, key=key, profile=profile)
    raise NotImplementedError(f"Renderer {binding.renderer_type!r} is declared but not implemented by the Streamlit adapter yet")



def render_interaction_with_autonomy(
    st: Any,
    interaction_id: str,
    inputs: Mapping[str, Any],
    *,
    signals: Mapping[str, Any] | None,
    key: str,
    profile: str = "default",
    audit_recorder: Any | None = None,
) -> InteractionRenderResult:
    """Resolve an interaction autonomously when policy permits, otherwise render UI.

    The UI adapter does not contain policy rules.  Domain code supplies explicit
    signals and the autonomy layer decides whether the human boundary is needed.
    """
    from research_fellow.application.dsl.autonomy import (
        AUTONOMY_AUTO, AUTONOMY_AUTO_NOTIFY, evaluate_interaction_autonomy,
    )

    decision = evaluate_interaction_autonomy(
        interaction_id, signals or {}, inputs=inputs,
    )
    if decision.audit_record and callable(audit_recorder):
        audit_recorder({
            "workflow_id": "ui_interaction",
            "step": interaction_id,
            **decision.as_dict(),
        })
    if decision.action == AUTONOMY_AUTO:
        return InteractionRenderResult(interaction_id, True, dict(decision.resolution))
    if decision.action == AUTONOMY_AUTO_NOTIFY:
        contract = interaction_contract(interaction_id)
        if contract.requires_response:
            return InteractionRenderResult(interaction_id, True, dict(decision.resolution))
        # ``inform`` interactions still render their notification when a UI is
        # present; they simply do not gate workflow progress.
        return render_interaction(st, interaction_id, inputs, key=key, profile=profile)
    return render_interaction(st, interaction_id, inputs, key=key, profile=profile)

def render_waiting_workflow_interaction(
    st: Any,
    waiting_result: Mapping[str, Any],
    *,
    key: str,
    profile: str = "default",
) -> InteractionRenderResult:
    """Render the interaction request emitted by a suspended ``WorkflowRun``."""
    if str(waiting_result.get("status") or "") != "waiting_for_interaction":
        raise ValueError("Workflow result is not waiting for interaction")
    request = waiting_result.get("interaction")
    if not isinstance(request, Mapping):
        raise ValueError("Waiting workflow result has no interaction request")
    interaction_id = str(request.get("interaction_id") or "")
    inputs = request.get("inputs")
    if not interaction_id or not isinstance(inputs, Mapping):
        raise ValueError("Waiting workflow interaction request is incomplete")
    return render_interaction(st, interaction_id, inputs, key=key, profile=profile)
