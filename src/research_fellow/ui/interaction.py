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
        st.markdown(f"**{renderer['title']}**")
        help_text = str(renderer.get("help") or "").strip()
        if help_text:
            st.caption(help_text)
        if items:
            st.caption(f"후보 {len(items)}건 · 아래 결과를 확인한 뒤 보존할 항목을 선택하세요.")
            for index, item in enumerate(items, start=1):
                if not isinstance(item, Mapping):
                    continue
                st.markdown(f"**{index}. {_item_label(item, str(renderer['item_label']))}**")
                values = [str(item.get(field) or "").strip() for field in list(renderer.get("item_detail") or [])]
                values = [value for value in values if value]
                if values:
                    st.caption(" · ".join(values))
                link = str(item.get("url") or item.get("source_url") or "").strip()
                pdf_link = str(item.get("pdf_url") or "").strip()
                links = []
                if link:
                    links.append(f"[원문]({link})")
                if pdf_link and pdf_link != link:
                    links.append(f"[PDF]({pdf_link})")
                if links:
                    st.markdown(" · ".join(links))
                if index < len(items):
                    st.divider()
        selected_keys = st.multiselect("서재함에 보존할 항목", list(option_by_key), format_func=lambda value: _item_label(option_by_key[value], str(renderer["item_label"])), key=f"interaction-{key}-selection")
        submitted = st.form_submit_button(str(renderer.get("submit_label") or "Submit"), type="primary")
    selected = [option_by_key[value] for value in selected_keys]
    min_selection = int((contract.raw.get("constraints") or {}).get("min_selection") or 0)
    if submitted and len(selected) < min_selection:
        st.info(str(renderer.get("empty_label") or f"최소 {min_selection}개를 선택하세요."))
        submitted = False
    elif submitted and not selected and renderer.get("empty_label"):
        st.info(str(renderer["empty_label"]))
    return InteractionRenderResult(interaction_id, submitted, {output_name: selected})



def _render_sequential_paper_review(
    st: Any,
    interaction_id: str,
    inputs: Mapping[str, Any],
    *,
    key: str,
    profile: str,
    item_action: Any | None = None,
) -> InteractionRenderResult:
    """Review one literature candidate completely before moving to the next.

    Preserved papers are linked to the current RQ immediately, then reviewed
    against that RQ in the same Review card.  The generated Knowledge Card
    candidates are shown read-only here; approval is intentionally deferred to
    Attention > Decisions.
    """
    contract = interaction_contract(interaction_id)
    renderer = interaction_binding(interaction_id, profile).renderer
    input_name = contract.required_inputs[0]
    output_name = contract.outputs[0]
    items = [dict(item) for item in list(inputs.get(input_name) or []) if isinstance(item, Mapping)]
    index_key = f"interaction-{key}-paper-index"
    selected_key = f"interaction-{key}-paper-selected"
    reviewed_key = f"interaction-{key}-paper-reviewed"
    active_key = f"interaction-{key}-paper-active"
    stored_index = st.session_state.get(index_key)
    selected = list(st.session_state.get(selected_key, []) or [])
    reviewed = list(st.session_state.get(reviewed_key, []) or [])
    active = dict(st.session_state.get(active_key, {}) or {})

    # Rehydrate completed per-paper reviews from durable paper×RQ analyses.
    # Previously the progress lived only in Streamlit session_state, so a page
    # refresh reset the index and showed already-reviewed papers again.
    durable_selected = [dict(item) for item in items if bool(item.get("review_completed"))]
    if durable_selected:
        by_identity = {
            (str(x.get("paper_id") or ""), str(x.get("source_id") or ""), str(x.get("title") or "")): dict(x)
            for x in selected
        }
        for completed in durable_selected:
            identity = (str(completed.get("paper_id") or ""), str(completed.get("source_id") or ""), str(completed.get("title") or ""))
            by_identity.setdefault(identity, completed)
        selected = list(by_identity.values())

    if stored_index is None:
        index = next((i for i, item in enumerate(items) if not bool(item.get("review_completed"))), len(items))
    else:
        index = int(stored_index or 0)
        while index < len(items) and bool(items[index].get("review_completed")):
            index += 1
    st.session_state[index_key] = index
    st.session_state[selected_key] = selected

    st.markdown(f"**{renderer['title']}**")
    if renderer.get('help'):
        st.caption(str(renderer['help']))
    if not items:
        return InteractionRenderResult(interaction_id, True, {output_name: []})

    if index >= len(items):
        st.success(f"후보 {len(items)}편 검토 완료 · 서재함 연결 {len(selected)}편")
        st.caption("보존한 논문의 RQ 관점 요약과 Knowledge Card 후보 생성까지 완료되었습니다. Knowledge Card 승인은 Decisions에서 계속합니다.")
        submitted = st.button(str(renderer.get('submit_label') or '문헌조사 Review 완료'), type='primary', key=f"interaction-{key}-paper-finish")
        if submitted:
            result_selected = list(selected)
            for state_key in (index_key, selected_key, reviewed_key, active_key):
                st.session_state.pop(state_key, None)
            return InteractionRenderResult(interaction_id, True, {output_name: result_selected})
        return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})

    item = items[index]
    title = str(item.get('title') or 'Untitled paper')
    st.caption(f"논문 {index + 1} / {len(items)}")
    st.markdown(f"### {title}")
    authors = item.get('authors') or []
    authors_text = ', '.join(str(x) for x in authors if str(x).strip()) if isinstance(authors, list) else str(authors)
    meta = [x for x in [authors_text, str(item.get('venue') or ''), str(item.get('published') or item.get('publication_year') or ''), f"relevance {item.get('relevance_score')}" if item.get('relevance_score') is not None else ''] if x]
    if meta:
        st.caption(' · '.join(meta))

    if bool(item.get('already_in_library')):
        linked = [str(x.get('question') or '').strip() for x in (item.get('linked_research_questions') or []) if str(x.get('question') or '').strip()]
        st.info("이미 서재함에 있는 논문입니다." + ((" 연결된 연구질문: " + " · ".join(linked[:4])) if linked else ""))

    why = str(item.get('why_relevant') or '').strip()
    if why:
        st.write(f"**Why relevant:** {why}")
    summary = str(item.get('summary') or item.get('abstract_or_summary') or '').strip()
    if summary:
        with st.expander('초록 / 탐색 요약', expanded=True):
            st.write(summary)

    abstract_url = str(item.get('abstract_url') or item.get('source_url') or item.get('url') or '').strip()
    full_text_url = str(item.get('full_text_url') or '').strip()
    pdf_url = str(item.get('pdf_url') or '').strip()
    link_cols = st.columns(3)
    if abstract_url:
        link_cols[0].link_button('초록 / 서지', abstract_url, use_container_width=True)
    if full_text_url:
        link_cols[1].link_button('원문 URL', full_text_url, use_container_width=True)
    if pdf_url:
        link_cols[2].link_button('PDF', pdf_url, use_container_width=True)
    if not any((abstract_url, full_text_url, pdf_url)):
        st.warning('검증된 논문 URL이 없습니다.')

    edited_full_text = st.text_input(
        '원문 URL 직접 보정',
        value=str(active.get('full_text_url') or full_text_url),
        key=f"interaction-{key}-paper-fulltext-{index}",
        help='탐색 결과에 원문 URL이 없거나 잘못된 경우 연구자가 직접 입력할 수 있습니다.',
    ).strip()

    # Once preserved, this paper stays in review mode until its summary/cards are
    # generated. This prevents the Review card from disappearing into another tab.
    if active and int(active.get('index', -1)) == index:
        paper_id = str(active.get('paper_id') or '')
        st.success('서재함에 연결되었습니다. 이제 이 연구질문 관점에서 논문을 해석합니다.')
        local_pdf = st.file_uploader(
            '로컬 PDF를 원문으로 연결',
            type=['pdf'],
            key=f"interaction-{key}-paper-local-pdf-{index}",
            help='웹 원문 URL 대신 보유한 PDF를 이 리뷰의 full-text source로 사용할 수 있습니다.',
        )
        if local_pdf is not None and not active.get('local_pdf_name') == local_pdf.name:
            candidate = dict(item)
            candidate['paper_id'] = paper_id
            candidate['filename'] = local_pdf.name
            candidate['content'] = local_pdf.getvalue()
            try:
                attached = item_action(candidate, 'attach_local_pdf') if callable(item_action) else {}
            except Exception as error:
                st.error(f"로컬 PDF 연결에 실패했습니다: {error}")
            else:
                active['local_pdf_name'] = local_pdf.name
                active['local_pdf_path'] = str((attached or {}).get('pdf_path') or '')
                active['full_text_url'] = str((attached or {}).get('full_text_url') or active.get('full_text_url') or '')
                active['prompt'] = str((attached or {}).get('review_prompt') or active.get('prompt') or '')
                st.session_state[active_key] = active
                st.rerun()
        if active.get('local_pdf_name'):
            st.caption(f"Full-text source · local PDF · {active.get('local_pdf_name')}")
            local_path = str(active.get('local_pdf_path') or '')
            if local_path:
                from pathlib import Path
                path = Path(local_path).expanduser()
                if path.is_file():
                    st.download_button(
                        '연결된 PDF 열기 / 외부 LLM 첨부용',
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime='application/pdf',
                        key=f"interaction-{key}-paper-local-pdf-download-{index}",
                    )
        prompt = str(active.get('prompt') or '').strip()
        if prompt:
            st.caption('외부 LLM에 아래 Prompt를 전달하고 결과 JSON을 붙여넣으세요.')
            st.code(prompt, language=None)
        response = st.text_area(
            'LLM 응답 붙여넣기',
            value=str(active.get('response') or ''),
            height=260,
            key=f"interaction-{key}-paper-review-response-{index}",
        )
        if not active.get('review_result'):
            if st.button('논문 해석 · Knowledge Card 후보 생성', type='primary', key=f"interaction-{key}-paper-apply-review-{index}"):
                if not response.strip():
                    st.error('LLM 응답을 붙여넣어 주세요.')
                else:
                    enriched = dict(item)
                    enriched['paper_id'] = paper_id
                    enriched['full_text_url'] = str(active.get('full_text_url') or edited_full_text)
                    enriched['review_response'] = response
                    try:
                        result = item_action(enriched, 'review_response') if callable(item_action) else {}
                    except Exception as error:
                        st.error(f"논문 리뷰 결과를 반영하지 못했습니다: {error}")
                        return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})
                    active['response'] = response
                    active['review_result'] = dict(result or {})
                    st.session_state[active_key] = active
                    st.rerun()
        else:
            result = dict(active.get('review_result') or {})
            paper_summary = str(result.get('summary') or '').strip()
            review_note = str(result.get('review_note') or (result.get('reviewed') or {}).get('review_note') or '').strip()
            if paper_summary:
                st.markdown('#### Summary')
                st.write(paper_summary)
            if review_note:
                st.markdown('**Review note**')
                st.write(review_note)
            cards = list(result.get('knowledge_cards') or [])
            st.markdown(f"#### Knowledge Card 후보 · {len(cards)}건")
            if not cards:
                st.caption('생성된 Knowledge Card 후보가 없습니다.')
            for card_index, card in enumerate(cards, start=1):
                with st.container(border=True):
                    st.markdown(f"**{card_index}. {card.get('title') or 'Untitled'}**")
                    if card.get('claim'):
                        st.markdown(f"**Claim**  \n{card.get('claim')}")
                    if card.get('evidence_excerpt'):
                        st.markdown(f"**Evidence**  \n{card.get('evidence_excerpt')}")
                    if card.get('limits'):
                        st.markdown(f"**Limits**  \n{card.get('limits')}")
                    labels = list(card.get('labels') or [])
                    if labels:
                        st.caption('Labels · ' + ' · '.join(str(x) for x in labels))
            st.caption('이 후보들은 읽기 전용입니다. 승인·보류·거절은 Attention > Decisions에서 진행합니다.')
            if st.button('이 논문 검토 완료 · 다음', type='primary', key=f"interaction-{key}-paper-reviewed-next-{index}"):
                chosen = dict(item)
                chosen['paper_id'] = paper_id
                chosen['full_text_url'] = str(active.get('full_text_url') or edited_full_text)
                chosen['paper_summary'] = paper_summary
                chosen['review_note'] = review_note
                chosen['knowledge_candidates'] = list((result.get('reviewed') or {}).get('knowledge_candidates') or [])
                chosen['knowledge_request_ids'] = list(result.get('knowledge_request_ids') or [])
                selected.append(chosen)
                reviewed.append({'source_id': str(item.get('source_id') or ''), 'decision': 'preserve_and_review'})
                st.session_state[selected_key] = selected
                st.session_state[reviewed_key] = reviewed
                st.session_state[index_key] = index + 1
                st.session_state.pop(active_key, None)
                st.rerun()
        return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})

    col_keep, col_skip = st.columns(2)
    keep_label = '서재함 연결 · 논문 리뷰' if bool(item.get('already_in_library')) else '서재함에 보존 · 논문 리뷰'
    keep = col_keep.button(keep_label, type='primary', key=f"interaction-{key}-paper-keep-{index}")
    skip = col_skip.button('이번 라운드에서 제외 · 다음', key=f"interaction-{key}-paper-skip-{index}")
    if keep:
        updated = dict(item)
        updated['full_text_url'] = edited_full_text
        try:
            persisted = item_action(updated, 'preserve') if callable(item_action) else {}
        except Exception as error:
            st.error(f"서재함 연결에 실패했습니다: {error}")
            return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})
        paper = persisted.get('paper') if isinstance(persisted, Mapping) and isinstance(persisted.get('paper'), Mapping) else {}
        paper_id = str(paper.get('paper_id') or '')
        if not paper_id:
            st.error('서재함 논문 ID를 확인하지 못했습니다.')
            return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})
        prompt = str((persisted or {}).get('review_prompt') or '') if isinstance(persisted, Mapping) else ''
        st.session_state[active_key] = {'index': index, 'paper_id': paper_id, 'prompt': prompt, 'full_text_url': edited_full_text}
        st.rerun()
    if skip:
        reviewed.append({'source_id': str(item.get('source_id') or ''), 'decision': 'exclude'})
        st.session_state[reviewed_key] = reviewed
        st.session_state[index_key] = index + 1
        st.rerun()
    return InteractionRenderResult(interaction_id, False, {output_name: list(selected)})

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



def _render_knowledge_card_decision(
    st: Any,
    interaction_id: str,
    request: Mapping[str, Any],
    *,
    key: str,
    output_name: str,
) -> InteractionRenderResult:
    payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
    card = payload.get("card") if isinstance(payload.get("card"), Mapping) else {}
    request_id = str(request.get("phenomenon_id") or request.get("request_id") or "")

    st.markdown("**지식카드 후보 검토**")
    rq = str(payload.get("research_question") or "").strip()
    if rq:
        st.caption(f"Research Question · {rq}")

    review_note = str(payload.get("review_note") or "").strip()
    paper_id = str(payload.get("paper_id") or card.get("paper_id") or "").strip()
    rq_id = str(payload.get("research_question_id") or payload.get("rq_id") or "").strip()
    review_group = f"{paper_id}::{rq_id}" if (paper_id or rq_id) else ""
    review_note_seen_key = "knowledge-decision-review-note-seen-group"
    show_review_note = bool(
        review_note
        and (not review_group or st.session_state.get(review_note_seen_key) != review_group)
    )
    if show_review_note:
        st.markdown("**Review note**")
        st.write(review_note)

    title = str(card.get("title") or payload.get("title") or "지식카드 후보").strip()
    st.markdown(f"### {title}")

    claim = str(card.get("claim") or "").strip()
    if claim:
        st.markdown("**Claim**")
        st.write(claim)

    context = str(card.get("context") or "").strip()
    if context:
        st.markdown("**Context**")
        st.write(context)

    implication = str(card.get("implication") or "").strip()
    if implication:
        st.markdown("**Implication**")
        st.write(implication)

    evidence = str(card.get("evidence_excerpt") or card.get("source_excerpt") or "").strip()
    if evidence:
        st.markdown("**Evidence**")
        st.write(evidence)

    conditions = str(card.get("conditions") or "").strip()
    if conditions:
        st.markdown("**Conditions**")
        st.write(conditions)

    limits = str(card.get("limits") or "").strip()
    if limits:
        st.markdown("**Limits**")
        st.write(limits)

    labels = [str(x).strip() for x in (card.get("labels") or []) if str(x).strip()]
    if labels:
        st.caption("Labels · " + " · ".join(labels))

    provenance = card.get("provenance") if isinstance(card.get("provenance"), Mapping) else {}
    source_name = str(provenance.get("source_name") or "").strip()
    if source_name:
        st.caption(f"Source · {source_name}")

    note = st.text_area("연구자 의견 (선택)", key=f"interaction-{key}-comment", height=80)
    c1, c2, c3 = st.columns(3)
    approved = c1.button("지식카드 승인", type="primary", key=f"interaction-{key}-approve")
    deferred = c2.button("보완 요청", key=f"interaction-{key}-defer")
    rejected = c3.button("반려", key=f"interaction-{key}-reject")

    if approved:
        decision = "approved"
    elif deferred:
        decision = "deferred"
    elif rejected:
        decision = "rejected"
    else:
        decision = ""
    submitted = bool(decision and request_id)
    if submitted and review_group and show_review_note:
        # The note belongs to the paper x research-question review, not to each
        # knowledge card. Once the first card in the group is resolved, keep it
        # hidden while the researcher continues through sibling cards.
        st.session_state[review_note_seen_key] = review_group
    resolutions = [{"request_id": request_id, "decision": decision, "note": note.strip()}] if submitted else []
    return InteractionRenderResult(interaction_id, submitted, {output_name: resolutions})


def _render_approval(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    input_name = contract.required_inputs[0]
    output_name = contract.outputs[0]
    items = list(inputs.get(input_name) or [])
    if interaction_id == "resolve_pending_decision_requests" and len(items) == 1:
        request = items[0] if isinstance(items[0], Mapping) else {}
        payload = request.get("payload") if isinstance(request.get("payload"), Mapping) else {}
        card = payload.get("card") if isinstance(payload.get("card"), Mapping) else {}
        if str(request.get("subject_type") or "") == "knowledge_card" and card:
            return _render_knowledge_card_decision(st, interaction_id, request, key=key, output_name=output_name)
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
        proposal = dict(review.get("proposal") or {})
        st.markdown(f"**{renderer['title']}**")
        if renderer.get("help"):
            st.caption(str(renderer["help"]))
        if proposal.get("summary"):
            st.info(str(proposal.get("summary")))
        for item in proposal.get("assignments") or []:
            st.write(f"- Type assignment · `{item.get('card_id','')}` → **{item.get('type') or item.get('type_id') or '?'}**")
            if item.get("reason"):
                st.caption(str(item.get("reason")))
        for item in proposal.get("new_types") or []:
            st.write(f"- New Type · **{item.get('name','?')}**")
            if item.get("description"):
                st.caption(str(item.get("description")))
        for item in proposal.get("relations") or []:
            st.write(f"- Type relation · `{item.get('source_type') or item.get('source_type_id')}` → **{item.get('relation_name','?')}** → `{item.get('target_type') or item.get('target_type_id')}`")
        card_relations = [dict(item) for item in (proposal.get("card_relations") or []) if isinstance(item, dict)]
        if card_relations:
            st.write(f"- Knowledge Card relations · {len(card_relations)} relation(s)")
            with st.expander("지식카드 관계 보기", expanded=False):
                for item in card_relations:
                    st.write(
                        f"`{item.get('source_card_id','')}` → **{item.get('relation_type','?')}** → `{item.get('target_card_id','')}`"
                    )
                    if item.get("evidence"):
                        st.caption(str(item.get("evidence")))
        facet_updates = [item for item in (proposal.get("type_updates") or []) if str(item.get("facet") or "").strip()]
        if facet_updates:
            st.write(f"- Facet assignments · {len(facet_updates)} Type(s)")
            with st.expander("Facet 배정 보기", expanded=False):
                for item in facet_updates:
                    st.caption(f"{item.get('name') or item.get('type_id')} → {item.get('facet')}")
        for warning in proposal.get("warnings") or []:
            st.warning(str(warning))
        with st.form(f"interaction-{key}"):
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


def _render_external_llm(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    task = dict(inputs.get(contract.required_inputs[0]) or {})
    prompt = str(task.get("prompt") or "")
    with st.form(f"interaction-{key}"):
        st.markdown(f"**{renderer['title']}**")
        if renderer.get("help"):
            st.caption(str(renderer["help"]))
        stage = str(task.get("stage") or "")
        if stage:
            st.caption(f"Stage: `{stage}` · local auto-call disabled by execution policy")
        st.caption(str(renderer.get("prompt_label") or "Prompt") + " · 오른쪽 위 복사 버튼으로 전체 프롬프트를 복사할 수 있습니다.")
        # st.code intentionally replaces a disabled text area: Streamlit renders a
        # native copy-to-clipboard control while preserving the exact prompt text.
        st.code(prompt, language=None)
        response = st.text_area(str(renderer.get("response_label") or "Response"), value="", height=300, key=f"interaction-{key}-response", placeholder="외부 LLM의 전체 응답을 붙여 넣으세요.")
        submitted = st.form_submit_button(str(renderer.get("submit_label") or "Validate and continue"), type="primary")
    if submitted and not response.strip():
        st.info("외부 LLM 응답을 붙여 넣으세요.")
        submitted = False
    return InteractionRenderResult(interaction_id, bool(submitted), {contract.outputs[0]: response.strip()})

def render_interaction(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str = "default", item_action: Any | None = None) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id); binding = interaction_binding(interaction_id, profile)
    missing = [name for name in contract.required_inputs if name not in inputs]
    if missing: raise ValueError(f"Missing interaction inputs {missing}: {interaction_id}")
    if binding.renderer_type == "message": return _render_message(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "text_input": return _render_text_input(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "multi_select":
        if interaction_id == "review_literature_candidates":
            return _render_sequential_paper_review(st, interaction_id, inputs, key=key, profile=profile, item_action=item_action)
        return _render_multi_select(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "confirm": return _render_confirm(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "approval": return _render_approval(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "review_form": return _render_review_form(st, interaction_id, inputs, key=key, profile=profile)
    if binding.renderer_type == "external_llm": return _render_external_llm(st, interaction_id, inputs, key=key, profile=profile)
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
