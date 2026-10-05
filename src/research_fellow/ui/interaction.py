"""Streamlit realization of Interaction Contracts through UI Bindings."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from research_fellow.application.dsl.interaction import interaction_contract
from research_fellow.application.dsl.interaction_binding import interaction_binding
from research_fellow.application.literature_discovery_sources import google_scholar_url
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


def _citation_label(item: Mapping[str, Any]) -> str:
    count = item.get("citation_count")
    if count in (None, ""):
        return ""
    try:
        rendered = f"{int(float(count)):,}"
    except (TypeError, ValueError):
        rendered = str(count).strip()
    if not rendered:
        return ""
    source = str(item.get("citation_source") or "").strip()
    return f"Citations {rendered}" + (f" · {source}" if source else "")


def _paper_access_links(item: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return deduplicated researcher-facing access links for a discovery candidate."""
    abstract_url = str(item.get("abstract_url") or item.get("source_url") or item.get("url") or "").strip()
    full_text_url = str(item.get("full_text_url") or "").strip()
    pdf_url = str(item.get("pdf_url") or "").strip()
    landing_url = str(item.get("landing_url") or "").strip()
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, url: str) -> None:
        normalized = str(url or "").strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        specs.append((label, normalized))

    add("초록 / 서지", abstract_url)
    # When full_text_url is the same direct PDF, render one PDF button rather than
    # two aliases for the same resource.
    if full_text_url and full_text_url != pdf_url:
        add("원문", full_text_url)
    add("PDF", pdf_url or (full_text_url if full_text_url.lower().endswith(".pdf") else ""))
    add("논문 페이지", landing_url)
    add("Google Scholar", google_scholar_url(dict(item)))
    return specs
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
                citation = _citation_label(item)
                if citation:
                    st.caption(citation)
                links = _paper_access_links(item)
                if links:
                    st.markdown(" · ".join(f"[{label}]({url})" for label, url in links))
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



def _paper_review_prompt_stats(prompt: str) -> dict[str, int | bool]:
    """Return compact UI stats without changing the prompt payload."""
    text = str(prompt or "")
    start_marker = "--- PAPER FULL TEXT START ---"
    end_marker = "--- PAPER FULL TEXT END ---"
    full_text_chars = 0
    cursor = 0
    while True:
        start = text.find(start_marker, cursor)
        if start < 0:
            break
        content_start = start + len(start_marker)
        end = text.find(end_marker, content_start)
        if end < 0:
            break
        full_text_chars += len(text[content_start:end].strip())
        cursor = end + len(end_marker)
    return {
        "prompt_chars": len(text),
        "full_text_chars": full_text_chars,
        "has_full_text": full_text_chars > 0,
    }


def _render_paper_review_prompt_for_copy(st: Any, prompt: str, *, key: str) -> None:
    """Keep very long review prompts out of the main UI while preserving one-click copy."""
    stats = _paper_review_prompt_stats(prompt)
    source_status = "원문 포함" if stats["has_full_text"] else "원문 없음"
    st.caption(
        f"{source_status} · 원문 블록 {int(stats['full_text_chars']):,}자 · "
        f"전체 프롬프트 {int(stats['prompt_chars']):,}자"
    )
    with st.expander("논문 해석 프롬프트 · 펼쳐서 복사", expanded=False):
        st.caption("오른쪽 위 복사 버튼을 누르면 전체 프롬프트가 그대로 복사됩니다.")
        st.code(prompt, language=None)


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
    index = int(st.session_state.get(index_key, 0) or 0)
    selected = list(st.session_state.get(selected_key, []) or [])
    reviewed = list(st.session_state.get(reviewed_key, []) or [])
    active = dict(st.session_state.get(active_key, {}) or {})

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
    # While a paper review is active, pin the exact candidate that produced the
    # prompt. The workflow candidate list can be re-projected/reordered on a
    # Streamlit rerun, so `items[index]` is not a stable identity.
    active_candidate = active.get('candidate') if isinstance(active.get('candidate'), Mapping) else None
    if int(active.get('index', -1)) == index and active_candidate:
        item = dict(active_candidate)

    # Durable projection: reconstruct per-paper progress from the paper shelf and
    # RQ-specific analysis instead of treating Streamlit session_state as the
    # business source of truth.  A rerun/browser refresh therefore skips papers
    # already reviewed for this RQ and can resume a previously registered paper.
    durable_state: dict[str, Any] = {}
    if callable(item_action):
        try:
            projected = item_action(dict(item), 'review_state')
            if isinstance(projected, Mapping):
                durable_state = dict(projected)
        except Exception:
            durable_state = {}
    fresh_review_result_visible = (
        int(active.get('index', -1)) == index
        and isinstance(active.get('review_result'), Mapping)
        and bool(active.get('review_result'))
    )
    if durable_state.get('reviewed') and not fresh_review_result_visible:
        paper_id = str(durable_state.get('paper_id') or '')
        already_selected = any(str(x.get('paper_id') or '') == paper_id for x in selected if isinstance(x, Mapping))
        if not already_selected:
            chosen = dict(item)
            chosen['paper_id'] = paper_id
            chosen['paper_summary'] = str(durable_state.get('summary') or '')
            chosen['knowledge_request_ids'] = list(durable_state.get('knowledge_request_ids') or [])
            chosen['knowledge_candidates'] = list(durable_state.get('knowledge_cards') or [])
            selected.append(chosen)
            st.session_state[selected_key] = selected
        reviewed.append({'source_id': str(item.get('source_id') or ''), 'decision': 'reuse_existing_review'})
        st.session_state[reviewed_key] = reviewed
        st.session_state[index_key] = index + 1
        st.session_state.pop(active_key, None)
        st.rerun()

    if not active and durable_state.get('paper_id') and durable_state.get('linked_to_rq'):
        active = {
            'index': index,
            'candidate': dict(item),
            'paper_id': str(durable_state.get('paper_id') or ''),
            'prompt': str(durable_state.get('review_prompt') or ''),
            'full_text_url': str(item.get('full_text_url') or ''),
            'full_text_extraction_note': str(durable_state.get('full_text_extraction_note') or ''),
            'full_text_acquisition_error': '' if durable_state.get('full_text_ready') else '저장된 원문 텍스트가 없습니다.',
            'local_pdf_name': str(durable_state.get('full_text_source_name') or '') if str(durable_state.get('full_text_source_type') or '') == 'uploaded_pdf' else '',
        }
        st.session_state[active_key] = active

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
    citation = _citation_label(item)
    if citation:
        st.caption(citation)
    caution = str(item.get('caution') or '').strip()
    if caution:
        st.caption(f"주의 · {caution}")
    access_links = _paper_access_links(item)
    verified_links = [(label, url) for label, url in access_links if label != 'Google Scholar']
    if not verified_links:
        st.warning('검증된 논문 URL이 없습니다. Google Scholar에서 제목으로 찾아보세요.')
    if access_links:
        link_cols = st.columns(len(access_links))
        for col, (label, url) in zip(link_cols, access_links):
            col.link_button(label, url, use_container_width=True)

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
        extraction_note = str(active.get('full_text_extraction_note') or '').strip()
        acquisition_error = str(active.get('full_text_acquisition_error') or '').strip()
        if extraction_note:
            st.caption(f'Full-text source · {extraction_note}')
        if acquisition_error and not active.get('prompt'):
            st.warning(
                '원문 텍스트를 자동으로 확보하지 못했습니다. 원문 URL을 보정해 다시 추출하거나 로컬 PDF를 연결해 주세요.\n\n'
                + acquisition_error
            )
            if st.button('원문 URL에서 텍스트 다시 추출', key=f"interaction-{key}-paper-refresh-fulltext-{index}"):
                candidate = dict(item)
                candidate['paper_id'] = paper_id
                candidate['full_text_url'] = edited_full_text
                try:
                    refreshed = item_action(candidate, 'refresh_full_text') if callable(item_action) else {}
                except Exception as error:
                    st.error(f"원문 텍스트 추출에 실패했습니다: {error}")
                else:
                    active['full_text_url'] = str((refreshed or {}).get('full_text_url') or edited_full_text)
                    pinned = dict(active.get('candidate') or item)
                    pinned['full_text_url'] = active['full_text_url']
                    pinned['paper_id'] = paper_id
                    active['candidate'] = pinned
                    active['full_text_extraction_note'] = str((refreshed or {}).get('full_text_extraction_note') or '')
                    active['full_text_acquisition_error'] = ''
                    active['prompt'] = str((refreshed or {}).get('review_prompt') or '')
                    st.session_state[active_key] = active
                    st.success('원문 텍스트를 다시 추출하고 논문 해석 프롬프트를 갱신했습니다.')
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
                pinned = dict(active.get('candidate') or item)
                pinned['paper_id'] = paper_id
                active['candidate'] = pinned
                active['local_pdf_path'] = str((attached or {}).get('pdf_path') or '')
                active['full_text_url'] = str((attached or {}).get('full_text_url') or active.get('full_text_url') or '')
                active['prompt'] = str((attached or {}).get('review_prompt') or active.get('prompt') or '')
                active['full_text_extraction_note'] = str((attached or {}).get('full_text_extraction_note') or '')
                active['full_text_acquisition_error'] = ''
                st.session_state[active_key] = active
                st.success('PDF를 저장하고 원문 텍스트를 추출했습니다. 아래의 갱신된 프롬프트를 사용하세요.')
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
            st.caption('외부 LLM에 프롬프트를 그대로 붙여넣고 결과 JSON을 아래에 붙여넣으세요.')
            _render_paper_review_prompt_for_copy(st, prompt, key=f"interaction-{key}-paper-review-prompt-{index}")
        review_ready = bool(prompt)
        response = st.text_area(
            'LLM 응답 붙여넣기',
            value=str(active.get('response') or ''),
            height=260,
            key=f"interaction-{key}-paper-review-response-{index}",
            disabled=not review_ready,
        )
        if not active.get('review_result'):
            if st.button(
                '논문 해석 · Knowledge Card 후보 생성',
                type='primary',
                key=f"interaction-{key}-paper-apply-review-{index}",
                disabled=not review_ready,
            ):
                if not response.strip():
                    st.error('LLM 응답을 붙여넣어 주세요.')
                else:
                    # Apply the pasted response to the exact candidate that
                    # generated the prompt, never to a re-projected item that now
                    # happens to occupy the same list index.
                    enriched = dict(active.get('candidate') or item)
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
            if paper_summary:
                st.markdown('#### Summary')
                st.write(paper_summary)
            reviewed_result = dict(result.get('reviewed') or {})
            review_note = str(reviewed_result.get('review_note') or '').strip()
            if review_note:
                st.markdown('#### Review note')
                st.write(review_note)
            cards = list(result.get('knowledge_cards') or [])
            source_title = str((active.get('candidate') or item).get('title') or '').strip()
            source_year = str((active.get('candidate') or item).get('publication_year') or (active.get('candidate') or item).get('published') or '').strip()[:4]
            source_meta = ' · '.join(x for x in [source_title, source_year] if x)
            if source_meta:
                st.caption('출처 논문 · ' + source_meta)
            st.markdown(f"#### Knowledge Card 후보 · {len(cards)}건")
            if not cards:
                st.caption('생성된 Knowledge Card 후보가 없습니다.')
            for card_index, card in enumerate(cards, start=1):
                with st.container(border=True):
                    card_title = str(card.get('title') or '').strip()
                    if not card_title:
                        card_title = str(card.get('claim') or '').strip()[:72] or 'Untitled'
                    st.markdown(f"**{card_index}. {card_title}**")
                    if card.get('claim'):
                        st.markdown(f"**Claim**  \n{card.get('claim')}")
                    if card.get('evidence_excerpt') or card.get('source_excerpt') or card.get('source_quotes'):
                        _render_knowledge_evidence(st, card.get('evidence_excerpt') or card.get('source_excerpt'), card.get('source_quotes'))
                    if card.get('limits'):
                        st.markdown(f"**Limits**  \n{card.get('limits')}")
                    labels = list(card.get('labels') or [])
                    if labels:
                        st.caption('Labels · ' + ' · '.join(str(x) for x in labels))
            st.caption('이 후보들은 읽기 전용입니다. 승인·보류·거절은 Attention > Decisions에서 진행합니다.')
            if st.button('이 논문 검토 완료 · 다음', type='primary', key=f"interaction-{key}-paper-reviewed-next-{index}"):
                chosen = dict(active.get('candidate') or item)
                chosen['paper_id'] = paper_id
                chosen['full_text_url'] = str(active.get('full_text_url') or edited_full_text)
                chosen['paper_summary'] = paper_summary
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
        st.session_state[active_key] = {
            'index': index,
            'candidate': dict(updated),
            'paper_id': paper_id,
            'prompt': prompt,
            'full_text_url': edited_full_text,
            'full_text_extraction_note': str((persisted or {}).get('full_text_extraction_note') or ''),
            'full_text_acquisition_error': str((persisted or {}).get('full_text_acquisition_error') or ''),
        }
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



def _split_knowledge_evidence(value: object) -> tuple[str, list[str]]:
    text = str(value or "").strip()
    if not text:
        return "", []
    quote_marker = "원문 근거 문장\n"
    summary_marker = "\n\n근거 요약\n"
    if text.startswith(quote_marker) and summary_marker in text:
        quote_block, summary = text[len(quote_marker):].split(summary_marker, 1)
        quotes: list[str] = []
        for line in quote_block.splitlines():
            item = line.strip()
            if item.startswith("-"):
                item = item[1:].strip()
            if item and item not in quotes:
                quotes.append(item)
        return summary.strip(), quotes[:3]
    return text, []


def _render_knowledge_evidence(st: Any, value: object, source_quotes: object = None) -> None:
    evidence, legacy_quotes = _split_knowledge_evidence(value)
    quotes: list[str] = []
    for quote in source_quotes or []:
        text = str(quote or "").strip()
        if text and text not in quotes:
            quotes.append(text)
        if len(quotes) >= 3:
            break
    if not quotes:
        quotes = legacy_quotes
    if evidence:
        st.markdown("**Evidence**")
        st.write(evidence)
    if quotes:
        st.markdown("**Source quotes**")
        for quote in quotes:
            st.markdown(f"> {quote}")


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

    claim = str(card.get("claim") or "").strip()
    title = str(card.get("title") or "").strip()
    if not title:
        title = claim[:72].strip() or str(payload.get("title") or "지식카드 후보").strip()
    st.markdown("**Title**")
    st.write(title)

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

    evidence = card.get("evidence_excerpt") or card.get("source_excerpt")
    source_quotes = card.get("source_quotes")
    if evidence or source_quotes:
        _render_knowledge_evidence(st, evidence, source_quotes)

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
    source_year = str(provenance.get("publication_year") or payload.get("publication_year") or "").strip()[:4]
    if source_name or source_year:
        st.caption("Source · " + " · ".join(x for x in [source_name, source_year] if x))

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
    if variant == "research_answer_draft":
        context = dict(inputs.get("research_answer_review_context") or {})
        markdown_text = str(context.get("markdown") or "")
        current_question = str(context.get("question") or "")
        approved = bool(context.get("approved"))
        with st.form(f"interaction-{key}"):
            st.markdown(f"**{renderer['title']}**")
            if renderer.get("help"):
                st.caption(str(renderer["help"]))
            if approved:
                st.success("현재 저장된 최신 Draft는 연구자 승인 상태입니다. 내용을 다시 수정해 저장하면 재승인이 필요합니다.")
            else:
                st.info("Draft를 검토·수정한 뒤 승인하세요. 승인 후에만 다음 문헌 라운드를 시작할 수 있습니다.")
            edited = st.text_area(
                "Markdown Draft", value=markdown_text, height=520,
                key=f"interaction-{key}-markdown",
                help="LLM 출력 형식이 원하는 형태가 아니면 Markdown을 직접 수정하세요.",
            )
            next_question = st.text_area(
                "다음 라운드 연구질문", value=current_question, height=110,
                key=f"interaction-{key}-next-question",
                help="Round 2에서 질문을 더 구체화하거나 범위를 조정하려면 수정하세요. 승인 시 연구질문 버전으로 저장됩니다.",
            )
            note = st.text_input(
                "승인 메모 (선택)", value="", key=f"interaction-{key}-note",
            )
            c1, c2 = st.columns(2)
            saved = c1.form_submit_button("수정 저장", use_container_width=True)
            approved_now = c2.form_submit_button(
                str(renderer.get("submit_label") or "Draft 승인"), type="primary", use_container_width=True
            )
        action = "approve" if approved_now else "save" if saved else ""
        if action and not edited.strip():
            st.info("답변 Draft는 비어 있을 수 없습니다.")
            action = ""
        resolution = {
            "action": action,
            "markdown": edited.strip(),
            "next_question": next_question.strip(),
            "note": note.strip(),
        }
        return InteractionRenderResult(interaction_id, bool(action), {output_name: resolution})
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


def _render_external_llm(st: Any, interaction_id: str, inputs: Mapping[str, Any], *, key: str, profile: str, item_action: Any | None = None) -> InteractionRenderResult:
    contract = interaction_contract(interaction_id)
    binding = interaction_binding(interaction_id, profile)
    renderer = binding.renderer
    task = dict(inputs.get(contract.required_inputs[0]) or {})
    stage = str(task.get("stage") or "")
    prompt_state_key = f"interaction-{key}-prompt-override"
    pdf_name_key = f"interaction-{key}-local-pdf-name"
    pdf_path_key = f"interaction-{key}-local-pdf-path"
    prompt = str(st.session_state.get(prompt_state_key) or task.get("prompt") or "")

    # Durable follow-up paper reviews use the generic external-LLM interaction.
    # Restore the same local-PDF workflow used by inline paper review: attach the
    # PDF to the paper shelf, extract its text, and rebuild the current 1-page
    # RQ-conditioned review prompt before the researcher calls the external LLM.
    if stage == "single_paper_first_review" and callable(item_action):
        local_pdf = st.file_uploader(
            "로컬 PDF를 원문으로 연결",
            type=["pdf"],
            key=f"interaction-{key}-paper-local-pdf",
            help="보유한 PDF를 이 논문의 full-text source로 연결하고 현재 1-Page Review prompt를 다시 생성합니다.",
        )
        if local_pdf is not None and st.session_state.get(pdf_name_key) != local_pdf.name:
            candidate = {
                "paper_id": str(task.get("item_key") or ""),
                "filename": str(local_pdf.name or "paper.pdf"),
                "content": bytes(local_pdf.getvalue()),
            }
            try:
                attached = item_action(candidate, "attach_local_pdf") or {}
            except Exception as error:
                st.error(f"로컬 PDF 연결에 실패했습니다: {error}")
            else:
                st.session_state[pdf_name_key] = local_pdf.name
                st.session_state[pdf_path_key] = str(attached.get("pdf_path") or "")
                refreshed_prompt = str(attached.get("review_prompt") or "").strip()
                if refreshed_prompt:
                    st.session_state[prompt_state_key] = refreshed_prompt
                    prompt = refreshed_prompt
                st.success("PDF를 원문으로 연결하고 1-Page Review prompt를 갱신했습니다.")
        local_name = str(st.session_state.get(pdf_name_key) or "").strip()
        local_path = str(st.session_state.get(pdf_path_key) or "").strip()
        if local_name:
            st.caption(f"Full-text source · local PDF · {local_name}")
        if local_path:
            from pathlib import Path
            path = Path(local_path).expanduser()
            if path.is_file():
                st.download_button(
                    "연결된 PDF 열기 / 외부 LLM 첨부용",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                    key=f"interaction-{key}-paper-local-pdf-download",
                )

    with st.form(f"interaction-{key}"):
        st.markdown(f"**{renderer['title']}**")
        if renderer.get("help"):
            st.caption(str(renderer["help"]))
        if stage:
            st.caption(f"Stage: `{stage}` · local auto-call disabled by execution policy")
        if stage == "single_paper_first_review":
            st.caption("외부 LLM에 프롬프트를 그대로 붙여넣고 결과를 아래에 붙여넣으세요.")
            _render_paper_review_prompt_for_copy(st, prompt, key=f"interaction-{key}-paper-review-prompt")
        else:
            st.caption(str(renderer.get("prompt_label") or "Prompt") + " · 오른쪽 위 복사 버튼으로 전체 프롬프트를 복사할 수 있습니다.")
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
    if binding.renderer_type == "external_llm": return _render_external_llm(st, interaction_id, inputs, key=key, profile=profile, item_action=item_action)
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
