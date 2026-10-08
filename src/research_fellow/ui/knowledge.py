"""Streamlit renderer for the job-centred Knowledge Workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from research_fellow.application.ontology import ontology_dot, ontology_plotly_figure
from research_fellow.application.workspace_audit import audit_workspace
from research_fellow.workspace_portable import export_workspace_snapshot, check_workspace_consistency, workspace_bundle_dir


def _short(text: str, n: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _is_web_url(value: str) -> bool:
    value = str(value or "").strip().lower()
    return value.startswith("https://") or value.startswith("http://")


def _render_knowledge_state(st: Any, snapshot: Mapping[str, Any], *, english: bool) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("### " + ("Knowledge State" if english else "지식 상태"))
        cards = list(snapshot.get("cards") or [])
        if not cards:
            st.caption("No approved knowledge cards yet." if english else "승인된 지식카드가 없습니다.")
        for index, card in enumerate(cards[:30]):
            labels = [str(card.get("status") or "verified"), str(card.get("evidence_level") or "provisional")]
            if card.get("ontology_types"):
                labels.append(" / ".join(card.get("ontology_types")[:3]))
            st.markdown(f"**{_short(card.get('title', ''), 140)}**")
            st.caption(" · ".join(labels))
            if card.get("claim"):
                st.write(_short(card.get("claim", ""), 240))
            st.caption(
                ("supporting evidence: " if english else "추가 근거: ")
                + str(int(card.get("supporting_evidence_count") or 0))
                + " · "
                + ("relations: " if english else "관계: ")
                + str(int(card.get("relation_count") or 0))
            )
            if index < min(len(cards), 30) - 1:
                st.divider()

    with right:
        st.markdown("### " + ("Evidence & Change" if english else "근거와 변화"))
        evidence_levels = dict(snapshot.get("evidence_levels") or {})
        if evidence_levels:
            st.caption(" · ".join(f"{name}: {count}" for name, count in sorted(evidence_levels.items())))
        changes = list(snapshot.get("changes") or [])
        if not changes:
            st.caption("No recent knowledge changes." if english else "최근 지식 변화가 없습니다.")
        for index, item in enumerate(changes[:20]):
            marker = "NEW" if item.get("status") == "new" else "reviewed"
            st.markdown(f"**{marker}** — {_short(item.get('title', ''), 150)}")
            if item.get("detail"):
                st.caption(_short(item.get("detail", ""), 180))
            if index < min(len(changes), 20) - 1:
                st.divider()


def _render_paper_links(st: Any, paper: Mapping[str, Any], *, english: bool) -> None:
    abstract_url = str(paper.get("abstract_url") or paper.get("source_url") or "").strip()
    full_text_url = str(paper.get("full_text_url") or "").strip()
    pdf_url = str(paper.get("pdf_url") or "").strip()
    pdf_path = str(paper.get("pdf_path") or "").strip()
    buttons: list[tuple[str, str]] = []
    if _is_web_url(abstract_url):
        buttons.append(("Abstract / metadata" if english else "초록 / 서지", abstract_url))
    if _is_web_url(full_text_url):
        buttons.append(("Full text" if english else "원문 URL", full_text_url))
    if _is_web_url(pdf_url):
        buttons.append(("PDF", pdf_url))
    if buttons:
        cols = st.columns(len(buttons))
        for col, (label, url) in zip(cols, buttons):
            col.link_button(label, url, use_container_width=True)
    if pdf_path and not _is_web_url(pdf_path):
        path = Path(pdf_path).expanduser()
        if path.is_file():
            try:
                st.download_button(
                    "Open local PDF copy" if english else "로컬 PDF 원문 열기",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                    key=f"paper-pdf-{paper.get('paper_id')}",
                    use_container_width=True,
                )
            except OSError:
                pass
        else:
            st.caption("Stored PDF path is no longer available on this machine." if english else "저장된 PDF 경로를 현재 머신에서 찾을 수 없습니다.")


def _render_evidence_library(
    st: Any,
    snapshot: Mapping[str, Any],
    *,
    english: bool,
    update_paper_metadata: Callable[[str, list[str] | str, str, str | None], Mapping[str, Any]] | None,
    attach_paper_pdf: Callable[[str, str, bytes], Mapping[str, Any]] | None,
) -> None:
    st.markdown("### " + ("Evidence Library" if english else "서재함 · 근거 논문"))
    papers = list(snapshot.get("evidence_library") or [])
    if not papers:
        st.caption("No papers are currently preserved in the evidence library." if english else "현재 서재함에 보존된 논문이 없습니다.")
        return

    counts = dict(snapshot.get("counts") or {})
    pcols = st.columns(3)
    pcols[0].metric("Papers" if english else "보존 논문", int(counts.get("papers", len(papers))))
    pcols[1].metric("Analyzed" if english else "분석 완료", int(counts.get("analyzed_papers", 0)))
    pcols[2].metric("Linked to knowledge" if english else "지식 연결 논문", int(counts.get("papers_with_knowledge", 0)))
    st.caption(
        "Papers are the researcher-facing evidence and citation units; knowledge cards remain the agent's internal semantic units."
        if english else
        "논문은 연구자가 검토·인용하는 근거 단위이고, 지식카드는 Agent 내부의 구조화된 의미 단위입니다."
    )

    query = st.text_input(
        "Search papers" if english else "논문 검색",
        key="evidence-library-search",
        placeholder="title, author, label, comment" if english else "제목, 저자, 레이블, 코멘트",
    ).strip().casefold()
    if query:
        filtered = []
        for paper in papers:
            interpretations = list(paper.get("question_interpretations") or [])
            haystack = " ".join([
                str(paper.get("title") or ""),
                " ".join(str(x) for x in (paper.get("authors") or [])),
                " ".join(str(x) for x in (paper.get("labels") or [])),
                str(paper.get("researcher_note") or ""),
                str(paper.get("summary") or ""),
                " ".join(
                    " ".join([str(x.get("question") or ""), str(x.get("summary") or "")])
                    for x in interpretations if isinstance(x, Mapping)
                ),
            ]).casefold()
            if query in haystack:
                filtered.append(paper)
        papers = filtered

    page_size = 10
    total_pages = max(1, (len(papers) + page_size - 1) // page_size)
    page = st.selectbox(
        "Page" if english else "페이지",
        list(range(1, total_pages + 1)),
        key="evidence-library-page",
        disabled=total_pages <= 1,
    )
    start = (int(page) - 1) * page_size
    visible = papers[start:start + page_size]
    st.caption(
        (f"Showing {start + 1}-{min(start + page_size, len(papers))} of {len(papers)} papers" if english else
         f"논문 {len(papers)}편 중 {start + 1}-{min(start + page_size, len(papers))}편 표시")
        if papers else ("No matching papers." if english else "검색 조건에 맞는 논문이 없습니다.")
    )

    for index, paper in enumerate(visible):
        paper_id = str(paper.get("paper_id") or "")
        status = f"{paper.get('shelf_status') or 'reference'} · {paper.get('reading_status') or 'unread'}"
        if paper.get("has_analysis"):
            status += " · analyzed" if english else " · 분석됨"
        st.markdown(f"**{_short(paper.get('title', ''), 190)}**")
        meta = []
        if paper.get("publication_year"):
            meta.append(str(paper.get("publication_year")))
        authors = list(paper.get("authors") or [])
        if authors:
            meta.append(", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""))
        if meta:
            st.caption(" · ".join(meta))
        st.caption(status + " · " + (("knowledge claims: " if english else "연결 지식: ") + str(int(paper.get("knowledge_card_count") or 0))))

        _render_paper_links(st, paper, english=english)

        summary = str(paper.get("summary") or "").strip()
        if summary:
            source_label = "abstract / general summary" if paper.get("summary_source") == "abstract" else "general summary"
            if not english:
                source_label = "초록·대표 요약" if paper.get("summary_source") == "abstract" else "대표 요약"
            st.caption("Paper summary" if english else "논문 요약")
            st.write(summary)
            st.caption(source_label)

        paper_labels = list(paper.get("labels") or [])
        if paper_labels:
            st.caption(("Labels: " if english else "레이블: ") + " · ".join(paper_labels))
        researcher_note = str(paper.get("researcher_note") or "").strip()
        if researcher_note:
            st.caption("Researcher comment" if english else "연구자 코멘트")
            st.write(_short(researcher_note, 520))

        interpretations = [dict(x) for x in (paper.get("question_interpretations") or []) if isinstance(x, Mapping)]
        if interpretations:
            st.markdown("**Used in Research Questions**" if english else "**사용된 연구질문**")
            for interpretation in interpretations:
                rq_id = str(interpretation.get("rq_id") or "legacy")
                question = str(interpretation.get("question") or ("Unscoped review" if english else "연구질문 미지정 리뷰"))
                pending = list(interpretation.get("pending_knowledge_cards") or [])
                approved = list(interpretation.get("approved_knowledge_cards") or [])
                suffix_parts = []
                if pending:
                    suffix_parts.append(("pending" if english else "승인대기") + f" {len(pending)}")
                if approved:
                    suffix_parts.append(("approved" if english else "승인") + f" {len(approved)}")
                expander_title = _short(question, 150)
                if suffix_parts:
                    expander_title += " · " + " / ".join(suffix_parts)
                with st.expander(expander_title, expanded=False):
                    comment = str(interpretation.get("researcher_comment") or "").strip()
                    if comment:
                        st.caption("Researcher comment" if english else "연구자 코멘트")
                        st.write(comment)
                    rq_summary = str(interpretation.get("summary") or "").strip()
                    if rq_summary:
                        st.caption("Executive 1-Page Summary" if english else "연구질문 관점 1-Page Summary")
                        st.markdown(rq_summary)
                    review_note = str(interpretation.get("review_note") or "").strip()
                    if review_note:
                        st.caption("Review note" if english else "리뷰 메모")
                        st.write(review_note)
                    review = str(interpretation.get("review") or "").strip()
                    if review and review != rq_summary:
                        st.caption("Detailed review" if english else "상세 논문 해석")
                        st.write(review)
                    if pending:
                        st.caption("Pending Knowledge Cards" if english else "승인 대기 Knowledge Cards")
                        for card in pending:
                            st.markdown(f"- **{_short(card.get('title', ''), 120)}** — {_short(card.get('claim', ''), 280)}")
                    if approved:
                        st.caption("Approved Knowledge Cards" if english else "승인된 Knowledge Cards")
                        for card in approved:
                            st.markdown(f"- **{_short(card.get('title', ''), 120)}** — {_short(card.get('claim', ''), 280)}")
                    if not rq_summary and not review and not pending and not approved:
                        st.caption("Linked to this research question; no paper review has been completed yet." if english else "이 연구질문과 연결되어 있으나 아직 논문 리뷰가 완료되지 않았습니다.")

        origins = [dict(x) for x in (paper.get("origin_links") or []) if isinstance(x, Mapping)]
        rq_origins = [str(x.get("research_question") or "").strip() for x in origins if str(x.get("origin_type") or "") == "researcher_question"]
        rq_origins = list(dict.fromkeys(x for x in rq_origins if x))
        if rq_origins:
            st.caption(("Research questions: " if english else "연결 연구질문: ") + " · ".join(rq_origins[:5]))
        elif origins:
            origin_labels = [str(item.get("label") or item.get("research_title") or "").strip() for item in origins]
            origin_labels = [item for item in origin_labels if item]
            if origin_labels:
                st.caption(("Research origin: " if english else "연구 출처: ") + " · ".join(origin_labels[:3]))

        if (update_paper_metadata is not None or attach_paper_pdf is not None) and paper_id:
            with st.expander("Annotate paper" if english else "논문 코멘트 · 레이블", expanded=False):
                labels_text = st.text_input(
                    "Labels" if english else "레이블",
                    value=", ".join(paper_labels),
                    key=f"paper-library-labels-{paper_id}",
                    help="Comma-separated labels" if english else "쉼표로 구분해 입력합니다.",
                )
                full_text_text = st.text_input(
                    "Full-text URL" if english else "원문 URL",
                    value=str(paper.get("full_text_url") or ""),
                    key=f"paper-library-fulltext-{paper_id}",
                    help="Correct or add the readable full-text URL." if english else "원문 HTML/공식 본문 URL을 직접 추가하거나 수정할 수 있습니다.",
                )
                local_pdf = st.file_uploader(
                    "Attach local PDF" if english else "로컬 PDF 원문 연결",
                    type=["pdf"],
                    key=f"paper-library-local-pdf-{paper_id}",
                    help=("Keep the web full-text URL and attach a researcher-provided local PDF as the readable original." if english else "원문 URL은 유지하고, 연구자가 보유한 PDF를 이 논문의 로컬 원문으로 연결합니다."),
                )
                if local_pdf is not None and attach_paper_pdf is not None:
                    if st.button(
                        "Attach PDF" if english else "PDF 연결",
                        key=f"paper-library-attach-pdf-{paper_id}",
                        use_container_width=True,
                    ):
                        try:
                            attach_paper_pdf(paper_id, str(local_pdf.name), bytes(local_pdf.getvalue()))
                            st.success("Local PDF attached." if english else "로컬 PDF 원문을 연결했습니다.")
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                note_text = st.text_area(
                    "Researcher comment" if english else "연구자 코멘트",
                    value=researcher_note,
                    height=110,
                    key=f"paper-library-note-{paper_id}",
                )
                if update_paper_metadata is not None and st.button(
                    "Save annotation" if english else "코멘트·레이블 저장",
                    key=f"paper-library-save-{paper_id}",
                    use_container_width=True,
                ):
                    try:
                        update_paper_metadata(paper_id, labels_text, note_text, full_text_text)
                        st.success("Saved." if english else "저장했습니다.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        if index < len(visible) - 1:
            st.divider()


def _render_ontology_editor(
    st: Any,
    ontology: Mapping[str, Any],
    *,
    english: bool,
    manage_ontology: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None,
) -> None:
    facets = [dict(x) for x in (ontology.get("facet_rows") or [])]
    types = [dict(x) for x in (ontology.get("type_rows") or [])]
    relations = [dict(x) for x in (ontology.get("relation_rows") or [])]
    cards_by_type = {
        str(type_id): [dict(card) for card in (cards or []) if isinstance(card, Mapping)]
        for type_id, cards in dict(ontology.get("cards_by_type") or {}).items()
    }

    st.markdown("### " + ("Ontology" if english else "온톨로지"))
    st.caption(
        "Approved Type/Relation graph. LLM proposals are reviewed in Attention; this area is for inspection and researcher correction."
        if english else
        "승인된 Type/Relation 그래프입니다. LLM 제안은 Attention에서 검토하고, 여기서는 현재 온톨로지를 확인하고 연구자가 직접 보정합니다."
    )

    if types:
        figure = ontology_plotly_figure(types, relations, facets)
        if figure is not None:
            try:
                figure.update_layout(height=720, margin=dict(l=20, r=20, t=30, b=20))
            except Exception:
                pass
            st.caption(
                "Use the mouse wheel or trackpad to zoom, drag to pan, and the Plotly toolbar to zoom/reset."
                if english else
                "마우스 휠·트랙패드로 확대/축소하고, 드래그로 이동할 수 있습니다. 오른쪽 위 Plotly 도구막대에서도 확대·초기화를 할 수 있습니다."
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={
                    "scrollZoom": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "responsive": True,
                    "doubleClick": "reset",
                },
            )
        else:
            st.graphviz_chart(ontology_dot(types, relations, facets), use_container_width=True)
    else:
        st.info("No ontology Types yet." if english else "아직 승인된 Ontology Type이 없습니다.")

    if manage_ontology is None:
        return

    type_tab, relation_tab, facet_tab = st.tabs([
        "Types" if english else "Type 편집",
        "Relations" if english else "Type 관계 편집",
        "Facets" if english else "Facet 편집",
    ])
    facet_name_by_id = {str(x.get("facet_id") or ""): str(x.get("name") or "") for x in facets}
    type_name_by_id = {str(x.get("type_id") or ""): str(x.get("name") or "") for x in types}

    with type_tab:
        with st.expander("Create Type" if english else "새 Type 추가", expanded=False):
            name = st.text_input("Name" if english else "Type 이름", key="ontology-create-type-name")
            desc = st.text_area("Description" if english else "설명", key="ontology-create-type-desc", height=90)
            facet_options = ["", *facet_name_by_id.keys()]
            facet_id = st.selectbox(
                "Facet" if english else "Facet", facet_options, key="ontology-create-type-facet",
                format_func=lambda value: ("None" if english else "미지정") if not value else facet_name_by_id.get(value, value),
            )
            if st.button("Create" if english else "Type 추가", key="ontology-create-type", disabled=not name.strip()):
                try:
                    manage_ontology("create_type", {"name": name, "description": desc, "facet_id": facet_id or None})
                    st.success("Created." if english else "Type을 추가했습니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        if types:
            selected_type = st.selectbox(
                "Type" if english else "편집할 Type",
                [str(x.get("type_id") or "") for x in types],
                format_func=lambda value: f"{facet_name_by_id.get(str(next((x.get('facet_id') for x in types if str(x.get('type_id')) == value), '') or ''), 'Facet 미지정')} · {type_name_by_id.get(value, value)}",
                key="ontology-edit-type-id",
            )
            current = next((x for x in types if str(x.get("type_id") or "") == selected_type), {})
            selected_cards = list(cards_by_type.get(selected_type) or [])
            st.markdown("**Knowledge Cards in this Type**" if english else "**이 Type에 속한 Knowledge Cards**")
            st.caption(
                (f"{len(selected_cards)} approved knowledge card(s)" if english else f"승인된 지식카드 {len(selected_cards)}건")
            )
            if selected_cards:
                for card_index, card in enumerate(selected_cards):
                    st.markdown(f"**{_short(card.get('title', ''), 150)}**")
                    if card.get("claim"):
                        st.write(_short(card.get("claim", ""), 360))
                    meta = [str(card.get("status") or "verified"), str(card.get("evidence_level") or "provisional")]
                    meta.append(("evidence" if english else "근거") + f" {int(card.get('supporting_evidence_count') or 0)}")
                    meta.append(("relations" if english else "관계") + f" {int(card.get('relation_count') or 0)}")
                    st.caption(" · ".join(meta))
                    if card_index < len(selected_cards) - 1:
                        st.divider()
            else:
                st.caption("No approved knowledge cards are assigned to this Type." if english else "이 Type에 배정된 승인 지식카드가 없습니다.")

            st.divider()
            edit_name = st.text_input("Name" if english else "Type 이름", value=str(current.get("name") or ""), key=f"ontology-edit-type-name-{selected_type}")
            edit_desc = st.text_area("Description" if english else "설명", value=str(current.get("description") or ""), height=90, key=f"ontology-edit-type-desc-{selected_type}")
            facet_options = ["", *facet_name_by_id.keys()]
            current_facet = str(current.get("facet_id") or "")
            edit_facet = st.selectbox(
                "Facet", facet_options, index=facet_options.index(current_facet) if current_facet in facet_options else 0,
                key=f"ontology-edit-type-facet-{selected_type}",
                format_func=lambda value: ("None" if english else "미지정") if not value else facet_name_by_id.get(value, value),
            )
            c1, c2 = st.columns(2)
            if c1.button("Save" if english else "Type 변경 저장", key=f"ontology-save-type-{selected_type}", use_container_width=True):
                try:
                    manage_ontology("update_type", {"type_id": selected_type, "name": edit_name, "description": edit_desc, "facet_id": edit_facet or None})
                    st.success("Saved." if english else "Type 변경을 반영했습니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if c2.button("Delete" if english else "Type 삭제", key=f"ontology-delete-type-{selected_type}", use_container_width=True):
                try:
                    manage_ontology("delete_type", {"type_id": selected_type})
                    st.warning("Deleted." if english else "Type을 삭제했습니다. 연결된 카드 배정과 관계도 함께 정리됩니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with relation_tab:
        if len(types) >= 2:
            with st.expander("Create relation" if english else "새 Type 관계 추가", expanded=False):
                source = st.selectbox("Source" if english else "출발 Type", list(type_name_by_id), format_func=lambda x: type_name_by_id.get(x, x), key="ontology-create-rel-source")
                target = st.selectbox("Target" if english else "도착 Type", list(type_name_by_id), format_func=lambda x: type_name_by_id.get(x, x), key="ontology-create-rel-target")
                rel_name = st.text_input("Relation" if english else "관계 이름", key="ontology-create-rel-name")
                rel_desc = st.text_area("Description" if english else "관계 설명", key="ontology-create-rel-desc", height=80)
                if st.button("Create relation" if english else "관계 추가", key="ontology-create-rel", disabled=not rel_name.strip() or source == target):
                    try:
                        manage_ontology("create_relation", {"source_type_id": source, "target_type_id": target, "relation_name": rel_name, "description": rel_desc})
                        st.success("Created." if english else "Type 관계를 추가했습니다.")
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        if relations:
            relation_ids = [str(x.get("relation_id") or "") for x in relations]
            selected_rel = st.selectbox(
                "Relation" if english else "편집할 관계", relation_ids,
                format_func=lambda rid: next((f"{type_name_by_id.get(str(x.get('source_type_id')), str(x.get('source_type_id')))} → {type_name_by_id.get(str(x.get('target_type_id')), str(x.get('target_type_id')))} · {x.get('relation_name')}" for x in relations if str(x.get('relation_id')) == rid), rid),
                key="ontology-edit-rel-id",
            )
            current = next((x for x in relations if str(x.get("relation_id") or "") == selected_rel), {})
            ids = list(type_name_by_id)
            current_source = str(current.get("source_type_id") or "")
            current_target = str(current.get("target_type_id") or "")
            source = st.selectbox("Source" if english else "출발 Type", ids, index=ids.index(current_source) if current_source in ids else 0, format_func=lambda x: type_name_by_id.get(x, x), key=f"ontology-edit-rel-source-{selected_rel}")
            target = st.selectbox("Target" if english else "도착 Type", ids, index=ids.index(current_target) if current_target in ids else min(1, len(ids)-1), format_func=lambda x: type_name_by_id.get(x, x), key=f"ontology-edit-rel-target-{selected_rel}")
            rel_name = st.text_input("Relation" if english else "관계 이름", value=str(current.get("relation_name") or ""), key=f"ontology-edit-rel-name-{selected_rel}")
            rel_desc = st.text_area("Description" if english else "관계 설명", value=str(current.get("description") or ""), height=80, key=f"ontology-edit-rel-desc-{selected_rel}")
            c1, c2 = st.columns(2)
            if c1.button("Save" if english else "관계 변경 저장", key=f"ontology-save-rel-{selected_rel}", use_container_width=True):
                try:
                    manage_ontology("update_relation", {"relation_id": selected_rel, "source_type_id": source, "target_type_id": target, "relation_name": rel_name, "description": rel_desc})
                    st.success("Saved." if english else "관계 변경을 반영했습니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if c2.button("Delete" if english else "관계 삭제", key=f"ontology-delete-rel-{selected_rel}", use_container_width=True):
                try:
                    manage_ontology("delete_relation", {"relation_id": selected_rel})
                    st.warning("Deleted." if english else "관계를 삭제했습니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        elif types:
            st.caption("No Type relations yet." if english else "아직 Type 관계가 없습니다.")

    with facet_tab:
        st.caption(
            "Facets are higher-level groupings of Types. LLM facet proposal remains a separate graph-level Attention task."
            if english else
            "Facet은 Type을 상위 관점에서 묶는 분류입니다. LLM Facet 제안은 기존처럼 별도의 graph-level Attention에서 처리합니다."
        )
        with st.expander("Create Facet" if english else "새 Facet 추가", expanded=False):
            facet_name = st.text_input("Name" if english else "Facet 이름", key="ontology-create-facet-name")
            facet_desc = st.text_area("Description" if english else "설명", key="ontology-create-facet-desc", height=80)
            if st.button("Create Facet" if english else "Facet 추가", key="ontology-create-facet", disabled=not facet_name.strip()):
                try:
                    manage_ontology("create_facet", {"name": facet_name, "description": facet_desc})
                    st.success("Created." if english else "Facet을 추가했습니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        for facet in facets:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{facet.get('name')}** · {int(facet.get('type_count') or 0)} Types")
            if facet.get("description"):
                c1.caption(str(facet.get("description")))
            if c2.button("Delete" if english else "삭제", key=f"ontology-delete-facet-{facet.get('facet_id')}"):
                try:
                    manage_ontology("delete_facet", {"facet_id": str(facet.get("facet_id") or "")})
                    st.warning("Deleted." if english else "Facet을 삭제했습니다. 소속 Type은 미지정 상태로 유지됩니다.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _render_structure_and_gaps(
    st: Any, snapshot: Mapping[str, Any], *, english: bool,
    enqueue_ontology_work: Callable[[], Mapping[str, Any]] | None = None,
    manage_ontology: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    st.markdown("### " + ("Knowledge Structure" if english else "지식 구조"))
    ontology = dict(snapshot.get("ontology") or {})
    relations = dict(snapshot.get("relations") or {})
    structure_cols = st.columns(5)
    structure_cols[0].metric("Ontology types" if english else "온톨로지 타입", int(ontology.get("types", 0)))
    structure_cols[1].metric("Type relations" if english else "타입 관계", int(ontology.get("relations", 0)))
    structure_cols[2].metric("Card relations" if english else "카드 관계", int(relations.get("total", 0)))
    structure_cols[3].metric("Contradictions" if english else "상충 관계", int(relations.get("contradictions", 0)))
    structure_cols[4].metric("Pending structure review" if english else "구조 검토 대기", int(ontology.get("pending_reviews", 0)))

    top_types = list(ontology.get("top_types") or [])
    if top_types:
        st.caption("Most populated ontology types" if english else "주요 온톨로지 타입")
        st.write(" · ".join(
            f"{item.get('name') or item.get('type_id')} ({int(item.get('card_count') or 0)})"
            for item in top_types[:8]
        ))

    _render_ontology_editor(st, ontology, english=english, manage_ontology=manage_ontology)
    if enqueue_ontology_work is not None:
        if st.button(
            "Prepare pending ontology proposals" if english else "Ontology 제안 작업 준비",
            use_container_width=True,
        ):
            try:
                result = dict(enqueue_ontology_work() or {})
                created = len(result.get("created_task_ids") or [])
                if created:
                    st.success(
                        (
                            f"Created {created} external-LLM Ontology task(s). Open Attention > Inputs to copy the prompt and paste the LLM response."
                            if english else
                            f"외부 LLM Ontology 작업 {created}건을 준비했습니다. Attention > Inputs에서 프롬프트를 복사하고 LLM 응답을 붙여넣으세요."
                        )
                    )
                else:
                    st.info(
                        "An Ontology proposal task is already pending in Attention > Inputs."
                        if english else
                        "이미 처리 대기 중인 Ontology 제안 작업이 있습니다. Attention > Inputs에서 계속 진행하세요."
                    )
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("### " + ("Knowledge Gaps & Structuring Needs" if english else "지식 과제와 구조화 필요"))
    gaps = list(snapshot.get("gaps") or [])
    if not gaps:
        st.success("No immediate knowledge-structure gap requires attention." if english else "즉시 처리할 지식 구조 과제가 없습니다.")
    else:
        for index, item in enumerate(gaps):
            st.markdown(f"**{str(item.get('priority') or 'medium').upper()} · {_short(item.get('title', ''), 180)}**")
            st.caption(_short(item.get("reason", ""), 280))
            if index < len(gaps) - 1:
                st.divider()



def _render_workspace_audit(st: Any, snapshot: Mapping[str, Any], *, english: bool) -> None:
    storage = dict(snapshot.get("workspace_storage") or {})
    db_path = str(storage.get("db_path") or "").strip()
    data_dir = str(storage.get("data_dir") or "").strip()
    st.markdown("### " + ("Workspace Storage Audit" if english else "Workspace 저장소 진단"))
    st.caption(
        "Read-only diagnostics. This does not delete, migrate, merge, or modify any workspace data."
        if english else
        "읽기 전용 진단입니다. 데이터 삭제·마이그레이션·병합·수정은 수행하지 않습니다."
    )
    if not db_path:
        st.warning("Workspace database path is unavailable." if english else "현재 Workspace DB 경로를 확인할 수 없습니다.")
        return

    # Portable workspace snapshot: workspace.json + pdf/.  SQLite is a working
    # cache that can be rebuilt from this pair when it is missing.
    from pathlib import Path as _Path
    db_file = _Path(db_path)
    if db_file.name == "research_fellow.db":
        workspace_key = "general"
    elif db_file.name.startswith("research_fellow_") and db_file.suffix == ".db":
        workspace_key = db_file.stem[len("research_fellow_"):]
    else:
        workspace_key = db_file.stem
    bundle = workspace_bundle_dir(data_dir or db_file.parent, workspace_key)
    st.markdown("#### " + ("Portable Workspace" if english else "Portable Workspace · JSON + PDF"))
    st.caption(
        ("workspace.json and pdf/ are the portable durable snapshot. SQLite is the fast working cache and can be rebuilt when absent."
         if english else
         "workspace.json과 pdf/를 휴대 가능한 저장본으로 사용합니다. SQLite DB는 빠른 작업용이며 없어도 JSON+PDF에서 복원할 수 있습니다.")
    )
    st.code(str(bundle), language=None)
    pcols = st.columns(2)
    if pcols[0].button(
        "Refresh JSON + PDF snapshot" if english else "JSON + PDF 저장본 갱신",
        type="primary", use_container_width=True, key=f"workspace-portable-export::{db_path}",
    ):
        try:
            result = export_workspace_snapshot(db_path, data_dir=data_dir or None, workspace_key=workspace_key)
            st.session_state[f"workspace-portable-export-result::{db_path}"] = result.as_dict()
            st.session_state[f"workspace-portable-consistency::{db_path}"] = check_workspace_consistency(
                db_path, data_dir=data_dir or None, workspace_key=workspace_key
            )
        except Exception as exc:
            st.error(str(exc))
    if pcols[1].button(
        "Check DB ↔ JSON/PDF" if english else "DB ↔ JSON/PDF 일치성 검사",
        use_container_width=True, key=f"workspace-portable-check::{db_path}",
    ):
        try:
            st.session_state[f"workspace-portable-consistency::{db_path}"] = check_workspace_consistency(
                db_path, data_dir=data_dir or None, workspace_key=workspace_key
            )
        except Exception as exc:
            st.error(str(exc))

    exported = st.session_state.get(f"workspace-portable-export-result::{db_path}")
    if isinstance(exported, Mapping):
        mb = float(exported.get("pdf_bytes") or 0) / (1024 * 1024)
        st.success(
            (f"Portable snapshot updated · {exported.get('row_count', 0)} records · "
             f"{exported.get('pdf_count', 0)} PDFs ({mb:.1f} MB). Keep this folder to rebuild the DB."
             if english else
             f"Portable 저장본 갱신 완료 · {exported.get('row_count', 0)}개 레코드 · "
             f"PDF {exported.get('pdf_count', 0)}개 ({mb:.1f} MB). 이 폴더만 보관하면 DB를 재구성할 수 있습니다.")
        )
        if int(exported.get("missing_pdf_sources") or 0):
            st.warning(
                (f"{exported.get('missing_pdf_sources')} DB PDF reference(s) could not be copied."
                 if english else
                 f"DB가 참조하지만 복사할 수 없는 PDF가 {exported.get('missing_pdf_sources')}건 있습니다.")
            )

    consistency = st.session_state.get(f"workspace-portable-consistency::{db_path}")
    if isinstance(consistency, Mapping):
        if consistency.get("status") == "missing_snapshot":
            st.info("Create the JSON+PDF snapshot first." if english else "먼저 JSON+PDF 저장본을 생성하세요.")
        elif consistency.get("status") == "missing_db":
            st.info("The DB is absent; it can be rebuilt from workspace.json on next load." if english else "DB가 없습니다. 다음 로딩 때 workspace.json에서 재구성할 수 있습니다.")
        else:
            ccols = st.columns(5)
            ccols[0].metric("DB only" if english else "DB만", int(consistency.get("db_only") or 0))
            ccols[1].metric("JSON only" if english else "JSON만", int(consistency.get("json_only") or 0))
            ccols[2].metric("Changed" if english else "내용 다름", int(consistency.get("changed") or 0))
            ccols[3].metric("Missing PDF" if english else "PDF 누락", int(consistency.get("pdf_missing") or 0))
            ccols[4].metric("Hash mismatch" if english else "PDF Hash 불일치", int(consistency.get("pdf_hash_mismatch") or 0))
            if consistency.get("consistent"):
                st.success("DB, workspace.json, and referenced PDFs are consistent." if english else "DB · workspace.json · 참조 PDF가 일치합니다.")
            else:
                st.warning(
                    "The working DB differs from the portable snapshot. Refresh the snapshot after confirming the DB is the desired current state."
                    if english else
                    "현재 DB와 portable 저장본이 다릅니다. DB가 원하는 최신 상태인지 확인한 뒤 JSON+PDF 저장본을 갱신하세요."
                )
            differing = [row for row in (consistency.get("tables") or []) if any(int(row.get(k) or 0) for k in ("db_only", "json_only", "changed"))]
            if differing:
                with st.expander("Table differences" if english else "테이블별 차이", expanded=False):
                    st.dataframe(differing, use_container_width=True, hide_index=True)

    st.divider()
    cache_key = f"workspace-audit-report::{db_path}"
    if st.button("Run audit" if english else "현재 Workspace 진단 실행", type="primary", key=f"workspace-audit-run::{db_path}"):
        try:
            st.session_state[cache_key] = audit_workspace(db_path, data_dir=data_dir or None)
        except Exception as exc:
            st.session_state[cache_key] = {"error": str(exc), "db_path": db_path, "data_dir": data_dir}

    report = st.session_state.get(cache_key)
    if not isinstance(report, Mapping):
        st.info("Run the audit to inspect the current workspace." if english else "진단을 실행하면 현재 Workspace의 DB·PDF·레코드 상태를 확인할 수 있습니다.")
        return
    if report.get("error"):
        st.error(str(report.get("error")))
        return

    st.caption(f"DB · {report.get('db_path', '')}")
    cols = st.columns(5)
    cols[0].metric("DB size" if english else "DB 용량", str(report.get("db_size") or "0 B"))
    cols[1].metric("Tables" if english else "테이블", int(report.get("table_count") or 0))
    cols[2].metric("Rows" if english else "전체 레코드", int(report.get("row_count") or 0))
    cols[3].metric("Orphans" if english else "Orphan", int(report.get("orphan_total") or 0))
    cols[4].metric("Integrity", str(report.get("integrity") or "unknown"))

    recovery = dict(report.get("research_recovery") or {})
    paper_counts = dict(recovery.get("paper_counts") or {})
    paper_rows = list(recovery.get("papers") or [])
    rq_rows = list(recovery.get("research_questions") or [])
    if paper_rows or rq_rows:
        st.markdown("#### " + ("Research asset integrity" if english else "연구 자산 Integrity"))
        st.caption(
            "Paper is the primary recovery unit. Research-question progress is reconstructed from its linked papers; no recovery action is executed here."
            if english else
            "Paper를 1차 복원 단위로 보고, 연결된 논문 상태에서 연구질문 작업 상태를 재구성합니다. 이 화면에서는 실제 복원을 실행하지 않습니다."
        )
        total_papers = int(paper_counts.get("total") or 0)
        complete = int(paper_counts.get("complete") or 0)
        recoverable = sum(int(paper_counts.get(key) or 0) for key in ("knowledge_ready", "reviewed", "source_only"))
        broken = int(paper_counts.get("broken") or 0)
        icols = st.columns(4)
        icols[0].metric("Papers", total_papers)
        icols[1].metric("Complete" if english else "완전 연결", complete, f"{round(100 * complete / total_papers) if total_papers else 0}%")
        icols[2].metric("Recoverable" if english else "중간부터 재개", recoverable)
        icols[3].metric("Broken" if english else "원천 복구 필요", broken)

        stage_label = {
            "complete": "Complete" if english else "Complete · 그대로 사용",
            "knowledge_ready": "Knowledge-ready" if english else "KC 정상 · Ontology 보완",
            "reviewed": "Reviewed" if english else "Review 정상 · KC부터",
            "source_only": "Source-only" if english else "원문 있음 · Review부터",
            "broken": "Broken" if english else "원문/연결 복구 필요",
        }
        recovery_label = {
            "keep": "Keep" if english else "그대로 사용",
            "ontology": "Resume at Ontology" if english else "Ontology 연결부터",
            "knowledge_card": "Resume at Knowledge Card" if english else "Knowledge Card 생성부터",
            "paper_review": "Resume at Paper Review" if english else "Paper Review부터",
            "source": "Restore source" if english else "원문 확보부터",
        }
        with st.expander("Paper integrity" if english else "Paper 단위 현재 상태", expanded=True):
            filter_options = ["all", "complete", "knowledge_ready", "reviewed", "source_only", "broken"]
            selected = st.selectbox(
                "Status" if english else "상태 필터", filter_options, index=0,
                format_func=lambda value: ("All" if value == "all" and english else "전체" if value == "all" else stage_label.get(value, value)),
                key=f"paper-integrity-filter::{db_path}",
            )
            shown = [row for row in paper_rows if selected == "all" or str(row.get("stage")) == selected]
            display_rows = []
            for row in shown:
                kc = int(row.get("knowledge_cards") or 0)
                assigned = int(row.get("ontology_assigned") or 0)
                display_rows.append({
                    "Paper" if english else "논문": _short(row.get("title", ""), 110),
                    "Source" if english else "원문": "✓" if row.get("source_available") else "—",
                    "Review": "✓" if row.get("reviewed") else "—",
                    "KC": kc,
                    "Type": f"{assigned}/{kc}" if kc else "—",
                    "Status" if english else "상태": stage_label.get(str(row.get("stage") or ""), str(row.get("stage") or "")),
                    "Recovery" if english else "복원 시작점": recovery_label.get(str(row.get("recovery") or ""), str(row.get("recovery") or "")),
                })
            if display_rows:
                st.dataframe(display_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No papers in this status." if english else "해당 상태의 논문이 없습니다.")

        with st.expander("Research question integrity" if english else "연구질문 단위 복원 상태", expanded=True):
            if rq_rows:
                rq_display = []
                for row in rq_rows:
                    paper_total = int(row.get("papers") or 0)
                    rq_display.append({
                        "Research question" if english else "연구질문": _short(row.get("question", ""), 100),
                        "Papers" if english else "관련 논문": paper_total,
                        "Complete" if english else "정상": f"{int(row.get('complete_papers') or 0)}/{paper_total} ({int(row.get('complete_pct') or 0)}%)" if paper_total else "0",
                        "Reviewed" if english else "Review 정상률": f"{int(row.get('reviewed_pct') or 0)}%" if paper_total else "—",
                        "Knowledge" if english else "KC 보유율": f"{int(row.get('knowledge_pct') or 0)}%" if paper_total else "—",
                        "Status" if english else "RQ 상태": str(row.get("status") or ""),
                    })
                st.dataframe(rq_display, use_container_width=True, hide_index=True)
                st.caption(
                    "Complete % means the share of linked papers whose Review → Knowledge Card → Type assignment chain is intact."
                    if english else
                    "정상 %는 해당 연구질문에 연결된 논문 중 Review → Knowledge Card → Type 할당이 모두 이어진 논문의 비율입니다."
                )
            else:
                st.caption("No research questions found." if english else "복원 상태를 계산할 연구질문이 없습니다.")

        with st.expander("Recovery plan (preview only)" if english else "복원 방법 · Preview", expanded=False):
            plan = list(recovery.get("recovery_plan") or [])
            if plan:
                plan_rows = [{
                    "Recovery" if english else "복원 방법": recovery_label.get(str(item.get("action") or ""), str(item.get("action") or "")),
                    "Papers" if english else "논문 수": int(item.get("papers") or 0),
                } for item in plan]
                st.dataframe(plan_rows, use_container_width=True, hide_index=True)
            ontology_status = dict(recovery.get("ontology") or {})
            if ontology_status:
                st.caption(
                    (f"Ontology coverage: {ontology_status.get('assigned_cards', 0)}/{ontology_status.get('active_cards', 0)} active Knowledge Cards assigned ({ontology_status.get('assignment_pct', 0)}%)."
                     if english else
                     f"Ontology 연결: 승인 Knowledge Card {ontology_status.get('active_cards', 0)}개 중 {ontology_status.get('assigned_cards', 0)}개 Type 할당 ({ontology_status.get('assignment_pct', 0)}%).")
                )
            st.info(
                "Recovery execution is intentionally disabled. The current step only identifies which existing assets can be reused and where each paper should resume."
                if english else
                "복원 실행은 아직 비활성화되어 있습니다. 현재 단계에서는 기존 자산을 어디까지 재사용할 수 있는지와 각 논문의 재시작 지점만 계산합니다."
            )

    assets = dict(report.get("assets") or {})
    st.markdown("#### " + ("PDF / file footprint" if english else "PDF · 파일 사용량"))
    acols = st.columns(5)
    acols[0].metric("PDF files" if english else "PDF 파일", int(assets.get("data_dir_pdf_files") or 0))
    acols[1].metric("PDF size" if english else "PDF 총용량", str(assets.get("data_dir_pdf_size") or "0 B"))
    acols[2].metric("Missing refs" if english else "경로 누락", int(assets.get("missing_pdf_rows") or 0))
    acols[3].metric("Unreferenced" if english else "미참조 PDF", int(assets.get("unreferenced_pdf_files") or 0))
    acols[4].metric("Duplicate candidates" if english else "중복 후보", int(assets.get("same_size_candidate_groups") or 0))
    st.caption(
        (f"Current workspace references {assets.get('unique_referenced_files', 0)} unique local PDF files ({assets.get('referenced_size', '0 B')}). "
         f"Unreferenced PDFs under the data directory occupy {assets.get('unreferenced_pdf_size', '0 B')}.")
        if english else
        (f"현재 Workspace가 참조하는 로컬 PDF는 {assets.get('unique_referenced_files', 0)}개 ({assets.get('referenced_size', '0 B')})입니다. "
         f"Data 디렉터리의 미참조 PDF는 {assets.get('unreferenced_pdf_files', 0)}개 ({assets.get('unreferenced_pdf_size', '0 B')})입니다.")
    )

    issues = list(report.get("issues") or [])
    if issues:
        st.markdown("#### " + ("Findings" if english else "확인 필요 항목"))
        st.dataframe(issues, use_container_width=True, hide_index=True)
    else:
        st.success("No obvious storage issues detected." if english else "현재 1차 진단에서 뚜렷한 저장소 이상은 발견되지 않았습니다.")

    with st.expander("Table inventory" if english else "테이블별 레코드 현황", expanded=False):
        rows = list(report.get("tables") or [])
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        unclassified = list(report.get("unclassified_tables") or [])
        if unclassified:
            st.warning(("Unclassified tables: " if english else "Sync 정책 미분류 테이블: ") + ", ".join(unclassified))
        missing_sync = list(report.get("declared_sync_tables_missing") or [])
        if missing_sync:
            st.caption(("Declared sync tables absent in this DB: " if english else "현재 DB에 없는 선언된 sync 테이블: ") + ", ".join(missing_sync))

    with st.expander("Orphan / duplicate candidates" if english else "Orphan · 중복 후보", expanded=False):
        orphans = [row for row in (report.get("orphans") or []) if int(row.get("count") or 0) > 0]
        if orphans:
            st.caption("Orphan references" if english else "부모 레코드를 찾을 수 없는 참조")
            st.dataframe(orphans, use_container_width=True, hide_index=True)
        duplicates = dict(report.get("duplicates") or {})
        shown = False
        for label, rows in duplicates.items():
            if rows:
                shown = True
                st.caption(label)
                st.dataframe(rows, use_container_width=True, hide_index=True)
        if not orphans and not shown:
            st.caption("No orphan or obvious exact-duplicate candidates found." if english else "1차 규칙에서 orphan 또는 명백한 중복 후보가 발견되지 않았습니다.")

    with st.expander("PDF path details" if english else "PDF 경로 상세", expanded=False):
        missing = list(assets.get("missing_examples") or [])
        if missing:
            st.caption("Referenced but missing" if english else "DB에는 있으나 현재 머신에서 찾을 수 없는 PDF")
            st.dataframe(missing, use_container_width=True, hide_index=True)
        unreferenced = list(assets.get("unreferenced_examples") or [])
        if unreferenced:
            st.caption("Unreferenced PDF examples" if english else "현재 Workspace에서 참조하지 않는 PDF 예시")
            for item in unreferenced:
                st.code(str(item), language=None)
        same_size = list(assets.get("same_size_examples") or [])
        if same_size:
            st.caption("Same-size duplicate candidates; content hash is not calculated yet." if english else "동일 크기 중복 후보입니다. 아직 내용 hash 비교는 하지 않습니다.")
            for item in same_size:
                st.write(f"{item.get('size', '')} · {len(item.get('files') or [])} files")
                for file_path in item.get("files") or []:
                    st.code(str(file_path), language=None)

    with st.expander("Runtime / pending record counts" if english else "실행·대기 레코드 상태", expanded=False):
        statuses = dict(report.get("status_counts") or {})
        if not statuses:
            st.caption("No status-bearing tables found." if english else "상태 집계 대상 테이블이 없습니다.")
        for table, values in statuses.items():
            st.caption(table)
            st.write(" · ".join(f"{key}: {value}" for key, value in values.items()))

def render_knowledge_workspace(
    st: Any, snapshot: Mapping[str, Any], *, english: bool = True,
    update_paper_metadata: Callable[[str, list[str] | str, str, str | None], Mapping[str, Any]] | None = None,
    attach_paper_pdf: Callable[[str, str, bytes], Mapping[str, Any]] | None = None,
    enqueue_ontology_work: Callable[[], Mapping[str, Any]] | None = None,
    manage_ontology: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    st.subheader("Knowledge" if english else "지식")
    st.caption(
        "See what the Research Fellow currently knows, which papers support that knowledge, and where structure or review is still needed."
        if english else
        "연구위원이 무엇을 알고 있는지, 어떤 논문이 그 지식을 뒷받침하는지, 어디에 구조화·검토가 필요한지 봅니다."
    )

    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(5)
    cols[0].metric("Knowledge cards" if english else "지식카드", int(counts.get("cards", 0)))
    cols[1].metric("Reinforced" if english else "근거 보강", int(counts.get("reinforced_cards", 0)))
    cols[2].metric("Contested" if english else "충돌 지식", int(counts.get("contested_cards", 0)))
    cols[3].metric("Papers" if english else "근거 논문", int(counts.get("papers", 0)))
    cols[4].metric("Knowledge gaps" if english else "지식 과제", int(counts.get("gaps", 0)))

    knowledge_tab, papers_tab, structure_tab, audit_tab = st.tabs([
        "Knowledge" if english else "지식 상태",
        "Evidence Library" if english else "서재함 · 근거 논문",
        "Ontology & Gaps" if english else "온톨로지 · 지식과제",
        "Workspace Audit" if english else "Workspace 진단",
    ])
    with knowledge_tab:
        _render_knowledge_state(st, snapshot, english=english)
    with papers_tab:
        _render_evidence_library(st, snapshot, english=english, update_paper_metadata=update_paper_metadata, attach_paper_pdf=attach_paper_pdf)
    with structure_tab:
        _render_structure_and_gaps(st, snapshot, english=english, enqueue_ontology_work=enqueue_ontology_work, manage_ontology=manage_ontology)
    with audit_tab:
        _render_workspace_audit(st, snapshot, english=english)
