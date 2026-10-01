"""Streamlit renderer for the job-centred Knowledge Workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


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
            st.markdown("**Research-question interpretations**" if english else "**연구질문별 해석**")
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
                        st.caption("Question-specific summary" if english else "연구질문 관점 요약")
                        st.write(rq_summary)
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

        if update_paper_metadata is not None and paper_id:
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
                note_text = st.text_area(
                    "Researcher comment" if english else "연구자 코멘트",
                    value=researcher_note,
                    height=110,
                    key=f"paper-library-note-{paper_id}",
                )
                if st.button(
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


def _render_structure_and_gaps(st: Any, snapshot: Mapping[str, Any], *, english: bool) -> None:
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


def render_knowledge_workspace(
    st: Any, snapshot: Mapping[str, Any], *, english: bool = True,
    update_paper_metadata: Callable[[str, list[str] | str, str, str | None], Mapping[str, Any]] | None = None,
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

    knowledge_tab, papers_tab, structure_tab = st.tabs([
        "Knowledge" if english else "지식 상태",
        "Evidence Library" if english else "서재함 · 근거 논문",
        "Structure & Gaps" if english else "지식 구조 · 과제",
    ])
    with knowledge_tab:
        _render_knowledge_state(st, snapshot, english=english)
    with papers_tab:
        _render_evidence_library(st, snapshot, english=english, update_paper_metadata=update_paper_metadata)
    with structure_tab:
        _render_structure_and_gaps(st, snapshot, english=english)
