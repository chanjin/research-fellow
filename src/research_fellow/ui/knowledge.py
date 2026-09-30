"""Streamlit renderer for the job-centred Knowledge Workspace."""
from __future__ import annotations
from typing import Any, Mapping


def _short(text: str, n: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_knowledge_workspace(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    st.subheader("Knowledge" if english else "지식")
    st.caption(
        "See what the Research Fellow currently knows, how well it is supported, and where structure or review is still needed."
        if english else
        "연구위원이 현재 무엇을 알고 있는지, 근거가 얼마나 충분한지, 어디에 구조화·검토가 필요한지 봅니다."
    )

    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(5)
    cols[0].metric("Knowledge cards" if english else "지식카드", int(counts.get("cards", 0)))
    cols[1].metric("Reinforced" if english else "근거 보강", int(counts.get("reinforced_cards", 0)))
    cols[2].metric("Contested" if english else "충돌 지식", int(counts.get("contested_cards", 0)))
    cols[3].metric("Untyped" if english else "미구조화", int(counts.get("untyped_cards", 0)))
    cols[4].metric("Knowledge gaps" if english else "지식 과제", int(counts.get("gaps", 0)))

    left, right = st.columns(2)
    with left:
        st.markdown("### " + ("Knowledge State" if english else "지식 상태"))
        cards = list(snapshot.get("cards") or [])
        if not cards:
            st.caption("No approved knowledge cards yet." if english else "승인된 지식카드가 없습니다.")
        for card in cards[:15]:
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

    with right:
        st.markdown("### " + ("Evidence & Change" if english else "근거와 변화"))
        evidence_levels = dict(snapshot.get("evidence_levels") or {})
        if evidence_levels:
            st.caption(
                " · ".join(f"{name}: {count}" for name, count in sorted(evidence_levels.items()))
            )
        changes = list(snapshot.get("changes") or [])
        if not changes:
            st.caption("No recent knowledge changes." if english else "최근 지식 변화가 없습니다.")
        for item in changes[:12]:
            marker = "NEW" if item.get("status") == "new" else "reviewed"
            st.markdown(f"**{marker}** — {_short(item.get('title', ''), 150)}")
            if item.get("detail"):
                st.caption(_short(item.get("detail", ""), 180))

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

    st.markdown("### " + ("Knowledge Gaps & Structuring Needs" if english else "지식 과제와 구조화 필요"))
    gaps = list(snapshot.get("gaps") or [])
    if not gaps:
        st.success("No immediate knowledge-structure gap requires attention." if english else "즉시 처리할 지식 구조 과제가 없습니다.")
    else:
        for item in gaps:
            st.markdown(f"**{str(item.get('priority') or 'medium').upper()} · {_short(item.get('title', ''), 180)}**")
            st.caption(_short(item.get("reason", ""), 280))
