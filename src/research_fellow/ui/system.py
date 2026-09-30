"""Read-only Streamlit renderer for the System / Developer Workspace."""
from __future__ import annotations
from typing import Any, Mapping


def _short(text: str, n: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_system_workspace(st: Any, snapshot: Mapping[str, Any], *, english: bool = True) -> None:
    st.subheader("System" if english else "시스템")
    st.caption(
        "Developer-only read models of specification, contracts, autonomy, runtime, and diagnostics. No research state is edited here."
        if english else
        "명세·계약·자율성·런타임·진단 상태를 읽기 전용으로 보는 개발자 영역입니다. 여기서 연구 상태를 수정하지 않습니다."
    )
    counts = dict(snapshot.get("counts") or {})
    cols = st.columns(5)
    cols[0].metric("Workflows" if english else "워크플로우", int(counts.get("workflows", 0)))
    cols[1].metric("Capabilities" if english else "Capability", int(counts.get("capabilities", 0)))
    cols[2].metric("Interactions" if english else "Interaction", int(counts.get("interactions", 0)))
    cols[3].metric("Semantic types" if english else "Semantic Type", int(counts.get("semantic_types", 0)))
    cols[4].metric("Architecture issues" if english else "아키텍처 이슈", int(counts.get("architecture_issues", 0)))

    spec_tab, contracts_tab, autonomy_tab, runtime_tab, development_tab = st.tabs([
        "Agent Specification" if english else "Agent 명세",
        "Contracts" if english else "Contracts",
        "Autonomy" if english else "자율성",
        "Runtime" if english else "런타임",
        "Development" if english else "개발 진단",
    ])

    with spec_tab:
        specification = dict(snapshot.get("specification") or {})
        st.markdown("### AJD")
        for item in specification.get("ajd") or []:
            with st.expander(f"{item.get('id') or item.get('file')} · {item.get('role', '')}"):
                st.write(_short(item.get("purpose", ""), 500))
                st.caption(f"responsibilities: {item.get('responsibilities', 0)} · memory contracts: {item.get('memory_items', 0)} · {item.get('file', '')}")
        st.markdown("### " + ("Workflows" if english else "워크플로우"))
        for item in specification.get("workflows") or []:
            kinds = ", ".join(f"{key}:{value}" for key, value in sorted(dict(item.get("step_kinds") or {}).items()))
            st.markdown(f"**{item.get('id')}** · {item.get('agent')} · rev {item.get('revision')} · {item.get('step_count', 0)} steps")
            st.caption((kinds + " · " if kinds else "") + _short(item.get("purpose", ""), 260))
        st.markdown("### Topology")
        for item in specification.get("topologies") or []:
            st.markdown(f"**{item.get('id')}** · loops {item.get('loops', 0)} · links {item.get('links', 0)} · boundary {item.get('boundary_interactions', 0)}")
            st.caption(_short(item.get("purpose", ""), 320))

    with contracts_tab:
        contracts = dict(snapshot.get("contracts") or {})
        issues = list(contracts.get("capability_issues") or [])
        if issues:
            st.warning(("Contract catalog issues detected." if english else "Contract catalog 이슈가 있습니다.") + f" ({len(issues)})")
            for item in issues:
                st.caption(f"{item.get('file')} · {item.get('id')} · {item.get('issue')}")
        left, right = st.columns(2)
        with left:
            st.markdown("### Capability Contracts")
            for item in contracts.get("capabilities") or []:
                st.markdown(f"**{item.get('id') or item.get('file')}** · uses {item.get('uses', 0)} · produces {item.get('produces', 0)}")
                st.caption(_short(item.get("purpose", ""), 220))
        with right:
            st.markdown("### Interaction Contracts")
            for item in contracts.get("interactions") or []:
                response = "response" if item.get("requires_response") else "inform"
                st.markdown(f"**{item.get('id')}** · {item.get('mode')} · {item.get('actor')} · {response}")
                st.caption(_short(item.get("purpose", ""), 220))
        st.markdown("### Semantic Types & Bindings")
        st.caption(f"semantic types: {len(contracts.get('semantic_types') or [])} · version: {contracts.get('semantic_type_version') or '-'}")
        bindings = dict(contracts.get("bindings") or {})
        st.caption("capability bindings: " + ", ".join(bindings.get("capability_profiles") or ["none"]))
        st.caption("interaction bindings: " + ", ".join(bindings.get("interaction_profiles") or ["none"]))

    with autonomy_tab:
        autonomy = dict(snapshot.get("autonomy") or {})
        coverage = dict(autonomy.get("policy_coverage") or {})
        cols = st.columns(5)
        for index, level in enumerate(("H0", "H1", "H2", "H3")):
            cols[index].metric(level, int((autonomy.get("level_counts") or {}).get(level, 0)))
        cols[4].metric("Policy coverage", f"{coverage.get('covered', 0)}/{coverage.get('interaction_count', 0)}")
        if coverage.get("uncovered"):
            st.info(("Default ESCALATE (no explicit policy): " if english else "명시 Policy 없음 → 기본 ESCALATE: ") + ", ".join(coverage["uncovered"]))
        st.markdown("### " + ("Interaction classification" if english else "Interaction 분류"))
        for item in autonomy.get("classes") or []:
            transition = f" → {item.get('target_level')}" if item.get("current_level") != item.get("target_level") else ""
            st.markdown(f"**{item.get('current_level')}{transition} · {item.get('id')}** · {item.get('category')}")
            st.caption(_short(item.get("rationale", ""), 300))
        st.markdown("### " + ("Decision distribution" if english else "자율성 판단 분포"))
        decision_counts = dict(autonomy.get("decision_counts") or {})
        dcols = st.columns(3)
        dcols[0].metric("AUTO", int(decision_counts.get("auto", 0)))
        dcols[1].metric("AUTO_NOTIFY", int(decision_counts.get("auto_notify", 0)))
        dcols[2].metric("ESCALATE", int(decision_counts.get("escalate", 0)))
        reasons = dict(autonomy.get("escalation_reasons") or {})
        if reasons:
            st.caption(("Escalation reasons: " if english else "Escalation 사유: ") + " · ".join(f"{key} {value}" for key, value in reasons.items()))
        st.markdown("### " + ("Recent autonomy decisions" if english else "최근 자율성 판단"))
        decisions = list(autonomy.get("recent_decisions") or [])
        if not decisions:
            st.caption("No durable autonomy decisions have been recorded yet." if english else "아직 영속화된 자율성 판단 이력이 없습니다.")
        for item in decisions:
            st.markdown(f"**{item.get('action', '')} · {item.get('interaction_id', '')}**")
            detail = f"{item.get('autonomy_level', '')} · {item.get('policy_id', '')} · {item.get('reasons') or item.get('signals') or ''}"
            st.caption(_short(detail, 320))

    with runtime_tab:
        runtime = dict(snapshot.get("runtime") or {})
        counts = dict(runtime.get("counts") or {})
        cols = st.columns(4)
        cols[0].metric("Runs", int(counts.get("runs", 0)))
        cols[1].metric("Running", int(counts.get("running", 0)))
        cols[2].metric("Waiting checkpoints", int(counts.get("waiting_checkpoints", 0)))
        cols[3].metric("Failures", int(counts.get("failures_needing_attention", 0)))
        st.markdown("### Workflow Runs")
        for item in runtime.get("runs") or []:
            st.markdown(f"**{item.get('status', '')} · {item.get('run_id', '')}** · {item.get('current_stage', '')}")
            if item.get("last_error_message"):
                st.caption(_short(item.get("last_error_message", ""), 280))
        st.markdown("### Checkpoints")
        for item in runtime.get("checkpoints") or []:
            st.markdown(f"**{item.get('status', '')} · {item.get('workflow_id', '')}** · {item.get('checkpoint_id', '')}")
            if item.get("interaction_id"):
                st.caption(f"interaction: {item.get('interaction_id')} · step: {item.get('step_id')}")
        st.markdown("### Failures / Recovery")
        failures = list(runtime.get("failures") or [])
        if not failures:
            st.caption("No unresolved runtime failures." if english else "미해결 런타임 실패가 없습니다.")
        for item in failures:
            st.markdown(f"**{item.get('stage', '')} · {item.get('error_type', '')}**")
            st.caption(_short(item.get("error_message", ""), 300))

    with development_tab:
        development = dict(snapshot.get("development") or {})
        logs = dict(development.get("prompt_logs") or {})
        audit = dict(development.get("architecture_audit") or {})
        st.markdown("### Prompt Logs")
        st.caption(f"ledger records: {logs.get('count', 0)} · JSONL mirror: {logs.get('jsonl_count', 0)}")
        for item in logs.get("records") or []:
            size = item.get("prompt_estimated_tokens") or item.get("prompt_chars") or "-"
            st.markdown(f"**{item.get('profile_name', '')} · {item.get('model', '')}** · prompt {size}")
            if item.get("error"):
                st.caption("ERROR · " + _short(item.get("error", ""), 240))
            else:
                st.caption(f"response chars: {item.get('response_chars') or '-'} · elapsed: {item.get('elapsed_seconds') or '-'}s")
        st.markdown("### Execution Logs")
        execution = dict(development.get("execution_logs") or {})
        st.caption(f"runs: {len(execution.get('runs') or [])} · unresolved failures: {len(execution.get('failures') or [])}")
        st.markdown("### Architecture Audit")
        if audit.get("status") == "pass":
            st.success("Architecture audit passed." if english else "Architecture audit 통과")
            report = dict(audit.get("report") or {})
            st.caption(f"modules: {report.get('application_modules', '-')} · AJD links: {report.get('ajd_links', '-')} · capability gaps: {len(report.get('capability_implementation_needed') or [])}")
        else:
            st.error("Architecture audit found a baseline issue." if english else "Architecture audit에서 baseline 이슈를 발견했습니다.")
            st.code(str(audit.get("error") or "unknown architecture audit failure"))
