from __future__ import annotations

"""Read-only audit of one Research Fellow workspace and its local assets."""

from dataclasses import dataclass
from pathlib import Path
import json
import re
import sqlite3
from typing import Any, Iterable

from research_fellow.workspace_sync import LOCAL_ONLY_TABLES, SYNC_TABLES


def _bytes_label(value: int) -> str:
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{value} B"


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()}


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table)}").fetchone()[0])
    except sqlite3.DatabaseError:
        return -1


def _group_duplicates(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    *,
    where: str = "",
    normalize: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if table not in _table_names(conn) or column not in _columns(conn, table):
        return []
    expr = f"LOWER(TRIM({column}))" if normalize else f"TRIM({column})"
    clauses = [f"{column} IS NOT NULL", f"TRIM({column}) <> ''"]
    if where:
        clauses.append(f"({where})")
    sql = (
        f"SELECT {expr} AS value, COUNT(*) AS count FROM {_quote_identifier(table)} "
        f"WHERE {' AND '.join(clauses)} GROUP BY {expr} HAVING COUNT(*) > 1 "
        "ORDER BY count DESC, value LIMIT ?"
    )
    try:
        return [{"value": str(row[0] or ""), "count": int(row[1])} for row in conn.execute(sql, (limit,)).fetchall()]
    except sqlite3.DatabaseError:
        return []


def _orphan_count(
    conn: sqlite3.Connection,
    child: str,
    child_column: str,
    parent: str,
    parent_column: str,
    *,
    child_where: str = "",
) -> int | None:
    tables = set(_table_names(conn))
    if child not in tables or parent not in tables:
        return None
    if child_column not in _columns(conn, child) or parent_column not in _columns(conn, parent):
        return None
    clauses = [f"c.{child_column} IS NOT NULL", f"TRIM(CAST(c.{child_column} AS TEXT)) <> ''"]
    if child_where:
        clauses.append(f"({child_where})")
    sql = (
        f"SELECT COUNT(*) FROM {_quote_identifier(child)} c "
        f"LEFT JOIN {_quote_identifier(parent)} p ON c.{child_column}=p.{parent_column} "
        f"WHERE {' AND '.join(clauses)} AND p.{parent_column} IS NULL"
    )
    try:
        return int(conn.execute(sql).fetchone()[0])
    except sqlite3.DatabaseError:
        return None


ORPHAN_RULES: tuple[tuple[str, str, str, str, str], ...] = (
    ("research_question_intents", "rq_id", "research_questions", "rq_id", "RQ → intent"),
    ("research_question_sources", "rq_id", "research_questions", "rq_id", "RQ → source"),
    ("research_question_changes", "rq_id", "research_questions", "rq_id", "RQ → change"),
    ("research_question_threads", "rq_id", "research_questions", "rq_id", "RQ → thread"),
    ("research_question_versions", "rq_id", "research_questions", "rq_id", "RQ → version"),
    ("paper_question_analyses", "paper_id", "paper_shelf", "paper_id", "Paper → RQ analysis"),
    ("paper_question_analyses", "research_question_id", "research_questions", "rq_id", "RQ analysis → RQ"),
    ("paper_card_links", "paper_id", "paper_shelf", "paper_id", "Paper → card link"),
    ("paper_card_links", "card_id", "knowledge_cards", "card_id", "Card → paper link"),
    ("paper_abstracts", "paper_id", "paper_shelf", "paper_id", "Paper abstract"),
    ("paper_analyses", "paper_id", "paper_shelf", "paper_id", "Paper analysis"),
    ("paper_reading_questions", "paper_id", "paper_shelf", "paper_id", "Paper reading question"),
    ("paper_reading_reviews", "question_id", "paper_reading_questions", "question_id", "Reading review → question"),
    ("ontology_card_assignments", "type_id", "ontology_types", "type_id", "Ontology assignment → Type"),
    ("ontology_card_assignments", "card_id", "knowledge_cards", "card_id", "Ontology assignment → card"),
    ("ontology_type_relations", "source_type_id", "ontology_types", "type_id", "Ontology relation → source Type"),
    ("ontology_type_relations", "target_type_id", "ontology_types", "type_id", "Ontology relation → target Type"),
    ("knowledge_relations", "source_card_id", "knowledge_cards", "card_id", "Knowledge relation → source card"),
    ("knowledge_relations", "target_card_id", "knowledge_cards", "card_id", "Knowledge relation → target card"),
)


def _asset_audit(conn: sqlite3.Connection, data_dir: Path) -> dict[str, Any]:
    refs: list[dict[str, str]] = []
    if "paper_shelf" in _table_names(conn) and {"paper_id", "pdf_path"}.issubset(_columns(conn, "paper_shelf")):
        for row in conn.execute("SELECT paper_id, title, pdf_path FROM paper_shelf ORDER BY paper_id").fetchall():
            value = str(row[2] or "").strip()
            if value:
                refs.append({"paper_id": str(row[0] or ""), "title": str(row[1] or ""), "pdf_path": value})

    existing: list[tuple[dict[str, str], Path]] = []
    missing: list[dict[str, str]] = []
    remote_in_local_column: list[dict[str, str]] = []
    for item in refs:
        raw = item["pdf_path"]
        if raw.lower().startswith(("http://", "https://")):
            remote_in_local_column.append(item)
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            existing.append((item, path.resolve()))
        else:
            missing.append(item)

    unique_existing: dict[str, Path] = {}
    path_ref_counts: dict[str, int] = {}
    for _, path in existing:
        key = str(path)
        unique_existing[key] = path
        path_ref_counts[key] = path_ref_counts.get(key, 0) + 1
    referenced_bytes = sum(path.stat().st_size for path in unique_existing.values() if path.exists())

    all_pdfs: list[Path] = []
    try:
        all_pdfs = [p.resolve() for p in data_dir.rglob("*.pdf") if p.is_file()]
    except OSError:
        all_pdfs = []
    all_pdf_map = {str(p): p for p in all_pdfs}
    referenced_paths = set(unique_existing)
    unreferenced = [p for key, p in all_pdf_map.items() if key not in referenced_paths]
    all_pdf_bytes = sum(p.stat().st_size for p in all_pdf_map.values() if p.exists())
    unreferenced_bytes = sum(p.stat().st_size for p in unreferenced if p.exists())

    size_groups: dict[int, list[Path]] = {}
    for path in all_pdf_map.values():
        try:
            size_groups.setdefault(path.stat().st_size, []).append(path)
        except OSError:
            continue
    same_size_groups = [paths for size, paths in size_groups.items() if size > 0 and len(paths) > 1]

    return {
        "referenced_pdf_rows": len(refs),
        "existing_pdf_rows": len(existing),
        "missing_pdf_rows": len(missing),
        "remote_url_in_pdf_path_rows": len(remote_in_local_column),
        "unique_referenced_files": len(unique_existing),
        "duplicate_path_reference_rows": sum(max(0, n - 1) for n in path_ref_counts.values()),
        "referenced_bytes": referenced_bytes,
        "referenced_size": _bytes_label(referenced_bytes),
        "data_dir_pdf_files": len(all_pdf_map),
        "data_dir_pdf_bytes": all_pdf_bytes,
        "data_dir_pdf_size": _bytes_label(all_pdf_bytes),
        "unreferenced_pdf_files": len(unreferenced),
        "unreferenced_pdf_bytes": unreferenced_bytes,
        "unreferenced_pdf_size": _bytes_label(unreferenced_bytes),
        "same_size_candidate_groups": len(same_size_groups),
        "same_size_candidate_files": sum(len(paths) for paths in same_size_groups),
        "missing_examples": missing[:10],
        "unreferenced_examples": [str(p) for p in sorted(unreferenced)[:10]],
        "same_size_examples": [
            {"size": _bytes_label(paths[0].stat().st_size), "files": [str(p) for p in paths[:5]]}
            for paths in sorted(same_size_groups, key=lambda xs: xs[0].stat().st_size, reverse=True)[:10]
        ],
    }



def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _paper_recovery_audit(conn: sqlite3.Connection) -> dict[str, Any]:
    """Reconstruct research-asset integrity from papers upward without mutating state."""
    tables = set(_table_names(conn))
    if "paper_shelf" not in tables:
        return {
            "papers": [], "paper_counts": {}, "research_questions": [],
            "rq_counts": {}, "recovery_plan": [], "ontology": {},
        }

    paper_columns = _columns(conn, "paper_shelf")
    select_cols = [name for name in (
        "paper_id", "title", "pdf_path", "source_url", "full_text_url",
        "pdf_url", "origin_links_json", "reading_status", "shelf_status",
    ) if name in paper_columns]
    papers_raw = [dict(row) for row in conn.execute(
        f"SELECT {', '.join(select_cols)} FROM paper_shelf ORDER BY title, paper_id"
    ).fetchall()]

    active_cards: set[str] = set()
    if "knowledge_cards" in tables:
        cols = _columns(conn, "knowledge_cards")
        deleted_filter = "WHERE deleted_at IS NULL" if "deleted_at" in cols else ""
        active_cards = {str(row[0]) for row in conn.execute(
            f"SELECT card_id FROM knowledge_cards {deleted_filter}"
        ).fetchall()}

    cards_by_paper: dict[str, set[str]] = {}
    if "paper_card_links" in tables:
        for row in conn.execute("SELECT paper_id, card_id FROM paper_card_links").fetchall():
            paper_id, card_id = str(row[0] or ""), str(row[1] or "")
            if paper_id and card_id and (not active_cards or card_id in active_cards):
                cards_by_paper.setdefault(paper_id, set()).add(card_id)

    assigned_cards: set[str] = set()
    if "ontology_card_assignments" in tables:
        # Ignore assignments to deleted/missing Types where the schema supports that check.
        if "ontology_types" in tables and "deleted_at" in _columns(conn, "ontology_types"):
            rows = conn.execute(
                "SELECT a.card_id FROM ontology_card_assignments a "
                "JOIN ontology_types t ON t.type_id=a.type_id WHERE t.deleted_at IS NULL"
            ).fetchall()
        else:
            rows = conn.execute("SELECT card_id FROM ontology_card_assignments").fetchall()
        assigned_cards = {str(row[0] or "") for row in rows if str(row[0] or "")}

    rq_reviews_by_paper: dict[str, set[str]] = {}
    reviewed_papers: set[str] = set()
    if "paper_question_analyses" in tables:
        cols = _columns(conn, "paper_question_analyses")
        rq_col = "research_question_id" if "research_question_id" in cols else None
        if rq_col:
            for row in conn.execute(
                "SELECT paper_id, research_question_id, summary, reading_raw_output FROM paper_question_analyses"
            ).fetchall():
                paper_id = str(row[0] or "")
                rq_id = str(row[1] or "")
                has_content = bool(str(row[2] or "").strip() or str(row[3] or "").strip())
                if paper_id and has_content:
                    reviewed_papers.add(paper_id)
                    if rq_id:
                        rq_reviews_by_paper.setdefault(paper_id, set()).add(rq_id)
    if "paper_analyses" in tables:
        cols = _columns(conn, "paper_analyses")
        wanted = [c for c in ("summary", "reading_raw_output", "generated_at") if c in cols]
        if wanted:
            expr = " OR ".join(f"TRIM(COALESCE({c}, '')) <> ''" for c in wanted)
            reviewed_papers.update(str(row[0]) for row in conn.execute(
                f"SELECT paper_id FROM paper_analyses WHERE {expr}"
            ).fetchall())
    # Some workspaces contain the later research-analysis table. Its exact schema evolved,
    # so treat any row with a paper_id as an available analysis without assuming columns.
    if "paper_research_analyses" in tables and "paper_id" in _columns(conn, "paper_research_analyses"):
        reviewed_papers.update(str(row[0]) for row in conn.execute(
            "SELECT DISTINCT paper_id FROM paper_research_analyses WHERE paper_id IS NOT NULL"
        ).fetchall())

    rq_to_papers: dict[str, set[str]] = {}
    for paper_id, rq_ids in rq_reviews_by_paper.items():
        for rq_id in rq_ids:
            rq_to_papers.setdefault(rq_id, set()).add(paper_id)

    # Origin lineage preserves the RQ↔paper relation even before a review was completed.
    for paper in papers_raw:
        paper_id = str(paper.get("paper_id") or "")
        for origin in _json_list(paper.get("origin_links_json")):
            if not isinstance(origin, dict):
                continue
            if str(origin.get("origin_type") or "") != "researcher_question":
                continue
            rq_id = str(origin.get("origin_id") or "").strip()
            if rq_id:
                rq_to_papers.setdefault(rq_id, set()).add(paper_id)

    paper_rows: list[dict[str, Any]] = []
    by_paper: dict[str, dict[str, Any]] = {}
    recovery_counts: dict[str, int] = {}
    for paper in papers_raw:
        paper_id = str(paper.get("paper_id") or "")
        pdf_path = str(paper.get("pdf_path") or "").strip()
        local_pdf = False
        if pdf_path and not pdf_path.lower().startswith(("http://", "https://")):
            try:
                local_pdf = Path(pdf_path).expanduser().is_file()
            except OSError:
                local_pdf = False
        remote_source = any(str(paper.get(name) or "").strip().lower().startswith(("http://", "https://"))
                            for name in ("full_text_url", "pdf_url"))
        source_available = bool(local_pdf or remote_source)
        reviewed = paper_id in reviewed_papers
        card_ids = cards_by_paper.get(paper_id, set())
        card_count = len(card_ids)
        assigned_count = sum(1 for card_id in card_ids if card_id in assigned_cards)
        ontology_complete = card_count > 0 and assigned_count == card_count
        ontology_partial = assigned_count > 0 and not ontology_complete

        if reviewed and card_count > 0 and ontology_complete:
            stage = "complete"
            recovery = "keep"
        elif reviewed and card_count > 0:
            stage = "knowledge_ready"
            recovery = "ontology"
        elif reviewed:
            stage = "reviewed"
            recovery = "knowledge_card"
        elif source_available:
            stage = "source_only"
            recovery = "paper_review"
        else:
            stage = "broken"
            recovery = "source"
        recovery_counts[recovery] = recovery_counts.get(recovery, 0) + 1
        row = {
            "paper_id": paper_id,
            "title": str(paper.get("title") or paper_id),
            "source_available": source_available,
            "local_pdf": local_pdf,
            "reviewed": reviewed,
            "knowledge_cards": card_count,
            "ontology_assigned": assigned_count,
            "ontology_complete": ontology_complete,
            "ontology_partial": ontology_partial,
            "stage": stage,
            "recovery": recovery,
        }
        paper_rows.append(row)
        by_paper[paper_id] = row

    stage_order = {"complete": 0, "knowledge_ready": 1, "reviewed": 2, "source_only": 3, "broken": 4}
    paper_rows.sort(key=lambda row: (stage_order.get(str(row.get("stage")), 9), str(row.get("title") or "").casefold()))
    paper_counts: dict[str, int] = {key: 0 for key in stage_order}
    for row in paper_rows:
        paper_counts[str(row["stage"])] = paper_counts.get(str(row["stage"]), 0) + 1
    paper_counts["total"] = len(paper_rows)

    rq_rows: list[dict[str, Any]] = []
    if "research_questions" in tables:
        rq_cols = _columns(conn, "research_questions")
        where = "WHERE deleted_at IS NULL" if "deleted_at" in rq_cols else ""
        select = [c for c in ("rq_id", "question", "status") if c in rq_cols]
        for rq in conn.execute(f"SELECT {', '.join(select)} FROM research_questions {where} ORDER BY updated_at DESC").fetchall():
            item = dict(rq)
            rq_id = str(item.get("rq_id") or "")
            paper_ids = sorted(rq_to_papers.get(rq_id, set()))
            rows = [by_paper[p] for p in paper_ids if p in by_paper]
            total = len(rows)
            complete = sum(1 for row in rows if row.get("stage") == "complete")
            reviewed = sum(1 for row in rows if row.get("reviewed"))
            knowledge = sum(1 for row in rows if int(row.get("knowledge_cards") or 0) > 0)
            rq_rows.append({
                "rq_id": rq_id,
                "question": str(item.get("question") or rq_id),
                "status": str(item.get("status") or ""),
                "papers": total,
                "complete_papers": complete,
                "complete_pct": round(100.0 * complete / total) if total else 0,
                "reviewed_papers": reviewed,
                "reviewed_pct": round(100.0 * reviewed / total) if total else 0,
                "knowledge_papers": knowledge,
                "knowledge_pct": round(100.0 * knowledge / total) if total else 0,
                "paper_ids": paper_ids,
            })

    total_cards = len(active_cards)
    assigned_active = len(active_cards & assigned_cards) if active_cards else 0
    ontology = {
        "active_cards": total_cards,
        "assigned_cards": assigned_active,
        "unassigned_cards": max(0, total_cards - assigned_active),
        "assignment_pct": round(100.0 * assigned_active / total_cards) if total_cards else 0,
    }
    recovery_plan = [
        {"action": key, "papers": recovery_counts.get(key, 0)}
        for key in ("keep", "ontology", "knowledge_card", "paper_review", "source")
        if recovery_counts.get(key, 0)
    ]
    return {
        "papers": paper_rows,
        "paper_counts": paper_counts,
        "research_questions": rq_rows,
        "rq_counts": {"total": len(rq_rows), "with_papers": sum(1 for row in rq_rows if int(row.get("papers") or 0) > 0)},
        "recovery_plan": recovery_plan,
        "ontology": ontology,
    }

def audit_workspace(db_path: str | Path, *, data_dir: str | Path | None = None) -> dict[str, Any]:
    """Inspect one workspace without modifying the DB or any assets."""
    db = Path(db_path).expanduser()
    root = Path(data_dir).expanduser() if data_dir is not None else db.parent
    report: dict[str, Any] = {
        "db_path": str(db),
        "data_dir": str(root),
        "db_exists": db.is_file(),
        "db_bytes": db.stat().st_size if db.is_file() else 0,
    }
    report["db_size"] = _bytes_label(int(report["db_bytes"]))
    if not db.is_file():
        report.update({"integrity": "missing", "tables": [], "issues": [{"kind": "database", "detail": "DB file not found"}]})
        return report

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        try:
            integrity_rows = [str(row[0]) for row in conn.execute("PRAGMA integrity_check").fetchall()]
        except sqlite3.DatabaseError as exc:
            integrity_rows = [f"error: {exc}"]
        report["integrity"] = "ok" if integrity_rows == ["ok"] else "problem"
        report["integrity_detail"] = integrity_rows[:20]

        tables = _table_names(conn)
        table_rows = []
        for table in tables:
            count = _table_count(conn, table)
            classification = "sync" if table in SYNC_TABLES else "local" if table in LOCAL_ONLY_TABLES else "unclassified"
            table_rows.append({"table": table, "rows": count, "classification": classification})
        report["tables"] = sorted(table_rows, key=lambda x: (-int(x["rows"]), str(x["table"])))
        report["table_count"] = len(tables)
        report["row_count"] = sum(max(0, int(x["rows"])) for x in table_rows)
        report["sync_tables_present"] = sum(1 for x in table_rows if x["classification"] == "sync")
        report["local_tables_present"] = sum(1 for x in table_rows if x["classification"] == "local")
        report["unclassified_tables"] = [x["table"] for x in table_rows if x["classification"] == "unclassified"]
        report["declared_sync_tables_missing"] = sorted(set(SYNC_TABLES) - set(tables))

        meta: dict[str, str] = {}
        if "schema_meta" in tables:
            try:
                meta = {str(row[0]): str(row[1]) for row in conn.execute("SELECT key, value FROM schema_meta").fetchall()}
            except sqlite3.DatabaseError:
                meta = {}
        report["schema_meta"] = meta
        profile = {}
        raw_profile = meta.get("workspace_profile_json", "")
        if raw_profile:
            try:
                parsed = json.loads(raw_profile)
                if isinstance(parsed, dict):
                    profile = parsed
            except json.JSONDecodeError:
                profile = {"raw": raw_profile}
        report["workspace_profile"] = profile

        orphan_rows = []
        for child, child_col, parent, parent_col, label in ORPHAN_RULES:
            count = _orphan_count(conn, child, child_col, parent, parent_col)
            if count is not None:
                orphan_rows.append({"relation": label, "child": child, "column": child_col, "count": count})
        report["orphans"] = orphan_rows
        report["orphan_total"] = sum(int(x["count"]) for x in orphan_rows)

        duplicates = {
            "paper_source_id": _group_duplicates(conn, "paper_shelf", "source_id", limit=20),
            "paper_title": _group_duplicates(conn, "paper_shelf", "title", normalize=True, limit=20),
            "research_question": _group_duplicates(conn, "research_questions", "question", normalize=True, where="deleted_at IS NULL", limit=20),
            "ontology_type_name": _group_duplicates(conn, "ontology_types", "name", normalize=True, where="deleted_at IS NULL", limit=20),
        }
        report["duplicates"] = duplicates
        report["duplicate_group_total"] = sum(len(rows) for rows in duplicates.values())

        status_tables: dict[str, dict[str, int]] = {}
        for table, column in (
            ("research_questions", "status"),
            ("ontology_change_reviews", "status"),
            ("auto_research_runs", "status"),
            ("auto_research_failures", "status"),
        ):
            if table in tables and column in _columns(conn, table):
                try:
                    rows = conn.execute(
                        f"SELECT COALESCE({column}, ''), COUNT(*) FROM {_quote_identifier(table)} GROUP BY {column} ORDER BY COUNT(*) DESC"
                    ).fetchall()
                    status_tables[table] = {str(row[0] or "(blank)"): int(row[1]) for row in rows}
                except sqlite3.DatabaseError:
                    pass
        report["status_counts"] = status_tables
        report["assets"] = _asset_audit(conn, root)
        report["research_recovery"] = _paper_recovery_audit(conn)

        issues: list[dict[str, Any]] = []
        if report["integrity"] != "ok":
            issues.append({"kind": "database", "severity": "high", "detail": "SQLite integrity_check did not return ok."})
        if report["unclassified_tables"]:
            issues.append({"kind": "schema", "severity": "medium", "detail": f"Unclassified durable tables: {len(report['unclassified_tables'])}"})
        if report["orphan_total"]:
            issues.append({"kind": "orphan", "severity": "medium", "detail": f"Orphan references: {report['orphan_total']}"})
        if report["duplicate_group_total"]:
            issues.append({"kind": "duplicate", "severity": "low", "detail": f"Duplicate candidate groups: {report['duplicate_group_total']}"})
        assets = report["assets"]
        if assets["missing_pdf_rows"]:
            issues.append({"kind": "asset", "severity": "medium", "detail": f"Missing referenced PDFs: {assets['missing_pdf_rows']}"})
        if assets["unreferenced_pdf_files"]:
            issues.append({"kind": "asset", "severity": "low", "detail": f"Unreferenced PDFs under data dir: {assets['unreferenced_pdf_files']}"})
        if assets["same_size_candidate_groups"]:
            issues.append({"kind": "asset", "severity": "info", "detail": f"Same-size PDF duplicate candidates: {assets['same_size_candidate_groups']} groups"})
        report["issues"] = issues
        return report
    finally:
        conn.close()
