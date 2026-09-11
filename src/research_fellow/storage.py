from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain.phenomena import PhenomenonDraft, validate_payload


PHENOMENON_TYPES = {
    "research_update",
    "advice_report",
    "decision_request",
    "decision",
    "curation_intent",
    "knowledge_update",
    "advisory_exchange",
    "activity_summary",
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """SQLite record of cross-domain phenomena, not a replacement for knowledge memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    case_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS phenomena (
                    phenomenon_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    phenomenon_type TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    recipients_json TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    phenomenon_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    decided_by TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    FOREIGN KEY(phenomenon_id) REFERENCES phenomena(phenomenon_id)
                );
                CREATE TABLE IF NOT EXISTS search_profiles (
                    profile_id TEXT PRIMARY KEY,
                    intent_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    core_terms_json TEXT NOT NULL DEFAULT '[]',
                    cadence TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    query TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(profile_id) REFERENCES search_profiles(profile_id)
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    response TEXT NOT NULL,
                    error TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_cards (
                    card_id TEXT PRIMARY KEY,
                    card_json TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_cards_active
                    ON knowledge_cards(deleted_at, updated_at DESC);
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_card_id TEXT NOT NULL,
                    target_card_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_source ON knowledge_relations(source_card_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_target ON knowledge_relations(target_card_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_active ON knowledge_relations(deleted_at, relation_type);
                CREATE TABLE IF NOT EXISTS paper_shelf (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    publication_year TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    pdf_path TEXT NOT NULL DEFAULT '',
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    shelf_status TEXT NOT NULL DEFAULT 'reference',
                    reading_status TEXT NOT NULL DEFAULT 'unread',
                    asset_type TEXT NOT NULL DEFAULT 'paper',
                    intake_source TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_shelf_source_id
                    ON paper_shelf(source_id) WHERE source_id <> '';
                CREATE INDEX IF NOT EXISTS idx_paper_shelf_status
                    ON paper_shelf(shelf_status, reading_status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS paper_abstracts (
                    paper_id TEXT PRIMARY KEY,
                    abstract TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_analyses (
                    paper_id TEXT PRIMARY KEY,
                    research_question TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    reading_raw_output TEXT NOT NULL DEFAULT '',
                    researcher_note TEXT NOT NULL DEFAULT '',
                    generated_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_card_links (
                    paper_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    linked_at TEXT NOT NULL,
                    PRIMARY KEY(paper_id, card_id),
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_reading_questions (
                    question_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question TEXT NOT NULL,
                    tentative_answer TEXT NOT NULL, evidence_json TEXT NOT NULL, uncertainty TEXT NOT NULL,
                    research_relevance TEXT NOT NULL DEFAULT '', suggested_ontology TEXT NOT NULL DEFAULT '',
                    suggested_labels TEXT NOT NULL DEFAULT '', suggested_title TEXT NOT NULL DEFAULT '', suggested_concepts TEXT NOT NULL DEFAULT '',
                    suggested_applies_to TEXT NOT NULL DEFAULT '', suggested_conditions TEXT NOT NULL DEFAULT '',
                    suggested_limits TEXT NOT NULL DEFAULT '', suggested_context TEXT NOT NULL DEFAULT '',
                    suggested_implication TEXT NOT NULL DEFAULT '', suggested_source_excerpt TEXT NOT NULL DEFAULT '',
                    researcher_comment TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'proposed',
                    promotion_request_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_ontology_candidates (
                    candidate_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, question_id TEXT,
                    candidate_text TEXT NOT NULL, evidence_json TEXT NOT NULL,
                    researcher_comment TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'proposed',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS ontology_facets (
                    facet_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ontology_facets_name
                    ON ontology_facets(lower(name)) WHERE deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS ontology_types (
                    type_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    facet_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(facet_id) REFERENCES ontology_facets(facet_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ontology_types_name
                    ON ontology_types(lower(name)) WHERE deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS ontology_card_assignments (
                    type_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    assigned_at TEXT NOT NULL,
                    PRIMARY KEY(type_id, card_id),
                    FOREIGN KEY(type_id) REFERENCES ontology_types(type_id)
                );
                CREATE INDEX IF NOT EXISTS idx_ontology_card_assignments_card
                    ON ontology_card_assignments(card_id);
                CREATE TABLE IF NOT EXISTS ontology_type_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_type_id TEXT NOT NULL,
                    target_type_id TEXT NOT NULL,
                    relation_name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    FOREIGN KEY(source_type_id) REFERENCES ontology_types(type_id),
                    FOREIGN KEY(target_type_id) REFERENCES ontology_types(type_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_ontology_type_relation_unique
                    ON ontology_type_relations(source_type_id, target_type_id, lower(relation_name))
                    WHERE deleted_at IS NULL;
                CREATE TABLE IF NOT EXISTS research_questions (
                    rq_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    rationale TEXT NOT NULL DEFAULT '',
                    gap_or_tension TEXT NOT NULL DEFAULT '',
                    research_context TEXT NOT NULL DEFAULT '',
                    exploration_need TEXT NOT NULL DEFAULT '',
                    source_card_ids_json TEXT NOT NULL DEFAULT '[]',
                    source_update_ids_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'candidate',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_research_questions_unique
                    ON research_questions(lower(question)) WHERE deleted_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_research_questions_status
                    ON research_questions(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS research_question_intents (
                    rq_id TEXT NOT NULL,
                    intent_id TEXT NOT NULL,
                    request_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(rq_id, intent_id),
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE TABLE IF NOT EXISTS research_state_reviews (
                    review_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_card_count INTEGER NOT NULL DEFAULT 0,
                    generated_rq_count INTEGER NOT NULL DEFAULT 0,
                    selected_rq_count INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS research_state_review_cards (
                    review_id TEXT NOT NULL,
                    update_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(review_id, update_id),
                    FOREIGN KEY(review_id) REFERENCES research_state_reviews(review_id)
                );
                CREATE INDEX IF NOT EXISTS idx_research_state_review_cards_card
                    ON research_state_review_cards(card_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS research_state_review_questions (
                    review_id TEXT NOT NULL,
                    rq_id TEXT NOT NULL,
                    change_kind TEXT NOT NULL DEFAULT 'new',
                    priority_score INTEGER,
                    selection_reason TEXT NOT NULL DEFAULT '',
                    selected INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(review_id, rq_id),
                    FOREIGN KEY(review_id) REFERENCES research_state_reviews(review_id),
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE TABLE IF NOT EXISTS research_question_sources (
                    rq_id TEXT NOT NULL,
                    review_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    update_id TEXT NOT NULL DEFAULT '',
                    relation_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(rq_id, review_id, card_id),
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id),
                    FOREIGN KEY(review_id) REFERENCES research_state_reviews(review_id)
                );
                CREATE INDEX IF NOT EXISTS idx_research_question_sources_rq
                    ON research_question_sources(rq_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS research_question_changes (
                    change_id TEXT PRIMARY KEY,
                    rq_id TEXT NOT NULL,
                    review_id TEXT NOT NULL DEFAULT '',
                    change_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE INDEX IF NOT EXISTS idx_research_question_changes_rq
                    ON research_question_changes(rq_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS research_question_threads (
                    rq_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL DEFAULT 'm1_knowledge',
                    source_payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE TABLE IF NOT EXISTS research_question_versions (
                    version_id TEXT PRIMARY KEY,
                    rq_id TEXT NOT NULL,
                    version_no INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    change_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE(rq_id, version_no),
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE TABLE IF NOT EXISTS m2_reports (
                    report_id TEXT PRIMARY KEY,
                    rq_id TEXT NOT NULL,
                    case_id TEXT NOT NULL DEFAULT '',
                    report_type TEXT NOT NULL DEFAULT 'research_review',
                    generation_mode TEXT NOT NULL DEFAULT 'internal_llm',
                    question_text TEXT NOT NULL,
                    context_text TEXT NOT NULL DEFAULT '',
                    evidence_card_ids_json TEXT NOT NULL DEFAULT '[]',
                    report_text TEXT NOT NULL,
                    knowledge_gaps TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    deleted_at TEXT,
                    FOREIGN KEY(rq_id) REFERENCES research_questions(rq_id)
                );
                CREATE INDEX IF NOT EXISTS idx_m2_reports_rq ON m2_reports(rq_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS sensemaking_threads (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    linked_rq_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_sensemaking_threads_status
                    ON sensemaking_threads(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS sensemaking_turns (
                    turn_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence_card_ids_json TEXT NOT NULL DEFAULT '[]',
                    quick_papers_json TEXT NOT NULL DEFAULT '[]',
                    generation_mode TEXT NOT NULL DEFAULT 'internal_llm',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES sensemaking_threads(thread_id)
                );
                CREATE INDEX IF NOT EXISTS idx_sensemaking_turns_thread
                    ON sensemaking_turns(thread_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS thread_current_states (
                    thread_kind TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    current_question TEXT NOT NULL DEFAULT '',
                    body_text TEXT NOT NULL,
                    generation_mode TEXT NOT NULL DEFAULT 'internal_llm',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(thread_kind, thread_id)
                );
                CREATE TABLE IF NOT EXISTS thread_report_snapshots (
                    report_id TEXT PRIMARY KEY,
                    thread_kind TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body_text TEXT NOT NULL,
                    generation_mode TEXT NOT NULL DEFAULT 'internal_llm',
                    created_at TEXT NOT NULL,
                    archived_at TEXT,
                    deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_thread_report_snapshots_thread
                    ON thread_report_snapshots(thread_kind, thread_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS auto_research_runs (
                    run_id TEXT PRIMARY KEY,
                    review_id TEXT NOT NULL DEFAULT '',
                    intent_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'running',
                    current_stage TEXT NOT NULL DEFAULT '',
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    last_error_type TEXT NOT NULL DEFAULT '',
                    last_error_message TEXT NOT NULL DEFAULT '',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_auto_research_runs_status
                    ON auto_research_runs(status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS auto_research_failures (
                    failure_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL DEFAULT '',
                    review_id TEXT NOT NULL DEFAULT '',
                    intent_id TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL,
                    item_key TEXT NOT NULL DEFAULT '',
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL DEFAULT '',
                    recommended_action TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'needs_attention',
                    context_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_auto_research_failures_status
                    ON auto_research_failures(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS manual_recovery_attempts (
                    recovery_id TEXT PRIMARY KEY,
                    failure_id TEXT NOT NULL,
                    run_id TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL,
                    item_key TEXT NOT NULL DEFAULT '',
                    response_text TEXT NOT NULL,
                    validation_status TEXT NOT NULL,
                    validation_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    applied_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_manual_recovery_attempts_failure
                    ON manual_recovery_attempts(failure_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS paper_reading_reviews (
                    review_id TEXT PRIMARY KEY, question_id TEXT NOT NULL, refined_answer TEXT NOT NULL,
                    additional_evidence_json TEXT NOT NULL, remaining_uncertainty TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES paper_reading_questions(question_id)
                );
                CREATE TABLE IF NOT EXISTS paper_asset_events (
                    event_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, event_type TEXT NOT NULL,
                    detail_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_paper_asset_events_paper ON paper_asset_events(paper_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS episode_memories (
                    episode_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    episode_type TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    situation_summary TEXT NOT NULL,
                    decision_question TEXT NOT NULL,
                    advisory_plan_json TEXT NOT NULL,
                    answer_summary TEXT NOT NULL,
                    conditions_and_limits TEXT NOT NULL,
                    unresolved_items_json TEXT NOT NULL,
                    follow_up_intent_ids_json TEXT NOT NULL,
                    evidence_card_ids_json TEXT NOT NULL,
                    evidence_relation_ids_json TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                );
                CREATE INDEX IF NOT EXISTS idx_episode_memories_recall
                    ON episode_memories(lifecycle_status, episode_type, updated_at DESC);
                """
            )
            # Safe for v0.1 databases: SQLite preserves all existing rows.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(search_profiles)").fetchall()}
            if "core_terms_json" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN core_terms_json TEXT NOT NULL DEFAULT '[]'")
            if "deleted_at" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN deleted_at TEXT")
            if "deleted_note" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN deleted_note TEXT NOT NULL DEFAULT ''")
            paper_columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_shelf)").fetchall()}
            if "labels_json" not in paper_columns:
                conn.execute("ALTER TABLE paper_shelf ADD COLUMN labels_json TEXT NOT NULL DEFAULT '[]'")
            if "asset_type" not in paper_columns:
                conn.execute("ALTER TABLE paper_shelf ADD COLUMN asset_type TEXT NOT NULL DEFAULT 'paper'")
            if "intake_source" not in paper_columns:
                conn.execute("ALTER TABLE paper_shelf ADD COLUMN intake_source TEXT NOT NULL DEFAULT 'manual'")
            analysis_columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_analyses)").fetchall()}
            if "reading_raw_output" not in analysis_columns:
                conn.execute("ALTER TABLE paper_analyses ADD COLUMN reading_raw_output TEXT NOT NULL DEFAULT ''")
            reading_question_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(paper_reading_questions)").fetchall()
            }
            if "research_relevance" not in reading_question_columns:
                conn.execute("ALTER TABLE paper_reading_questions ADD COLUMN research_relevance TEXT NOT NULL DEFAULT ''")
            if "suggested_ontology" not in reading_question_columns:
                conn.execute("ALTER TABLE paper_reading_questions ADD COLUMN suggested_ontology TEXT NOT NULL DEFAULT ''")
            for column in ("suggested_labels", "suggested_title", "suggested_concepts", "suggested_applies_to", "suggested_conditions", "suggested_limits", "suggested_context", "suggested_implication", "suggested_source_excerpt"):
                if column not in reading_question_columns:
                    conn.execute(f"ALTER TABLE paper_reading_questions ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            ontology_type_columns = {row[1] for row in conn.execute("PRAGMA table_info(ontology_types)").fetchall()}
            if "facet_id" not in ontology_type_columns:
                conn.execute("ALTER TABLE ontology_types ADD COLUMN facet_id TEXT")
            conn.execute("INSERT OR REPLACE INTO schema_meta VALUES (?, ?)", ("schema_version", "16"))
            duplicates = conn.execute(
                "SELECT phenomenon_id FROM decisions GROUP BY phenomenon_id HAVING COUNT(*) > 1"
            ).fetchone()
            # A legacy database may already contain duplicate decisions. Preserve it
            # untouched; guarded transitions still prevent any new duplication.
            if duplicates is None:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS decisions_one_per_request "
                    "ON decisions(phenomenon_id)"
                )

    def record_llm_call(self, *, profile_name: str, model: str, prompt: str, settings: dict[str, Any], response: str | None, error: str | None, diagnostics: dict[str, Any] | None) -> None:
        """Append-only diagnostic record. It never changes research knowledge."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"llm-{uuid.uuid4().hex[:12]}", now(), profile_name, model, prompt,
                 json.dumps(settings, ensure_ascii=False), response or "", error or "", json.dumps(diagnostics or {}, ensure_ascii=False)),
            )

    def llm_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["settings"] = json.loads(item.pop("settings_json"))
            item["diagnostics"] = json.loads(item.pop("diagnostics_json"))
            result.append(item)
        return result

    def clear_llm_calls(self) -> int:
        """Remove only diagnostic LLM records; research knowledge and decisions remain intact."""
        with self.connect() as conn:
            result = conn.execute("DELETE FROM llm_calls")
        return result.rowcount

    def create_case(self, case_type: str, title: str, status: str = "open") -> str:
        case_id = f"case-{uuid.uuid4().hex[:12]}"
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, case_type, title, status, timestamp, timestamp),
            )
        return case_id

    def record(
        self,
        case_id: str,
        phenomenon_type: str,
        producer: str,
        recipients: list[str],
        subject_type: str,
        payload: dict[str, Any],
        subject_id: str | None = None,
        status: str = "proposed",
    ) -> str:
        if phenomenon_type not in PHENOMENON_TYPES:
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
        normalized_payload = validate_payload(phenomenon_type, payload)
        PhenomenonDraft(
            phenomenon_type=phenomenon_type, producer=producer, recipients=recipients,
            subject_type=subject_type, subject_id=subject_id, payload=normalized_payload, status=status,
        )
        phenomenon_id = f"ph-{uuid.uuid4().hex[:12]}"
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO phenomena VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    phenomenon_id,
                    case_id,
                    phenomenon_type,
                    producer,
                    json.dumps(recipients, ensure_ascii=False),
                    subject_type,
                    subject_id,
                    json.dumps(normalized_payload, ensure_ascii=False),
                    status,
                    timestamp,
                ),
            )
            conn.execute("UPDATE cases SET updated_at=? WHERE case_id=?", (timestamp, case_id))
        return phenomenon_id

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["recipients"] = json.loads(result.pop("recipients_json"))
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def phenomena(self, *, recipient: str | None = None, type_: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        clauses, values = [], []
        if type_:
            clauses.append("phenomenon_type=?")
            values.append(type_)
        if status:
            clauses.append("status=?")
            values.append(status)
        query = "SELECT * FROM phenomena"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            records = [self._row(row) for row in conn.execute(query, values).fetchall()]
        return [row for row in records if not recipient or recipient in row["recipients"]]

    def case(self, case_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return dict(row) if row else None

    def phenomenon(self, phenomenon_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM phenomena WHERE phenomenon_id=?", (phenomenon_id,)).fetchone()
        return self._row(row) if row else None

    def timeline(self, case_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM phenomena WHERE case_id=? ORDER BY created_at ASC", (case_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def cases(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM cases ORDER BY updated_at DESC").fetchall()]

    def upsert_episode_memory(self, episode: dict[str, Any]) -> None:
        """Store a recall projection without altering the immutable case timeline."""
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO episode_memories
                   (episode_id, case_id, episode_type, lifecycle_status, situation_summary, decision_question,
                    advisory_plan_json, answer_summary, conditions_and_limits, unresolved_items_json,
                    follow_up_intent_ids_json, evidence_card_ids_json, evidence_relation_ids_json, outcome,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(episode_id) DO UPDATE SET
                    lifecycle_status=excluded.lifecycle_status, answer_summary=excluded.answer_summary,
                    conditions_and_limits=excluded.conditions_and_limits,
                    unresolved_items_json=excluded.unresolved_items_json,
                    follow_up_intent_ids_json=excluded.follow_up_intent_ids_json,
                    evidence_card_ids_json=excluded.evidence_card_ids_json,
                    evidence_relation_ids_json=excluded.evidence_relation_ids_json,
                    outcome=excluded.outcome, updated_at=excluded.updated_at""",
                (
                    episode["episode_id"], episode["case_id"], episode["episode_type"], episode.get("lifecycle_status", "provisional"),
                    episode["situation_summary"], episode["decision_question"], json.dumps(episode.get("advisory_plan", []), ensure_ascii=False),
                    episode.get("answer_summary", ""), episode.get("conditions_and_limits", ""),
                    json.dumps(episode.get("unresolved_items", []), ensure_ascii=False),
                    json.dumps(episode.get("follow_up_intent_ids", []), ensure_ascii=False),
                    json.dumps(episode.get("evidence_card_ids", []), ensure_ascii=False),
                    json.dumps(episode.get("evidence_relation_ids", []), ensure_ascii=False), episode.get("outcome", ""),
                    timestamp, timestamp,
                ),
            )

    def episode_memories(self, *, case_id: str | None = None) -> list[dict[str, Any]]:
        query, values = "SELECT * FROM episode_memories", []
        if case_id:
            query += " WHERE case_id=?"
            values.append(case_id)
        query += " ORDER BY updated_at DESC"
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("advisory_plan", "unresolved_items", "follow_up_intent_ids", "evidence_card_ids", "evidence_relation_ids"):
                item[key] = json.loads(item.pop(f"{key}_json"))
            result.append(item)
        return result

    def update_episode_memory_outcome(self, episode_id: str, lifecycle_status: str, outcome: str = "") -> bool:
        if lifecycle_status not in {"provisional", "confirmed", "superseded"}:
            raise ValueError("지원하지 않는 일화 메모리 상태입니다.")
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE episode_memories SET lifecycle_status=?, outcome=?, updated_at=? WHERE episode_id=?",
                (lifecycle_status, outcome, now(), episode_id),
            )
        return result.rowcount == 1

    def set_status(self, phenomenon_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE phenomena SET status=? WHERE phenomenon_id=?", (status, phenomenon_id))

    def transition(self, phenomenon_id: str, expected_status: str, next_status: str) -> bool:
        """Compare-and-set transition used by application services for idempotency."""
        allowed = {
            "proposed": {"approved", "deferred", "rejected"},
            "ready": {"completed", "failed"},
        }
        if next_status not in allowed.get(expected_status, set()):
            raise ValueError(f"Invalid status transition: {expected_status} → {next_status}")
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE phenomena SET status=? WHERE phenomenon_id=? AND status=?",
                (next_status, phenomenon_id, expected_status),
            )
        return result.rowcount == 1

    def add_decision(self, phenomenon_id: str, decision: str, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                (f"dec-{uuid.uuid4().hex[:12]}", phenomenon_id, decision, note, "researcher", now()),
            )

    def decisions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM decisions ORDER BY decided_at DESC").fetchall()]


    def prepare_intent_for_retry(self, intent_id: str) -> dict[str, Any] | None:
        """Reset one failed auto-dispatched M1 intent to a runnable state."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM phenomena WHERE phenomenon_type='curation_intent' AND subject_id=? ORDER BY created_at DESC LIMIT 1",
                (intent_id,),
            ).fetchone()
            if not row:
                return None
            event = self._row(row)
            if event.get("status") == "failed":
                conn.execute("UPDATE phenomena SET status='ready' WHERE phenomenon_id=?", (event["phenomenon_id"],))
                event["status"] = "ready"
            profile = conn.execute("SELECT profile_id FROM search_profiles WHERE intent_id=?", (intent_id,)).fetchone()
            if profile:
                conn.execute(
                    "UPDATE search_profiles SET is_active=1, deleted_at=NULL, deleted_note='', updated_at=? WHERE intent_id=?",
                    (now(), intent_id),
                )
        return event

    def create_search_profile(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Create one editable M1 search profile when a researcher approves an Intent."""
        intent_id = str(intent["intent_id"])
        timestamp = now()
        profile = {
            "profile_id": f"sp-{uuid.uuid4().hex[:12]}", "intent_id": intent_id,
            "title": str(intent["title"]), "question": str(intent["question"]),
            "context": "\n".join(filter(None, [
                f"Research context: {intent.get('research_context') or intent['question']}",
                f"탐색 목적: {intent.get('purpose', '')}", f"연구 질문: {intent['question']}",
                f"기대 근거: {intent.get('expected_evidence', '')}", f"완료 조건: {intent.get('completion_condition', '')}",
            ])),
            # arXiv receives only English search terms. Korean labels remain in
            # the Intent context and M2 translates them on the profile screen.
            "keywords": [str(item) for item in intent.get("labels", []) if _is_english_search_term(str(item))],
            "core_terms": [],
            "cadence": "daily", "is_active": True, "created_at": timestamp, "updated_at": timestamp, "last_run_at": None,
        }
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM search_profiles WHERE intent_id=?", (intent_id,)).fetchone()
            if row:
                return self._search_profile_row(row)
            conn.execute(
                "INSERT INTO search_profiles (profile_id, intent_id, title, question, context, keywords_json, core_terms_json, cadence, is_active, created_at, updated_at, last_run_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile["profile_id"], profile["intent_id"], profile["title"], profile["question"], profile["context"],
                 json.dumps(profile["keywords"], ensure_ascii=False), json.dumps(profile["core_terms"], ensure_ascii=False), profile["cadence"], int(profile["is_active"]),
                 timestamp, timestamp, None),
            )
        return profile

    @staticmethod
    def _search_profile_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["keywords"] = json.loads(result.pop("keywords_json"))
        result["core_terms"] = json.loads(result.pop("core_terms_json", "[]"))
        result["is_active"] = bool(result["is_active"])
        return result

    def search_profiles(self, *, active_only: bool = False, include_deleted: bool = False) -> list[dict[str, Any]]:
        conditions = [] if include_deleted else ["deleted_at IS NULL"]
        if active_only:
            conditions.append("is_active=1")
        query = "SELECT * FROM search_profiles" + (" WHERE " + " AND ".join(conditions) if conditions else "") + " ORDER BY updated_at DESC"
        with self.connect() as conn:
            return [self._search_profile_row(row) for row in conn.execute(query).fetchall()]

    def update_search_profile(self, profile_id: str, *, context: str, keywords: list[str], cadence: str, is_active: bool, core_terms: list[str] | None = None) -> None:
        if cadence not in {"daily", "weekly", "manual"}:
            raise ValueError("지원하지 않는 탐색 주기입니다.")
        cleaned = [item.strip() for item in keywords if item.strip()][:12]
        if not cleaned:
            raise ValueError("탐색 키워드를 한 개 이상 지정하세요.")
        if invalid := [item for item in cleaned if not _is_english_search_term(item)]:
            raise ValueError(f"arXiv 탐색 키워드는 영어로만 입력하세요: {', '.join(invalid[:3])}")
        core_terms = [item.strip() for item in (core_terms or []) if item.strip()][:5]
        if invalid := [item for item in core_terms if not _is_english_search_term(item)]:
            raise ValueError(f"핵심 검색어는 영어로만 입력하세요: {', '.join(invalid[:3])}")
        with self.connect() as conn:
            conn.execute(
                "UPDATE search_profiles SET context=?, keywords_json=?, core_terms_json=?, cadence=?, is_active=?, updated_at=? WHERE profile_id=?",
                (context.strip(), json.dumps(cleaned, ensure_ascii=False), json.dumps(core_terms, ensure_ascii=False), cadence, int(is_active), now(), profile_id),
            )

    def complete_search_profile(self, profile_id: str) -> None:
        """Remove an executed Intent from the runnable queue but keep its history."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE search_profiles SET is_active=0, updated_at=? WHERE profile_id=?",
                (now(), profile_id),
            )

    def delete_search_profile(self, profile_id: str, note: str = "연구자가 탐색 큐에서 삭제") -> bool:
        """Hide an Intent from queue/history while retaining it and its runs for audit."""
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE search_profiles SET is_active=0, deleted_at=?, deleted_note=?, updated_at=? WHERE profile_id=? AND deleted_at IS NULL",
                (now(), note, now(), profile_id),
            )
        return result.rowcount == 1

    def record_search_run(self, profile_id: str, trigger: str, query: str, candidates: list[dict[str, Any]], status: str, error: str = "") -> str:
        run_id, timestamp = f"sr-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO search_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, profile_id, trigger, query, json.dumps(candidates, ensure_ascii=False), status, error, timestamp),
            )
            conn.execute("UPDATE search_profiles SET last_run_at=?, updated_at=? WHERE profile_id=?", (timestamp, timestamp, profile_id))
        return run_id

    def search_runs(self, profile_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM search_runs WHERE profile_id=? ORDER BY created_at DESC LIMIT ?", (profile_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["candidates"] = json.loads(item.pop("candidates_json"))
            result.append(item)
        return result

    def update_search_run_candidates(self, run_id: str, candidates: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE search_runs SET candidates_json=? WHERE run_id=?", (json.dumps(candidates, ensure_ascii=False), run_id))

    def upsert_knowledge_relation(self, relation: dict[str, Any]) -> None:
        """Persist the approved relation projection used by graph traversal.

        The append-only JSONL relation record remains the audit artifact; this
        table is the indexed, mutable projection for local graph queries.
        """
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO knowledge_relations
                   (relation_id, source_card_id, target_card_id, relation_type, evidence, conditions, confidence, approved_at, deleted_at, deleted_note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '')
                   ON CONFLICT(relation_id) DO NOTHING""",
                (str(relation["relation_id"]), str(relation["source_card_id"]), str(relation["target_card_id"]),
                 str(relation["relation_type"]), str(relation["evidence"]), str(relation["conditions"]),
                 str(relation["confidence"]), str(relation.get("approved_at") or now())),
            )

    def sync_knowledge_relations(self, relations: list[dict[str, Any]]) -> None:
        for relation in relations:
            self.upsert_knowledge_relation(relation)

    def active_knowledge_relations(self, card_ids: set[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM knowledge_relations WHERE deleted_at IS NULL"
        values: list[object] = []
        if card_ids is not None:
            if not card_ids:
                return []
            placeholders = ", ".join("?" for _ in card_ids)
            query += f" AND source_card_id IN ({placeholders}) AND target_card_id IN ({placeholders})"
            values.extend(sorted(card_ids))
            values.extend(sorted(card_ids))
        query += " ORDER BY approved_at DESC"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, values).fetchall()]



    # --- Auto research retry / failure state ------------------------------

    def create_auto_research_run(self, *, review_id: str = "", intent_id: str = "", stage: str = "") -> str:
        run_id, timestamp = f"arr-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO auto_research_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, review_id, intent_id, "running", stage, "{}", "", "", 0, timestamp, timestamp, None),
            )
        return run_id

    def update_auto_research_run(self, run_id: str, *, status: str | None = None, stage: str | None = None, checkpoint: dict[str, Any] | None = None, error_type: str | None = None, error_message: str | None = None, retry_count: int | None = None) -> None:
        fields, values = [], []
        mapping = {
            "status": status, "current_stage": stage,
            "checkpoint_json": json.dumps(checkpoint, ensure_ascii=False) if checkpoint is not None else None,
            "last_error_type": error_type, "last_error_message": error_message,
            "retry_count": retry_count,
        }
        for key, value in mapping.items():
            if value is not None:
                fields.append(f"{key}=?"); values.append(value)
        fields.append("updated_at=?"); values.append(now())
        if status == "completed":
            fields.append("completed_at=?"); values.append(now())
        values.append(run_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE auto_research_runs SET {', '.join(fields)} WHERE run_id=?", values)

    def auto_research_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM auto_research_runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        item = dict(row); item["checkpoint"] = json.loads(item.pop("checkpoint_json") or "{}")
        return item

    def record_auto_research_failure(self, payload: dict[str, Any], *, run_id: str = "", review_id: str = "", intent_id: str = "", item_key: str = "") -> str:
        failure_id = str(payload.get("failure_id") or f"arf-{uuid.uuid4().hex[:12]}")
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO auto_research_failures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (failure_id, run_id, review_id, intent_id, str(payload.get("stage", "")), item_key,
                 int(payload.get("attempt_count", 0)), str(payload.get("error_type", "unexpected_error")),
                 str(payload.get("error_message", "")), str(payload.get("recommended_action", "")),
                 "needs_attention", json.dumps(payload.get("context") or {}, ensure_ascii=False), now(), None),
            )
        return failure_id

    def auto_research_failures(self, *, status: str = "needs_attention", limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM auto_research_failures WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, max(1, min(int(limit), 200))),
            ).fetchall()
        result=[]
        for row in rows:
            item=dict(row); item["context"] = json.loads(item.pop("context_json") or "{}"); result.append(item)
        return result

    def resolve_auto_research_failure(self, failure_id: str, *, status: str = "resolved") -> None:
        with self.connect() as conn:
            conn.execute("UPDATE auto_research_failures SET status=?, resolved_at=? WHERE failure_id=?", (status, now(), failure_id))

    def set_manual_recovery_override(self, run_id: str, *, stage: str, response: str, item_key: str = "") -> None:
        run = self.auto_research_run(run_id)
        if not run:
            raise ValueError(f"자동 연구 run을 찾지 못했습니다: {run_id}")
        checkpoint = dict(run.get("checkpoint") or {})
        overrides = dict(checkpoint.get("manual_recovery_overrides") or {})
        key = f"{stage}:{item_key}"
        overrides[key] = response
        checkpoint["manual_recovery_overrides"] = overrides
        self.update_auto_research_run(run_id, checkpoint=checkpoint)

    def manual_recovery_override(self, run_id: str, *, stage: str, item_key: str = "") -> str:
        run = self.auto_research_run(run_id)
        checkpoint = dict((run or {}).get("checkpoint") or {})
        overrides = dict(checkpoint.get("manual_recovery_overrides") or {})
        return str(overrides.get(f"{stage}:{item_key}") or "")

    def clear_manual_recovery_override(self, run_id: str, *, stage: str, item_key: str = "") -> None:
        run = self.auto_research_run(run_id)
        if not run:
            return
        checkpoint = dict(run.get("checkpoint") or {})
        overrides = dict(checkpoint.get("manual_recovery_overrides") or {})
        overrides.pop(f"{stage}:{item_key}", None)
        checkpoint["manual_recovery_overrides"] = overrides
        self.update_auto_research_run(run_id, checkpoint=checkpoint)

    def record_manual_recovery_attempt(self, failure_id: str, *, run_id: str, stage: str, item_key: str = "", response_text: str, validation_status: str, validation_message: str = "", applied: bool = False) -> str:
        recovery_id = f"mra-{uuid.uuid4().hex[:12]}"
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO manual_recovery_attempts VALUES (?,?,?,?,?,?,?,?,?,?)",
                (recovery_id, failure_id, run_id, stage, item_key, response_text, validation_status, validation_message, timestamp, timestamp if applied else None),
            )
        return recovery_id

    def manual_recovery_attempts(self, failure_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM manual_recovery_attempts WHERE failure_id=? ORDER BY created_at DESC",
                (failure_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Research-state review batches ------------------------------------

    def reviewed_knowledge_update_ids(self) -> set[str]:
        """Knowledge-update events consumed by a completed M2 state review."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT rc.update_id
                   FROM research_state_review_cards rc
                   JOIN research_state_reviews r ON r.review_id=rc.review_id
                   WHERE r.status='completed'"""
            ).fetchall()
        return {str(row[0]) for row in rows}

    def create_research_state_review(self, mode: str, updates: list[dict[str, Any]]) -> str:
        mode = mode if mode in {"manual", "auto"} else "manual"
        review_id, timestamp = f"rsr-{uuid.uuid4().hex[:12]}", now()
        card_ids = {
            str((item.get("payload") or {}).get("card_id", ""))
            for item in updates if str((item.get("payload") or {}).get("card_id", ""))
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO research_state_reviews VALUES (?,?,?,?,?,?,?,?,?)",
                (review_id, mode, "processing", len(card_ids), 0, 0, "", timestamp, None),
            )
            for item in updates:
                payload = item.get("payload") or {}
                card_id = str(payload.get("card_id", ""))
                pending_ids = item.get("pending_update_ids") or [item.get("phenomenon_id", "")]
                if not card_id:
                    continue
                for update_id in pending_ids:
                    if update_id:
                        conn.execute(
                            "INSERT OR IGNORE INTO research_state_review_cards VALUES (?,?,?,?)",
                            (review_id, str(update_id), card_id, timestamp),
                        )
        return review_id

    def complete_research_state_review(self, review_id: str, *, generated_rq_count: int, selected_rq_count: int = 0, summary: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE research_state_reviews
                   SET status='completed', generated_rq_count=?, selected_rq_count=?, summary=?, completed_at=?
                   WHERE review_id=?""",
                (generated_rq_count, selected_rq_count, summary.strip(), now(), review_id),
            )

    def fail_research_state_review(self, review_id: str, summary: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE research_state_reviews SET status='failed', summary=?, completed_at=? WHERE review_id=?",
                (summary.strip(), now(), review_id),
            )

    def rollback_research_state_review(self, review_id: str) -> None:
        """Return a completed review batch to the unreviewed pool without deleting its audit trail."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE research_state_reviews SET status='rolled_back', completed_at=? WHERE review_id=?",
                (now(), review_id),
            )

    def research_state_reviews(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_state_reviews ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def link_research_question_review(self, review_id: str, rq_id: str, change_kind: str) -> None:
        kind = change_kind if change_kind in {"new", "strengthened"} else "strengthened"
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_state_review_questions(review_id,rq_id,change_kind,priority_score,selection_reason,selected,created_at) VALUES (?,?,?,?,?,?,?)",
                (review_id, rq_id, kind, None, "", 0, now()),
            )

    def update_review_question_selection(self, review_id: str, rq_id: str, *, score: int, reason: str, selected: bool) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE research_state_review_questions SET priority_score=?, selection_reason=?, selected=? WHERE review_id=? AND rq_id=?",
                (max(1, min(int(score), 5)), reason.strip(), 1 if selected else 0, review_id, rq_id),
            )

    def research_state_review_questions(self, review_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT rrq.*, rq.question, rq.status
                   FROM research_state_review_questions rrq
                   JOIN research_questions rq ON rq.rq_id=rrq.rq_id
                   WHERE rrq.review_id=? ORDER BY rrq.selected DESC, COALESCE(rrq.priority_score,0) DESC, rrq.created_at""",
                (review_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_research_question_source(self, rq_id: str, review_id: str, card_id: str, update_id: str = "", relation_reason: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_question_sources VALUES (?,?,?,?,?,?)",
                (rq_id, review_id, card_id, update_id, relation_reason.strip(), now()),
            )

    def research_question_sources(self, rq_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT s.*, r.completed_at AS review_completed_at, r.mode AS review_mode
                   FROM research_question_sources s
                   JOIN research_state_reviews r ON r.review_id=s.review_id
                   WHERE s.rq_id=? ORDER BY s.created_at DESC""",
                (rq_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_research_question_change(self, rq_id: str, change_type: str, summary: str, *, review_id: str = "") -> str:
        change_id = f"rqc-{uuid.uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO research_question_changes VALUES (?,?,?,?,?,?)",
                (change_id, rq_id, review_id, change_type.strip() or "updated", summary.strip(), now()),
            )
        return change_id

    def research_question_changes(self, rq_id: str | None = None, *, review_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        clauses, values = [], []
        if rq_id:
            clauses.append("rq_id=?"); values.append(rq_id)
        if review_id:
            clauses.append("review_id=?"); values.append(review_id)
        query = "SELECT * FROM research_question_changes"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(max(1, min(int(limit), 500)))
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [dict(row) for row in rows]

    def latest_research_question_change(self, rq_id: str) -> dict[str, Any] | None:
        rows = self.research_question_changes(rq_id, limit=1)
        return rows[0] if rows else None

    def research_questions_for_intent(self, intent_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT rq.* FROM research_question_intents rqi
                   JOIN research_questions rq ON rq.rq_id=rqi.rq_id
                   WHERE rqi.intent_id=? AND rq.deleted_at IS NULL
                   ORDER BY rq.updated_at DESC""",
                (intent_id,),
            ).fetchall()
        return [self._research_question_row(row) for row in rows]

    # --- Research question backlog -----------------------------------------

    def upsert_research_question(self, candidate: dict[str, Any]) -> dict[str, Any]:
        question = str(candidate.get("question", "")).strip()
        rationale = str(candidate.get("rationale", "")).strip()
        if not question or not rationale:
            raise ValueError("연구질문과 도출 이유가 필요합니다.")
        timestamp = now()
        source_card_ids = list(dict.fromkeys(str(item).strip() for item in candidate.get("source_card_ids", []) if str(item).strip()))[:12]
        source_update_ids = list(dict.fromkeys(str(item).strip() for item in candidate.get("source_update_ids", []) if str(item).strip()))[:12]
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM research_questions WHERE lower(question)=lower(?) AND deleted_at IS NULL",
                (question,),
            ).fetchone()
            if existing:
                rq_id = str(existing["rq_id"])
                # Keep the researcher's lifecycle status and accumulate provenance when new M1 evidence strengthens the same RQ.
                prior_cards = json.loads(existing["source_card_ids_json"] or "[]")
                prior_updates = json.loads(existing["source_update_ids_json"] or "[]")
                merged_cards = list(dict.fromkeys([*prior_cards, *source_card_ids]))[:50]
                merged_updates = list(dict.fromkeys([*prior_updates, *source_update_ids]))[:100]
                conn.execute(
                    """UPDATE research_questions
                       SET rationale=?, gap_or_tension=?, research_context=?, exploration_need=?,
                           source_card_ids_json=?, source_update_ids_json=?, updated_at=?
                       WHERE rq_id=?""",
                    (rationale, str(candidate.get("gap_or_tension", "")).strip(),
                     str(candidate.get("research_context", "")).strip(),
                     str(candidate.get("exploration_need", "")).strip(),
                     json.dumps(merged_cards, ensure_ascii=False), json.dumps(merged_updates, ensure_ascii=False),
                     timestamp, rq_id),
                )
            else:
                rq_id = str(candidate.get("rq_id") or f"rq-{uuid.uuid4().hex[:12]}")
                status = str(candidate.get("status", "candidate"))
                if status not in {"candidate", "interested", "exploring", "hold", "rejected"}:
                    status = "candidate"
                conn.execute(
                    """INSERT INTO research_questions(
                           rq_id,question,rationale,gap_or_tension,research_context,exploration_need,
                           source_card_ids_json,source_update_ids_json,status,created_at,updated_at,deleted_at
                       ) VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)""",
                    (rq_id, question, rationale, str(candidate.get("gap_or_tension", "")).strip(),
                     str(candidate.get("research_context", "")).strip(), str(candidate.get("exploration_need", "")).strip(),
                     json.dumps(source_card_ids, ensure_ascii=False), json.dumps(source_update_ids, ensure_ascii=False),
                     status, timestamp, timestamp),
                )
            row = conn.execute("SELECT * FROM research_questions WHERE rq_id=?", (rq_id,)).fetchone()
        return self._research_question_row(row)

    @staticmethod
    def _research_question_row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        item["source_card_ids"] = json.loads(item.pop("source_card_ids_json") or "[]")
        item["source_update_ids"] = json.loads(item.pop("source_update_ids_json") or "[]")
        return item

    def research_question_by_text(self, question: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_questions WHERE lower(question)=lower(?) AND deleted_at IS NULL",
                (question.strip(),),
            ).fetchone()
        return self._research_question_row(row) if row else None

    def research_question(self, rq_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_questions WHERE rq_id=? AND deleted_at IS NULL", (rq_id,)
            ).fetchone()
        return self._research_question_row(row) if row else None

    def research_question_backlog(self, statuses: list[str] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM research_questions WHERE deleted_at IS NULL"
        values: list[object] = []
        if statuses:
            clean = [item for item in statuses if item in {"candidate", "interested", "exploring", "hold", "rejected"}]
            if clean:
                marks = ",".join("?" for _ in clean)
                query += f" AND status IN ({marks})"
                values.extend(clean)
        query += " ORDER BY CASE status WHEN 'exploring' THEN 0 WHEN 'interested' THEN 1 WHEN 'candidate' THEN 2 WHEN 'hold' THEN 3 ELSE 4 END, updated_at DESC LIMIT ?"
        values.append(max(1, min(int(limit), 500)))
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._research_question_row(row) for row in rows]

    def update_research_question_status(self, rq_id: str, status: str) -> bool:
        if status not in {"candidate", "interested", "exploring", "hold", "rejected"}:
            raise ValueError("지원하지 않는 연구질문 상태입니다.")
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM research_questions WHERE rq_id=? AND deleted_at IS NULL", (rq_id,)).fetchone()
            previous = str(row[0]) if row else ""
            result = conn.execute(
                "UPDATE research_questions SET status=?, updated_at=? WHERE rq_id=? AND deleted_at IS NULL",
                (status, now(), rq_id),
            )
        if result.rowcount == 1 and previous != status:
            self.add_research_question_change(rq_id, "status_changed", f"상태가 {previous or '미지정'} → {status}로 변경되었습니다.")
        return result.rowcount == 1

    def link_research_question_intent(self, rq_id: str, intent_id: str, request_id: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_question_intents(rq_id,intent_id,request_id,created_at) VALUES (?,?,?,?)",
                (rq_id, intent_id, request_id, now()),
            )
            conn.execute(
                "UPDATE research_questions SET status='exploring', updated_at=? WHERE rq_id=? AND deleted_at IS NULL",
                (now(), rq_id),
            )
        self.add_research_question_change(rq_id, "intent_created", f"M1 탐색 Intent {intent_id}가 연결되어 후속 탐색을 시작했습니다.")

    def research_question_intents(self, rq_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_question_intents WHERE rq_id=? ORDER BY created_at DESC", (rq_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # --- Research question threads / M2 reports -------------------------

    def ensure_research_question_thread(self, rq_id: str, *, source_type: str = "m1_knowledge", source_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        timestamp = now()
        payload = json.dumps(source_payload or {}, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO research_question_threads(rq_id,source_type,source_payload_json,created_at,updated_at)
                   VALUES (?,?,?,?,?) ON CONFLICT(rq_id) DO UPDATE SET
                   source_type=CASE WHEN excluded.source_type<>'' THEN excluded.source_type ELSE research_question_threads.source_type END,
                   source_payload_json=CASE WHEN excluded.source_payload_json<>'{}' THEN excluded.source_payload_json ELSE research_question_threads.source_payload_json END,
                   updated_at=excluded.updated_at""",
                (rq_id, source_type or "m1_knowledge", payload, timestamp, timestamp),
            )
            rq = conn.execute("SELECT question FROM research_questions WHERE rq_id=?", (rq_id,)).fetchone()
            if rq:
                exists = conn.execute("SELECT 1 FROM research_question_versions WHERE rq_id=? LIMIT 1", (rq_id,)).fetchone()
                if not exists:
                    conn.execute(
                        "INSERT INTO research_question_versions VALUES (?,?,?,?,?,?)",
                        (f"rqv-{uuid.uuid4().hex[:12]}", rq_id, 1, str(rq["question"]), "초기 질문", timestamp),
                    )
            row = conn.execute("SELECT * FROM research_question_threads WHERE rq_id=?", (rq_id,)).fetchone()
        item = dict(row) if row else {}
        if item:
            item["source_payload"] = json.loads(item.pop("source_payload_json") or "{}")
        return item

    def create_research_question_thread(self, *, question: str, source_type: str, rationale: str, research_context: str = "", source_payload: dict[str, Any] | None = None, status: str = "interested") -> dict[str, Any]:
        rq = self.upsert_research_question({
            "question": question, "rationale": rationale or f"{source_type}에서 시작된 질문",
            "research_context": research_context, "status": status,
        })
        self.ensure_research_question_thread(str(rq["rq_id"]), source_type=source_type, source_payload=source_payload or {})
        return self.research_question_thread(str(rq["rq_id"])) or rq

    def research_question_thread(self, rq_id: str) -> dict[str, Any] | None:
        rq = self.research_question(rq_id)
        if not rq:
            return None
        with self.connect() as conn:
            meta = conn.execute("SELECT * FROM research_question_threads WHERE rq_id=?", (rq_id,)).fetchone()
            versions = conn.execute("SELECT * FROM research_question_versions WHERE rq_id=? ORDER BY version_no ASC", (rq_id,)).fetchall()
        result = dict(rq)
        if meta:
            result["source_type"] = str(meta["source_type"])
            result["source_payload"] = json.loads(meta["source_payload_json"] or "{}")
        else:
            result["source_type"] = "m1_knowledge"
            result["source_payload"] = {}
        result["versions"] = [dict(row) for row in versions]
        return result

    def refine_research_question(self, rq_id: str, new_question: str, change_reason: str = "") -> bool:
        new_question = new_question.strip()
        if not new_question:
            return False
        timestamp = now()
        with self.connect() as conn:
            current = conn.execute("SELECT question FROM research_questions WHERE rq_id=? AND deleted_at IS NULL", (rq_id,)).fetchone()
            if not current:
                return False
            max_version = conn.execute("SELECT COALESCE(MAX(version_no),0) FROM research_question_versions WHERE rq_id=?", (rq_id,)).fetchone()[0]
            if max_version == 0:
                conn.execute("INSERT INTO research_question_versions VALUES (?,?,?,?,?,?)", (f"rqv-{uuid.uuid4().hex[:12]}", rq_id, 1, str(current["question"]), "초기 질문", timestamp))
                max_version = 1
            try:
                conn.execute("UPDATE research_questions SET question=?, updated_at=? WHERE rq_id=?", (new_question, timestamp, rq_id))
            except sqlite3.IntegrityError:
                raise ValueError("같은 질문이 다른 활성 Thread에 이미 있습니다.")
            conn.execute("INSERT INTO research_question_versions VALUES (?,?,?,?,?,?)", (f"rqv-{uuid.uuid4().hex[:12]}", rq_id, int(max_version)+1, new_question, change_reason.strip(), timestamp))
            conn.execute("UPDATE research_question_threads SET updated_at=? WHERE rq_id=?", (timestamp, rq_id))
        self.add_research_question_change(rq_id, "question_refined", f"질문이 구체화되었습니다: {new_question}")
        return True

    def save_m2_report(self, *, rq_id: str, report_text: str, context_text: str, evidence_card_ids: list[str], generation_mode: str, report_type: str = "research_review", case_id: str = "", knowledge_gaps: str = "") -> dict[str, Any]:
        rq = self.research_question(rq_id)
        if not rq:
            raise ValueError("연구질문 Thread를 찾을 수 없습니다.")
        report_id, timestamp = f"m2r-{uuid.uuid4().hex[:12]}", now()
        evidence = list(dict.fromkeys(str(item) for item in evidence_card_ids if str(item)))
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO m2_reports VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (report_id, rq_id, case_id, report_type, generation_mode, str(rq["question"]), context_text.strip(),
                 json.dumps(evidence, ensure_ascii=False), report_text.strip(), knowledge_gaps.strip(), "active", timestamp, None, None),
            )
            if knowledge_gaps.strip():
                conn.execute(
                    "UPDATE research_questions SET exploration_need=?, updated_at=? WHERE rq_id=? AND deleted_at IS NULL",
                    (knowledge_gaps.strip(), timestamp, rq_id),
                )
        self.add_research_question_change(rq_id, "m2_report_created", f"M2 보고서가 생성되었습니다 ({generation_mode}).")
        return self.m2_report(report_id) or {}

    @staticmethod
    def _m2_report_row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        item["evidence_card_ids"] = json.loads(item.pop("evidence_card_ids_json") or "[]")
        return item

    def m2_report(self, report_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM m2_reports WHERE report_id=?", (report_id,)).fetchone()
        return self._m2_report_row(row) if row else None

    def m2_reports(self, rq_id: str | None = None, *, include_archived: bool = True, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM m2_reports WHERE deleted_at IS NULL"
        values: list[Any] = []
        if rq_id:
            query += " AND rq_id=?"; values.append(rq_id)
        if not include_archived:
            query += " AND archived_at IS NULL"
        query += " ORDER BY created_at DESC LIMIT ?"; values.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, values).fetchall()
        return [self._m2_report_row(row) for row in rows]

    def archive_m2_report(self, report_id: str, archived: bool = True) -> bool:
        with self.connect() as conn:
            result = conn.execute("UPDATE m2_reports SET archived_at=?, status=? WHERE report_id=? AND deleted_at IS NULL", (now() if archived else None, "archived" if archived else "active", report_id))
        return result.rowcount == 1

    def delete_m2_report(self, report_id: str) -> bool:
        with self.connect() as conn:
            result = conn.execute("UPDATE m2_reports SET deleted_at=?, status='deleted' WHERE report_id=? AND deleted_at IS NULL", (now(), report_id))
        return result.rowcount == 1

    # --- Research Sensemaking ---------------------------------------------

    def create_sensemaking_thread(self, title: str, first_message: str = "") -> dict[str, Any]:
        thread_id, timestamp = f"sm-{uuid.uuid4().hex[:12]}", now()
        title = title.strip() or (first_message.strip()[:90] if first_message.strip() else "새 Sensemaking Thread")
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sensemaking_threads VALUES (?,?,?,?,?,?,NULL)",
                (thread_id, title, "active", "", timestamp, timestamp),
            )
        if first_message.strip():
            self.add_sensemaking_turn(thread_id, "user", first_message)
        return self.sensemaking_thread(thread_id) or {}

    def sensemaking_threads(self, *, include_archived: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM sensemaking_threads"
        values: list[Any] = []
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, updated_at DESC LIMIT ?"
        values.append(max(1, min(int(limit), 500)))
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, values).fetchall()]

    def sensemaking_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sensemaking_threads WHERE thread_id=?", (thread_id,)).fetchone()
        return dict(row) if row else None

    def add_sensemaking_turn(self, thread_id: str, role: str, content: str, *, evidence_card_ids: list[str] | None = None, quick_papers: list[dict[str, Any]] | None = None, generation_mode: str = "internal_llm") -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Sensemaking turn role은 user 또는 assistant여야 합니다.")
        turn_id, timestamp = f"smt-{uuid.uuid4().hex[:12]}", now()
        evidence = list(dict.fromkeys(str(x) for x in (evidence_card_ids or []) if str(x)))
        papers = list(quick_papers or [])[:20]
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sensemaking_turns VALUES (?,?,?,?,?,?,?,?)",
                (turn_id, thread_id, role, content.strip(), json.dumps(evidence, ensure_ascii=False), json.dumps(papers, ensure_ascii=False), generation_mode, timestamp),
            )
            conn.execute("UPDATE sensemaking_threads SET updated_at=? WHERE thread_id=?", (timestamp, thread_id))
        return self.sensemaking_turn(turn_id) or {}

    def sensemaking_turn(self, turn_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM sensemaking_turns WHERE turn_id=?", (turn_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["evidence_card_ids"] = json.loads(item.pop("evidence_card_ids_json") or "[]")
        item["quick_papers"] = json.loads(item.pop("quick_papers_json") or "[]")
        return item

    def sensemaking_turns(self, thread_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM sensemaking_turns WHERE thread_id=? ORDER BY created_at ASC", (thread_id,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence_card_ids"] = json.loads(item.pop("evidence_card_ids_json") or "[]")
            item["quick_papers"] = json.loads(item.pop("quick_papers_json") or "[]")
            result.append(item)
        return result

    def link_sensemaking_to_rq(self, thread_id: str, rq_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sensemaking_threads SET linked_rq_id=?, updated_at=? WHERE thread_id=?", (rq_id, now(), thread_id))

    def archive_sensemaking_thread(self, thread_id: str, archived: bool = True) -> bool:
        timestamp = now()
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE sensemaking_threads SET archived_at=?, status=?, updated_at=? WHERE thread_id=?",
                (timestamp if archived else None, "archived" if archived else "active", timestamp, thread_id),
            )
        return result.rowcount == 1

    # --- Shared thread current state / report snapshots -------------------

    def save_thread_current_state(self, *, thread_kind: str, thread_id: str, current_question: str, body_text: str, generation_mode: str = "internal_llm") -> dict[str, Any]:
        if thread_kind not in {"sensemaking", "research_question"}:
            raise ValueError("지원하지 않는 thread_kind입니다.")
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO thread_current_states(thread_kind,thread_id,current_question,body_text,generation_mode,updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(thread_kind,thread_id) DO UPDATE SET
                   current_question=excluded.current_question, body_text=excluded.body_text,
                   generation_mode=excluded.generation_mode, updated_at=excluded.updated_at""",
                (thread_kind, thread_id, current_question.strip(), body_text.strip(), generation_mode, timestamp),
            )
            row = conn.execute("SELECT * FROM thread_current_states WHERE thread_kind=? AND thread_id=?", (thread_kind, thread_id)).fetchone()
        return dict(row) if row else {}

    def thread_current_state(self, thread_kind: str, thread_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM thread_current_states WHERE thread_kind=? AND thread_id=?", (thread_kind, thread_id)).fetchone()
        return dict(row) if row else None

    def create_thread_report_snapshot(self, *, thread_kind: str, thread_id: str, title: str, body_text: str, generation_mode: str = "internal_llm") -> dict[str, Any]:
        report_id, timestamp = f"trp-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO thread_report_snapshots VALUES (?,?,?,?,?,?,?,?,NULL)",
                (report_id, thread_kind, thread_id, title.strip(), body_text.strip(), generation_mode, timestamp, None),
            )
            row = conn.execute("SELECT * FROM thread_report_snapshots WHERE report_id=?", (report_id,)).fetchone()
        return dict(row) if row else {}

    def thread_report_snapshots(self, thread_kind: str, thread_id: str, *, include_archived: bool = True, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM thread_report_snapshots WHERE thread_kind=? AND thread_id=? AND deleted_at IS NULL"
        values: list[Any] = [thread_kind, thread_id]
        if not include_archived:
            query += " AND archived_at IS NULL"
        query += " ORDER BY created_at DESC LIMIT ?"
        values.append(max(1, min(int(limit), 500)))
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, values).fetchall()]

    def archive_thread_report_snapshot(self, report_id: str, archived: bool = True) -> bool:
        with self.connect() as conn:
            result = conn.execute("UPDATE thread_report_snapshots SET archived_at=? WHERE report_id=? AND deleted_at IS NULL", (now() if archived else None, report_id))
        return result.rowcount == 1

    def delete_thread_report_snapshot(self, report_id: str) -> bool:
        with self.connect() as conn:
            result = conn.execute("UPDATE thread_report_snapshots SET deleted_at=? WHERE report_id=? AND deleted_at IS NULL", (now(), report_id))
        return result.rowcount == 1

    # --- Ontology schema -------------------------------------------------

    def create_ontology_facet(self, name: str, description: str = "") -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Facet 이름은 비어 있을 수 없습니다.")
        facet_id, timestamp = f"of-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            try:
                conn.execute("INSERT INTO ontology_facets VALUES (?, ?, ?, ?, ?, NULL)", (facet_id, name, description.strip(), timestamp, timestamp))
            except sqlite3.IntegrityError as error:
                raise ValueError("같은 이름의 활성 Facet이 이미 있습니다.") from error
            row = conn.execute("SELECT * FROM ontology_facets WHERE facet_id=?", (facet_id,)).fetchone()
        return dict(row)

    def ontology_facets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT f.*, COUNT(t.type_id) AS type_count
                FROM ontology_facets f
                LEFT JOIN ontology_types t ON t.facet_id=f.facet_id AND t.deleted_at IS NULL
                WHERE f.deleted_at IS NULL
                GROUP BY f.facet_id ORDER BY lower(f.name)""").fetchall()
        return [dict(row) for row in rows]

    def delete_ontology_facet(self, facet_id: str) -> bool:
        timestamp = now()
        with self.connect() as conn:
            result = conn.execute("UPDATE ontology_facets SET deleted_at=?, updated_at=? WHERE facet_id=? AND deleted_at IS NULL", (timestamp, timestamp, facet_id))
            if result.rowcount:
                conn.execute("UPDATE ontology_types SET facet_id=NULL, updated_at=? WHERE facet_id=? AND deleted_at IS NULL", (timestamp, facet_id))
        return result.rowcount == 1

    def create_ontology_type(self, name: str, description: str = "", facet_id: str | None = None) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("온톨로지 타입 이름은 비어 있을 수 없습니다.")
        type_id, timestamp = f"ot-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            if facet_id and not conn.execute("SELECT 1 FROM ontology_facets WHERE facet_id=? AND deleted_at IS NULL", (facet_id,)).fetchone():
                raise ValueError("활성 Facet을 찾을 수 없습니다.")
            try:
                conn.execute("INSERT INTO ontology_types(type_id,name,description,facet_id,created_at,updated_at,deleted_at) VALUES (?,?,?,?,?,?,NULL)", (type_id, name, description.strip(), facet_id, timestamp, timestamp))
            except sqlite3.IntegrityError as error:
                raise ValueError("같은 이름의 활성 온톨로지 타입이 이미 있습니다.") from error
            row = conn.execute("SELECT * FROM ontology_types WHERE type_id=?", (type_id,)).fetchone()
        return dict(row)

    def update_ontology_type(self, type_id: str, *, name: str, description: str = "", facet_id: str | None = None) -> bool:
        name = name.strip()
        if not name:
            raise ValueError("온톨로지 타입 이름은 비어 있을 수 없습니다.")
        with self.connect() as conn:
            if facet_id and not conn.execute("SELECT 1 FROM ontology_facets WHERE facet_id=? AND deleted_at IS NULL", (facet_id,)).fetchone():
                raise ValueError("활성 Facet을 찾을 수 없습니다.")
            try:
                result = conn.execute("UPDATE ontology_types SET name=?, description=?, facet_id=?, updated_at=? WHERE type_id=? AND deleted_at IS NULL", (name, description.strip(), facet_id, now(), type_id))
            except sqlite3.IntegrityError as error:
                raise ValueError("같은 이름의 활성 온톨로지 타입이 이미 있습니다.") from error
        return result.rowcount == 1

    def ontology_types(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT t.*, f.name AS facet_name, COUNT(a.card_id) AS card_count
                FROM ontology_types t
                LEFT JOIN ontology_facets f ON f.facet_id=t.facet_id AND f.deleted_at IS NULL
                LEFT JOIN ontology_card_assignments a ON a.type_id=t.type_id
                WHERE t.deleted_at IS NULL
                GROUP BY t.type_id
                ORDER BY COALESCE(lower(f.name), 'zzzz'), lower(t.name)""").fetchall()
        return [dict(row) for row in rows]

    def delete_ontology_type(self, type_id: str) -> bool:
        timestamp = now()
        with self.connect() as conn:
            result = conn.execute("UPDATE ontology_types SET deleted_at=?, updated_at=? WHERE type_id=? AND deleted_at IS NULL", (timestamp, timestamp, type_id))
            if result.rowcount:
                conn.execute("DELETE FROM ontology_card_assignments WHERE type_id=?", (type_id,))
                conn.execute("UPDATE ontology_type_relations SET deleted_at=?, updated_at=? WHERE deleted_at IS NULL AND (source_type_id=? OR target_type_id=?)", (timestamp, timestamp, type_id, type_id))
        return result.rowcount == 1

    def assign_cards_to_ontology_type(self, type_id: str, card_ids: list[str]) -> int:
        cleaned = sorted({str(card_id).strip() for card_id in card_ids if str(card_id).strip()})
        timestamp = now()
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM ontology_types WHERE type_id=? AND deleted_at IS NULL", (type_id,)).fetchone():
                raise ValueError("활성 온톨로지 타입을 찾을 수 없습니다.")
            before = conn.total_changes
            conn.executemany("INSERT OR IGNORE INTO ontology_card_assignments(type_id,card_id,assigned_at) VALUES (?,?,?)", [(type_id, card_id, timestamp) for card_id in cleaned])
            return conn.total_changes - before

    def unassign_card_from_ontology_type(self, card_id: str, type_id: str) -> bool:
        with self.connect() as conn:
            result = conn.execute("DELETE FROM ontology_card_assignments WHERE card_id=? AND type_id=?", (str(card_id).strip(), str(type_id).strip()))
        return result.rowcount == 1

    def set_card_ontology_types(self, card_id: str, type_ids: list[str]) -> None:
        card_id = str(card_id).strip()
        if not card_id:
            raise ValueError("지식카드 ID는 비어 있을 수 없습니다.")
        cleaned = sorted({str(type_id).strip() for type_id in type_ids if str(type_id).strip()})
        timestamp = now()
        with self.connect() as conn:
            if cleaned:
                marks = ",".join("?" for _ in cleaned)
                active = {str(row["type_id"]) for row in conn.execute(f"SELECT type_id FROM ontology_types WHERE deleted_at IS NULL AND type_id IN ({marks})", cleaned).fetchall()}
                if active != set(cleaned):
                    raise ValueError("활성 온톨로지 타입을 찾을 수 없습니다.")
            conn.execute("DELETE FROM ontology_card_assignments WHERE card_id=?", (card_id,))
            conn.executemany("INSERT INTO ontology_card_assignments(type_id,card_id,assigned_at) VALUES (?,?,?)", [(type_id, card_id, timestamp) for type_id in cleaned])

    def set_card_ontology_type(self, card_id: str, type_id: str | None) -> None:
        self.set_card_ontology_types(card_id, [type_id] if type_id else [])

    def replace_ontology_type_cards(self, type_id: str, card_ids: list[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM ontology_card_assignments WHERE type_id=?", (type_id,))
        self.assign_cards_to_ontology_type(type_id, card_ids)

    def ontology_card_ids(self, type_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT card_id FROM ontology_card_assignments WHERE type_id=? ORDER BY assigned_at, card_id", (type_id,)).fetchall()
        return [str(row["card_id"]) for row in rows]

    def ontology_types_for_card(self, card_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("""SELECT t.*, f.name AS facet_name
                FROM ontology_types t JOIN ontology_card_assignments a ON a.type_id=t.type_id
                LEFT JOIN ontology_facets f ON f.facet_id=t.facet_id AND f.deleted_at IS NULL
                WHERE a.card_id=? AND t.deleted_at IS NULL
                ORDER BY COALESCE(lower(f.name), 'zzzz'), lower(t.name)""", (card_id,)).fetchall()
        return [dict(row) for row in rows]

    def create_ontology_type_relation(
        self, source_type_id: str, target_type_id: str, relation_name: str, description: str = ""
    ) -> dict[str, Any]:
        if source_type_id == target_type_id:
            raise ValueError("타입 관계의 출발·도착 타입은 달라야 합니다.")
        relation_name = relation_name.strip()
        if not relation_name:
            raise ValueError("타입 관계 이름은 비어 있을 수 없습니다.")
        relation_id, timestamp = f"otr-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO ontology_type_relations VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                    (relation_id, source_type_id, target_type_id, relation_name, description.strip(), timestamp, timestamp),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("같은 타입 사이에 같은 이름의 활성 관계가 이미 있습니다.") from error
            row = conn.execute(
                "SELECT * FROM ontology_type_relations WHERE relation_id=?", (relation_id,)
            ).fetchone()
        return dict(row)

    def ontology_type_relations(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ontology_type_relations WHERE deleted_at IS NULL ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_ontology_type_relation(self, relation_id: str) -> bool:
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE ontology_type_relations SET deleted_at=?, updated_at=? WHERE relation_id=? AND deleted_at IS NULL",
                (now(), now(), relation_id),
            )
        return result.rowcount == 1

    def delete_knowledge_relation(self, relation_id: str, note: str = "") -> bool:
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE knowledge_relations SET deleted_at=?, deleted_note=? WHERE relation_id=? AND deleted_at IS NULL",
                (now(), note, relation_id),
            )
        return result.rowcount == 1

    def upsert_shelf_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Store a paper as a research asset, independently of knowledge approval."""
        title = str(paper.get("title", "")).strip()
        if not title:
            raise ValueError("논문 제목은 비어 있을 수 없습니다.")
        source_id = str(paper.get("source_id", "")).strip()
        timestamp = now()
        with self.connect() as conn:
            existing = None
            if source_id:
                existing = conn.execute("SELECT paper_id FROM paper_shelf WHERE source_id=?", (source_id,)).fetchone()
            paper_id = str(existing["paper_id"]) if existing else str(paper.get("paper_id") or f"paper-{uuid.uuid4().hex[:12]}")
            conn.execute(
                """INSERT INTO paper_shelf
                   (paper_id, title, authors_json, publication_year, source_url, source_id, pdf_path, labels_json, shelf_status, reading_status, asset_type, intake_source, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id) DO UPDATE SET
                     title=excluded.title, authors_json=excluded.authors_json, publication_year=excluded.publication_year,
                     source_url=excluded.source_url, source_id=excluded.source_id,
                     pdf_path=CASE WHEN excluded.pdf_path <> '' THEN excluded.pdf_path ELSE paper_shelf.pdf_path END,
                     labels_json=CASE WHEN excluded.labels_json <> '[]' THEN excluded.labels_json ELSE paper_shelf.labels_json END,
                     shelf_status=excluded.shelf_status, reading_status=excluded.reading_status,
                     asset_type=excluded.asset_type, intake_source=excluded.intake_source, updated_at=excluded.updated_at""",
                (paper_id, title, json.dumps(paper.get("authors", []), ensure_ascii=False), str(paper.get("publication_year", "")),
                 str(paper.get("source_url", "")), source_id, str(paper.get("pdf_path", "")),
                 json.dumps(_clean_paper_labels(paper.get("labels", [])), ensure_ascii=False),
                 str(paper.get("shelf_status", "reference")), str(paper.get("reading_status", "unread")),
                 str(paper.get("asset_type", "paper")), str(paper.get("intake_source", "manual")), timestamp, timestamp),
            )
            abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
            if abstract:
                conn.execute(
                    "INSERT INTO paper_abstracts (paper_id, abstract, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(paper_id) DO UPDATE SET abstract=excluded.abstract, updated_at=excluded.updated_at",
                    (paper_id, abstract, timestamp),
                )
            self._record_paper_event(conn, paper_id, "intake", {"source": paper.get("intake_source", "manual")})
        return self.shelf_paper(paper_id) or {}

    def shelf_papers(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT p.*, COALESCE(a.abstract, '') AS abstract FROM paper_shelf p "
                "LEFT JOIN paper_abstracts a ON a.paper_id=p.paper_id ORDER BY p.updated_at DESC"
            ).fetchall()
        return [self._paper_shelf_row(row) for row in rows]

    def shelf_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT p.*, COALESCE(a.abstract, '') AS abstract FROM paper_shelf p "
                "LEFT JOIN paper_abstracts a ON a.paper_id=p.paper_id WHERE p.paper_id=?", (paper_id,)
            ).fetchone()
        return self._paper_shelf_row(row) if row else None

    def _paper_shelf_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["authors"] = json.loads(item.pop("authors_json"))
        item["labels"] = json.loads(item.pop("labels_json"))
        return item

    def update_shelf_paper(
        self, paper_id: str, *, shelf_status: str, reading_status: str, labels: list[str] | None = None,
        title: str | None = None, authors: list[str] | None = None, publication_year: str | None = None,
    ) -> None:
        if shelf_status not in {"core", "reference", "held", "excluded"}:
            raise ValueError("지원하지 않는 서재 상태입니다.")
        if reading_status not in {"unread", "reading", "read"}:
            raise ValueError("지원하지 않는 읽기 상태입니다.")
        if title is not None and not title.strip():
            raise ValueError("논문 제목은 비어 있을 수 없습니다.")
        with self.connect() as conn:
            sets, values = ["shelf_status=?", "reading_status=?"], [shelf_status, reading_status]
            if title is not None:
                sets.append("title=?")
                values.append(title.strip())
            if labels is not None:
                sets.append("labels_json=?")
                values.append(json.dumps(_clean_paper_labels(labels), ensure_ascii=False))
            if authors is not None:
                sets.append("authors_json=?")
                values.append(json.dumps([author for author in authors if author], ensure_ascii=False))
            if publication_year is not None:
                sets.append("publication_year=?")
                values.append(publication_year if re.fullmatch(r"(?:19|20)\d{2}", publication_year) else "")
            values.extend([now(), paper_id])
            conn.execute(f"UPDATE paper_shelf SET {', '.join(sets)}, updated_at=? WHERE paper_id=?", values)
            self._record_paper_event(conn, paper_id, "state_updated", {"importance": shelf_status, "reading_status": reading_status})

    def update_shelf_pdf_path(self, paper_id: str, pdf_path: str) -> None:
        """Update only the machine-local original path for a shelf paper."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE paper_shelf SET pdf_path=?, updated_at=? WHERE paper_id=?",
                (str(pdf_path or ""), now(), paper_id),
            )
            self._record_paper_event(conn, paper_id, "local_pdf_updated", {"pdf_path": str(pdf_path or "")})

    def delete_shelf_paper(self, paper_id: str) -> dict[str, int]:
        """Remove one shelf asset and its paper-specific working records.

        Knowledge cards are deliberately retained: they can be supported by more
        than one paper and are independently approved research knowledge.
        """
        with self.connect() as conn:
            question_rows = conn.execute(
                "SELECT question_id FROM paper_reading_questions WHERE paper_id=?", (paper_id,)
            ).fetchall()
            question_ids = [str(row["question_id"]) for row in question_rows]
            deleted = {
                "reviews": 0,
                "questions": 0,
                "ontology_candidates": 0,
                "analysis": 0,
                "card_links": 0,
                "events": 0,
                "paper": 0,
            }
            if question_ids:
                placeholders = ", ".join("?" for _ in question_ids)
                deleted["reviews"] = conn.execute(
                    f"DELETE FROM paper_reading_reviews WHERE question_id IN ({placeholders})", question_ids
                ).rowcount
            deleted["ontology_candidates"] = conn.execute(
                "DELETE FROM paper_ontology_candidates WHERE paper_id=?", (paper_id,)
            ).rowcount
            deleted["questions"] = conn.execute(
                "DELETE FROM paper_reading_questions WHERE paper_id=?", (paper_id,)
            ).rowcount
            conn.execute("DELETE FROM paper_abstracts WHERE paper_id=?", (paper_id,))
            deleted["analysis"] = conn.execute(
                "DELETE FROM paper_analyses WHERE paper_id=?", (paper_id,)
            ).rowcount
            deleted["card_links"] = conn.execute(
                "DELETE FROM paper_card_links WHERE paper_id=?", (paper_id,)
            ).rowcount
            deleted["events"] = conn.execute(
                "DELETE FROM paper_asset_events WHERE paper_id=?", (paper_id,)
            ).rowcount
            deleted["paper"] = conn.execute(
                "DELETE FROM paper_shelf WHERE paper_id=?", (paper_id,)
            ).rowcount
        return deleted

    def paper_analysis(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM paper_analyses WHERE paper_id=?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def save_paper_analysis(self, paper_id: str, *, research_question: str = "", summary: str = "", reading_raw_output: str = "", researcher_note: str = "", generated: bool = False) -> None:
        timestamp = now()
        previous = self.paper_analysis(paper_id) or {}
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO paper_analyses (paper_id, research_question, summary, reading_raw_output, researcher_note, generated_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id) DO UPDATE SET research_question=excluded.research_question,
                     summary=excluded.summary, reading_raw_output=excluded.reading_raw_output, researcher_note=excluded.researcher_note,
                     generated_at=CASE WHEN excluded.generated_at <> '' THEN excluded.generated_at ELSE paper_analyses.generated_at END,
                     updated_at=excluded.updated_at""",
                (paper_id, research_question or previous.get("research_question", ""), summary or previous.get("summary", ""),
                 reading_raw_output or previous.get("reading_raw_output", ""), researcher_note, timestamp if generated else "", timestamp),
            )
            self._record_paper_event(conn, paper_id, "analysis_generated" if generated else "researcher_note_updated", {})

    def paper_card_ids(self, paper_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT card_id FROM paper_card_links WHERE paper_id=? ORDER BY linked_at DESC", (paper_id,)).fetchall()
        return [str(row["card_id"]) for row in rows]

    def set_paper_card_links(self, paper_id: str, card_ids: list[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_card_links WHERE paper_id=?", (paper_id,))
            conn.executemany("INSERT INTO paper_card_links VALUES (?, ?, ?)", [(paper_id, card_id, now()) for card_id in dict.fromkeys(card_ids)])
            self._record_paper_event(conn, paper_id, "approved_cards_linked", {"card_ids": list(dict.fromkeys(card_ids))})

    @staticmethod
    def _record_paper_event(conn: sqlite3.Connection, paper_id: str, event_type: str, detail: dict[str, Any]) -> None:
        conn.execute("INSERT INTO paper_asset_events VALUES (?, ?, ?, ?, ?)", (
            f"pae-{uuid.uuid4().hex[:12]}", paper_id, event_type, json.dumps(detail, ensure_ascii=False), now(),
        ))

    def paper_asset_events(self, paper_id: str, limit: int = 40) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_asset_events WHERE paper_id=? ORDER BY created_at DESC LIMIT ?", (paper_id, limit)).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail_json"])} for row in rows]

    def add_paper_reading_questions(self, paper_id: str, questions: list[dict[str, Any]]) -> list[str]:
        timestamp, question_ids = now(), []
        with self.connect() as conn:
            # Early installations used two additional required columns
            # (research_relevance and suggested_ontology). Newer databases no
            # longer need them, but a running ledger must remain writable.
            table_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(paper_reading_questions)").fetchall()
            }
            existing_questions = {
                " ".join(str(row["question"]).split())
                for row in conn.execute("SELECT question FROM paper_reading_questions WHERE paper_id=?", (paper_id,)).fetchall()
            }
            for question in questions:
                normalized_question = " ".join(str(question["question"]).split())
                if normalized_question in existing_questions:
                    continue
                question_id = f"prq-{uuid.uuid4().hex[:12]}"
                values: dict[str, Any] = {
                    "question_id": question_id,
                    "paper_id": paper_id,
                    "question": question["question"],
                    "tentative_answer": question["tentative_answer"],
                    "evidence_json": json.dumps(question.get("evidence", []), ensure_ascii=False),
                    "uncertainty": question.get("uncertainty", ""),
                    # Compatibility values for the prior 13-column schema.
                    # Empty means the flow does not infer ontology/relevance
                    # before the researcher reviews the reading question.
                    "research_relevance": str(question.get("research_relevance", "")),
                    "suggested_ontology": str(question.get("suggested_ontology", "")),
                    "suggested_labels": str(question.get("suggested_labels", "")),
                    "suggested_title": str(question.get("suggested_title", "")),
                    "suggested_concepts": str(question.get("suggested_concepts", "")),
                    "suggested_applies_to": str(question.get("suggested_applies_to", "")),
                    "suggested_conditions": str(question.get("suggested_conditions", "")),
                    "suggested_limits": str(question.get("suggested_limits", "")),
                    "suggested_context": str(question.get("suggested_context", "")),
                    "suggested_implication": str(question.get("suggested_implication", "")),
                    "suggested_source_excerpt": str(question.get("suggested_source_excerpt", "")),
                    "researcher_comment": "",
                    "status": "proposed",
                    "promotion_request_id": None,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                }
                columns = [name for name in values if name in table_columns]
                placeholders = ", ".join("?" for _ in columns)
                conn.execute(
                    f"INSERT INTO paper_reading_questions ({', '.join(columns)}) VALUES ({placeholders})",
                    [values[name] for name in columns],
                )
                question_ids.append(question_id)
                existing_questions.add(normalized_question)
            self._record_paper_event(conn, paper_id, "reading_questions_generated", {"count": len(question_ids)})
        return question_ids

    def paper_reading_questions(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_reading_questions WHERE paper_id=? ORDER BY created_at", (paper_id,)).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence_json"])} for row in rows]

    def update_paper_reading_question(self, question_id: str, *, researcher_comment: str, status: str, promotion_request_id: str | None = None) -> bool:
        if status not in {"proposed", "approved", "needs_revision", "deferred", "irrelevant", "promoted", "registered"}:
            raise ValueError("지원하지 않는 논문 읽기 질문 상태입니다.")
        with self.connect() as conn:
            result = conn.execute("UPDATE paper_reading_questions SET researcher_comment=?, status=?, promotion_request_id=COALESCE(?, promotion_request_id), updated_at=? WHERE question_id=?", (researcher_comment, status, promotion_request_id, now(), question_id))
            paper = conn.execute("SELECT paper_id FROM paper_reading_questions WHERE question_id=?", (question_id,)).fetchone()
            if paper:
                self._record_paper_event(conn, str(paper["paper_id"]), "reading_question_reviewed", {"question_id": question_id, "status": status})
        return result.rowcount == 1

    def add_paper_ontology_candidate(self, paper_id: str, question_id: str | None, candidate_text: str, evidence: list[str]) -> str:
        candidate_id, timestamp = f"poc-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            conn.execute("INSERT INTO paper_ontology_candidates VALUES (?, ?, ?, ?, ?, '', 'proposed', ?, ?)", (candidate_id, paper_id, question_id, candidate_text, json.dumps(evidence, ensure_ascii=False), timestamp, timestamp))
            self._record_paper_event(conn, paper_id, "ontology_candidate_added", {"candidate_id": candidate_id})
        return candidate_id

    def paper_ontology_candidates(self, paper_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_ontology_candidates WHERE paper_id=? ORDER BY created_at", (paper_id,)).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence_json"])} for row in rows]

    def all_paper_ontology_candidates(self) -> list[dict[str, Any]]:
        """Return the cross-paper work queue for researcher ontology curation."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT candidate.*, paper.title AS paper_title, paper.shelf_status, paper.reading_status
                   FROM paper_ontology_candidates AS candidate
                   JOIN paper_shelf AS paper ON paper.paper_id = candidate.paper_id
                   ORDER BY candidate.updated_at DESC"""
            ).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence_json"])} for row in rows]

    def update_paper_ontology_candidate(self, candidate_id: str, *, researcher_comment: str, status: str, candidate_text: str | None = None) -> bool:
        if status not in {"proposed", "pending_approval", "approved", "needs_revision", "rejected"}:
            raise ValueError("지원하지 않는 온톨로지 후보 상태입니다.")
        with self.connect() as conn:
            result = conn.execute(
                """UPDATE paper_ontology_candidates
                   SET researcher_comment=?, status=?, candidate_text=COALESCE(?, candidate_text), updated_at=?
                   WHERE candidate_id=?""",
                (researcher_comment, status, candidate_text.strip() if candidate_text else None, now(), candidate_id),
            )
            row = conn.execute("SELECT paper_id FROM paper_ontology_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if row:
                self._record_paper_event(conn, str(row["paper_id"]), "ontology_candidate_reviewed", {"candidate_id": candidate_id, "status": status})
        return result.rowcount == 1

    def upsert_paper_reading_reviews(self, reviews: list[dict[str, Any]]) -> int:
        """Keep the first-pass question and the second-pass evidence augmentation separately."""
        timestamp, saved = now(), 0
        with self.connect() as conn:
            for review in reviews:
                question_id = str(review.get("question_id", ""))
                if not question_id:
                    continue
                existing = conn.execute("SELECT review_id FROM paper_reading_reviews WHERE question_id=?", (question_id,)).fetchone()
                review_id = str(existing["review_id"]) if existing else f"prr-{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """INSERT INTO paper_reading_reviews
                       (review_id, question_id, refined_answer, additional_evidence_json, remaining_uncertainty, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(review_id) DO UPDATE SET refined_answer=excluded.refined_answer,
                         additional_evidence_json=excluded.additional_evidence_json,
                         remaining_uncertainty=excluded.remaining_uncertainty, updated_at=excluded.updated_at""",
                    (review_id, question_id, review["refined_answer"], json.dumps(review.get("additional_evidence", []), ensure_ascii=False),
                     review.get("remaining_uncertainty", ""), timestamp, timestamp),
                )
                row = conn.execute("SELECT paper_id FROM paper_reading_questions WHERE question_id=?", (question_id,)).fetchone()
                if row:
                    self._record_paper_event(conn, str(row["paper_id"]), "second_pass_reading_completed", {"question_id": question_id})
                saved += 1
        return saved

    def paper_reading_reviews(self, paper_id: str) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT review.* FROM paper_reading_reviews review
                   JOIN paper_reading_questions question ON question.question_id=review.question_id
                   WHERE question.paper_id=?""", (paper_id,),
            ).fetchall()
        return {str(row["question_id"]): {**dict(row), "additional_evidence": json.loads(row["additional_evidence_json"])} for row in rows}


def _is_english_search_term(value: str) -> bool:
    """Permit searchable ASCII terms; Korean UI/context never enter an arXiv query."""
    return bool(value.strip()) and not any("가" <= character <= "힣" for character in value) and all(
        character.isascii() and (character.isalnum() or character in " -_()/+.#") for character in value
    )


def _clean_paper_labels(labels: Any) -> list[str]:
    """Keep shelf labels compact, deterministic and appropriate for filtering."""
    values = labels.split(",") if isinstance(labels, str) else labels
    cleaned: list[str] = []
    for value in values if isinstance(values, list) else []:
        label = " ".join(str(value).strip().split())
        key = label.casefold()
        if 1 < len(label) <= 48 and key not in {item.casefold() for item in cleaned}:
            cleaned.append(label)
        if len(cleaned) == 5:
            break
    return cleaned
