from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_fellow.memory import KnowledgeMemory
from research_fellow.storage import Ledger
from research_fellow.workspace_profiles import ResearchWorkspaceProfile
from research_fellow.workspace_sync import AllWorkspacesSync, LOCAL_ONLY_TABLES, SYNC_TABLES, WorkspaceSync


class WorkspaceSyncTests(unittest.TestCase):
    def test_every_durable_table_has_an_explicit_sync_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "policy.db"
            Ledger(database)
            sync = WorkspaceSync(database, Path(tmp) / "server.db")
            self.assertEqual(sync.unclassified_tables(database), set())
            self.assertTrue({"paper_projects", "paper_project_events"}.issubset(SYNC_TABLES))
            self.assertTrue({"schema_meta", "llm_calls"}.issubset(LOCAL_ONLY_TABLES))

    def test_paper_project_and_manuscript_history_round_trip_between_machines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            school_db, home_db, server_db = root / "school.db", root / "home.db", root / "server.db"
            school = Ledger(school_db)
            Ledger(home_db); Ledger(server_db)
            project = school.create_paper_project(
                title="Persistent job agent paper",
                research_question="How should persistent job agents be engineered?",
                origin_type="researcher_input",
                origin_ids=[],
            )
            school.add_paper_project_event(project["project_id"], "manuscript_version", {
                "manuscript": {"version": 1, "title": "Draft", "sections": []},
                "source": "external_draft",
                "diff": [],
            })

            WorkspaceSync(school_db, server_db).apply()
            WorkspaceSync(home_db, server_db).apply()

            downloaded = Ledger(home_db).paper_project(project["project_id"])
            self.assertIsNotNone(downloaded)
            self.assertEqual(downloaded["title"], "Persistent job agent paper")
            self.assertEqual(downloaded["events"][0]["event_type"], "manuscript_version")
            self.assertEqual(downloaded["events"][0]["payload"]["manuscript"]["version"], 1)

            school.update_paper_project_status(project["project_id"], "completed")
            WorkspaceSync(school_db, server_db).apply()
            WorkspaceSync(home_db, server_db).apply()
            completed = Ledger(home_db).paper_project(project["project_id"])
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["events"][-1]["event_type"], "project_status_changed")
            self.assertEqual(completed["events"][-1]["payload"]["status"], "completed")

    def test_add_download_and_delete_without_stale_absence_deleting_new_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            school_db = root / "school.db"
            home_db = root / "home.db"
            server_db = root / "server.db"
            Ledger(school_db); Ledger(home_db); Ledger(server_db)
            school = KnowledgeMemory(school_db)
            home = KnowledgeMemory(home_db)

            school.add({
                "card_id": "kc-school",
                "title": "School card",
                "source_kind": "external_paper",
                "claim": "A sufficiently long claim",
                "provenance": {"source_name": "paper", "source_url": "", "source_locator": ""},
            })
            WorkspaceSync(school_db, server_db).apply()

            # Home never saw the card before. Its absence must not delete the server row.
            home_sync = WorkspaceSync(home_db, server_db)
            preview = home_sync.preview()
            self.assertTrue(any(c.direction == "server→local" and c.action == "add" for c in preview.changes))
            home_sync.apply()
            self.assertEqual([c["card_id"] for c in home.all()], ["kc-school"])

            # A real local deletion after baseline is propagated.
            self.assertTrue(home.remove("kc-school", "not useful"))
            preview = home_sync.preview()
            self.assertTrue(any(c.direction == "local→server" for c in preview.changes))
            home_sync.apply()
            server = KnowledgeMemory(server_db)
            self.assertEqual(server.all(), [])

    def test_conflict_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_db, b_db, server_db = root / "a.db", root / "b.db", root / "server.db"
            Ledger(a_db); Ledger(b_db); Ledger(server_db)
            a = KnowledgeMemory(a_db)
            a.add({
                "card_id": "kc-1", "title": "Base", "source_kind": "external_paper", "claim": "Base claim text",
                "provenance": {"source_name": "paper", "source_url": "", "source_locator": ""},
            })
            WorkspaceSync(a_db, server_db).apply()
            WorkspaceSync(b_db, server_db).apply()

            # Update same row independently on both replicas.
            a.add({
                "card_id": "kc-1", "title": "A", "source_kind": "external_paper", "claim": "Changed at A machine",
                "provenance": {"source_name": "paper", "source_url": "", "source_locator": ""},
            })
            b = KnowledgeMemory(b_db)
            b.add({
                "card_id": "kc-1", "title": "B", "source_kind": "external_paper", "claim": "Changed at B machine",
                "provenance": {"source_name": "paper", "source_url": "", "source_locator": ""},
            })
            WorkspaceSync(a_db, server_db).apply()
            conflict_preview = WorkspaceSync(b_db, server_db).preview()
            self.assertTrue(any(c.action == "conflict" for c in conflict_preview.changes))


    def test_server_workspace_directory_initializes_from_local_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_db = root / "local.db"
            server_dir = root / "icloud" / "ResearchFellow"
            local = KnowledgeMemory(local_db)
            local.add({
                "card_id": "kc-init",
                "title": "Initial card",
                "source_kind": "external_paper",
                "claim": "Initial local knowledge becomes the server baseline",
                "provenance": {"source_name": "paper", "source_url": "", "source_locator": ""},
            })

            sync = WorkspaceSync(local_db, server_dir)
            self.assertFalse(sync.server_exists)
            self.assertEqual(sync.server_db, server_dir / "research_fellow.db")
            self.assertEqual(sync.initialization_counts().get("knowledge_cards"), 1)

            sync.initialize_server_from_local()
            self.assertTrue(sync.server_exists)
            self.assertEqual([c["card_id"] for c in KnowledgeMemory(sync.server_db).all()], ["kc-init"])
            self.assertEqual(sync.preview().changes, [])

    def test_pdf_path_is_local_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_db, server_db = root / "local.db", root / "server.db"
            local = Ledger(local_db)
            Ledger(server_db)
            with local.connect() as conn:
                conn.execute(
                    """INSERT INTO paper_shelf
                       (paper_id,title,authors_json,publication_year,source_url,source_id,pdf_path,labels_json,
                        shelf_status,reading_status,asset_type,intake_source,created_at,updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("p1", "Paper", "[]", "2026", "", "doi:x", "/local/paper.pdf", "[]", "reference", "read", "paper", "manual", "2026-01-01", "2026-01-01"),
                )
            WorkspaceSync(local_db, server_db).apply()
            server = Ledger(server_db)
            with server.connect() as conn:
                pdf_path = conn.execute("SELECT pdf_path FROM paper_shelf WHERE paper_id='p1'").fetchone()[0]
            self.assertEqual(pdf_path, "")

    def test_all_workspace_sync_uploads_and_restores_paper_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_data = root / "first-machine"
            second_data = root / "second-machine"
            server_root = root / "server"
            first_data.mkdir(); second_data.mkdir()
            profile = ResearchWorkspaceProfile(
                key="custom", label="Custom", short_label="Custom",
                db_filename="custom.db", cache_name="custom", purpose="test",
                topic_ko="test", topic_en="test", browser_title="Custom",
                expertise_instruction="test", server_db_filename="custom.db",
            )
            source_file = root / "downloaded-paper.pdf"
            source_file.write_bytes(b"%PDF-1.4\nworkspace asset\n")
            first = Ledger(first_data / "custom.db")
            with first.connect() as conn:
                conn.execute(
                    """INSERT INTO paper_shelf
                       (paper_id,title,authors_json,publication_year,source_url,source_id,pdf_path,labels_json,
                        shelf_status,reading_status,asset_type,intake_source,created_at,updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("paper-1", "Paper", "[]", "2026", "", "doi:test", str(source_file), "[]",
                     "reference", "read", "paper", "manual", "2026-01-01", "2026-01-01"),
                )

            first_sync = AllWorkspacesSync({"custom": profile}, first_data, server_root)
            result = first_sync.apply()[0]
            self.assertTrue(result.initialized)
            self.assertEqual(result.assets.uploaded, 1)
            server_assets = server_root / "workspaces" / "custom" / "assets" / "paper_shelf"
            self.assertEqual(len(list(server_assets.glob("paper-1--*.pdf"))), 1)

            second_sync = AllWorkspacesSync({"custom": profile}, second_data, server_root)
            restored = second_sync.apply()[0]
            self.assertFalse(restored.initialized)
            self.assertEqual(restored.assets.downloaded, 1)
            with Ledger(second_data / "custom.db").connect() as conn:
                local_path = Path(conn.execute("SELECT pdf_path FROM paper_shelf WHERE paper_id='paper-1'").fetchone()[0])
            self.assertTrue(local_path.is_file())
            self.assertEqual(local_path.read_bytes(), source_file.read_bytes())


if __name__ == "__main__":
    unittest.main()
