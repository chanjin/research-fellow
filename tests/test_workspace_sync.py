from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_fellow.memory import KnowledgeMemory
from research_fellow.storage import Ledger
from research_fellow.workspace_sync import WorkspaceSync


class WorkspaceSyncTests(unittest.TestCase):
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
                    "INSERT INTO paper_shelf VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("p1", "Paper", "[]", "2026", "", "doi:x", "/local/paper.pdf", "[]", "reference", "read", "paper", "manual", "2026-01-01", "2026-01-01"),
                )
            WorkspaceSync(local_db, server_db).apply()
            server = Ledger(server_db)
            with server.connect() as conn:
                pdf_path = conn.execute("SELECT pdf_path FROM paper_shelf WHERE paper_id='p1'").fetchone()[0]
            self.assertEqual(pdf_path, "")


if __name__ == "__main__":
    unittest.main()
