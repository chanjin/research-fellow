"""Content-addressed cache for non-authoritative page-level LLM drafts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CandidateDraftCache:
    """Cache drafts only; it is never semantic memory or an approval record."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, *, document_id: str, page_number: int, model: str, template_source: str) -> str | None:
        path = self._path(document_id, page_number, model, template_source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return str(payload.get("draft") or "").strip() or None
        except (OSError, ValueError):
            return None

    def put(self, *, document_id: str, page_number: int, model: str, template_source: str, draft: str) -> None:
        path = self._path(document_id, page_number, model, template_source)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"draft": draft}, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            return

    def _path(self, document_id: str, page_number: int, model: str, template_source: str) -> Path:
        fingerprint = hashlib.sha256(f"{model}\0{template_source}".encode("utf-8")).hexdigest()[:16]
        return self.root / document_id / f"p{page_number:03d}-{fingerprint}.json"
