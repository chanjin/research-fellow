"""Durable storage adapters for AJD workflow checkpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class JsonFileCheckpointStore:
    """Atomic JSON-file checkpoint store for local/single-process deployments."""

    root: Path

    def _path(self, checkpoint_id: str) -> Path:
        safe = "".join(ch for ch in checkpoint_id if ch.isalnum() or ch in {"-", "_"})
        if not safe:
            raise ValueError("checkpoint_id must contain at least one safe character")
        return self.root / f"{safe}.json"

    def save(self, checkpoint_id: str, payload: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(checkpoint_id)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def load(self, checkpoint_id: str) -> dict[str, Any]:
        path = self._path(checkpoint_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid checkpoint payload: {path}")
        return raw

    def delete(self, checkpoint_id: str) -> None:
        path = self._path(checkpoint_id)
        if path.exists():
            path.unlink()
