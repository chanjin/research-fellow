"""Topic-directory storage compatibility layer.

Phase 1 keeps SQLite as the working durable store while making primary research
assets portable by topic.  New PDFs and extracted source text are mirrored under
``data/topics/<topic-key>/sources`` so the directory can be synced independently
of a machine-local workspace DB.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import os
import re
from typing import Any, Mapping

import yaml

TOPIC_FORMAT = "research-fellow-topic-v1"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _safe_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").strip().lower()).strip("-._")
    return key or "topic"


def topic_dir(data_dir: str | Path, topic_key: str) -> Path:
    return Path(data_dir).expanduser() / "topics" / _safe_key(topic_key)


def ensure_topic_directory(
    data_dir: str | Path,
    *,
    topic_key: str,
    label: str = "",
    purpose: str = "",
) -> Path:
    """Create the self-contained topic skeleton without changing DB semantics."""
    root = topic_dir(data_dir, topic_key)
    for relative in (
        "sources/papers",
        "sources/html",
        "sources/text",
        "literature",
        "reviews",
        "knowledge",
        "research/answers",
        "index",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    manifest_path = root / "topic.yaml"
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                previous = dict(loaded)
        except Exception:
            previous = {}
    created_at = str(previous.get("created_at") or _utcnow())
    payload = {
        **previous,
        "format": TOPIC_FORMAT,
        "key": _safe_key(topic_key),
        "label": str(label or previous.get("label") or topic_key),
        "purpose": str(purpose or previous.get("purpose") or ""),
        "created_at": created_at,
        "updated_at": _utcnow(),
        "storage": {
            "papers": "sources/papers",
            "extracted_text": "sources/text",
            "source_manifest": "sources/sources.json",
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    sources_manifest = root / "sources" / "sources.json"
    if not sources_manifest.exists():
        sources_manifest.write_text(
            json.dumps({"format": TOPIC_FORMAT, "sources": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return root


def active_topic_dir() -> Path | None:
    value = os.environ.get("RESEARCH_FELLOW_TOPIC_DIR", "").strip()
    return Path(value).expanduser() if value else None


def active_topic_paper_dir() -> Path | None:
    root = active_topic_dir()
    return root / "sources" / "papers" if root else None


def _safe_source_filename(paper_id: str) -> str:
    raw = str(paper_id or "paper").strip()
    digest = __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:64] or "paper"
    return f"{stem}-{digest}.txt"


def _load_sources(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"format": TOPIC_FORMAT, "sources": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        sources = {}
    return {**payload, "format": TOPIC_FORMAT, "sources": sources}


def mirror_paper_source(
    *,
    paper_id: str,
    content: str = "",
    paper: Mapping[str, Any] | None = None,
    source_type: str = "",
    source_url: str = "",
    source_name: str = "",
    extraction_note: str = "",
    pdf_path: str = "",
) -> dict[str, str]:
    """Mirror reusable paper text/metadata into the active topic directory.

    This is intentionally best-effort and additive. SQLite remains the working
    store in Phase 1; a topic directory sync therefore cannot break live review.
    """
    root = active_topic_dir()
    if root is None:
        return {}
    (root / "sources" / "text").mkdir(parents=True, exist_ok=True)
    manifest_path = root / "sources" / "sources.json"
    payload = _load_sources(manifest_path)
    sources = payload["sources"]
    item = dict(sources.get(str(paper_id)) or {})
    paper_data = dict(paper or {})

    text_ref = str(item.get("full_text") or "")
    if str(content or "").strip():
        text_name = _safe_source_filename(str(paper_id))
        text_path = root / "sources" / "text" / text_name
        text_path.write_text(str(content), encoding="utf-8")
        text_ref = str(text_path.relative_to(root))

    resolved_pdf = str(pdf_path or paper_data.get("pdf_path") or item.get("pdf") or "").strip()
    pdf_ref = resolved_pdf
    if resolved_pdf:
        try:
            candidate = Path(resolved_pdf).expanduser().resolve()
            if candidate.is_relative_to(root.resolve()):
                pdf_ref = str(candidate.relative_to(root.resolve()))
        except Exception:
            pass

    sources[str(paper_id)] = {
        **item,
        "paper_id": str(paper_id),
        "title": str(paper_data.get("title") or item.get("title") or ""),
        "authors": paper_data.get("authors") or item.get("authors") or [],
        "publication_year": str(paper_data.get("publication_year") or item.get("publication_year") or ""),
        "doi_or_source_id": str(paper_data.get("source_id") or item.get("doi_or_source_id") or ""),
        "source_url": str(source_url or paper_data.get("full_text_url") or paper_data.get("source_url") or item.get("source_url") or ""),
        "source_type": str(source_type or item.get("source_type") or ""),
        "source_name": str(source_name or item.get("source_name") or ""),
        "pdf": pdf_ref,
        "full_text": text_ref,
        "extraction_note": str(extraction_note or item.get("extraction_note") or ""),
        "updated_at": _utcnow(),
    }
    payload["updated_at"] = _utcnow()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "full_text": text_ref, "pdf": pdf_ref}
