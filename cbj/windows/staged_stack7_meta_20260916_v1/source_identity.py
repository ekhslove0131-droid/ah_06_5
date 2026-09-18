"""Deterministic allowlist-based source identity for the standalone package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root):
    root = Path(root)
    allowlist_path = root / "SOURCE_ALLOWLIST.txt"
    try:
        names = [line.strip() for line in allowlist_path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        raise ValueError("source allowlist is missing") from exc
    if not names or names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("source allowlist must be non-empty, unique, and sorted")
    rows = []
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
            raise ValueError(f"source allowlist path is unsafe: {name}")
        path = root / relative
        if not path.is_file():
            raise ValueError(f"allowlisted source is missing: {name}")
        rows.append({"path": name, "sha256": file_sha256(path)})
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(payload).hexdigest(), rows

