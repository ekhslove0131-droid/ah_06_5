from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_identity(root):
    root = Path(root)
    allowlist = [line.strip() for line in (root / "SOURCE_ALLOWLIST.txt").read_text().splitlines() if line.strip()]
    rows = []
    for name in allowlist:
        path = root / name
        if not path.is_file(): raise ValueError(f"allowlisted source is missing: {name}")
        rows.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    raw = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest(), rows
