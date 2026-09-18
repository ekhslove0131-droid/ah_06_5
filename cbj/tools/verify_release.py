#!/usr/bin/env python3
"""Verify the public CBJ release using only the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


MANIFEST_NAME = "RELEASE_MANIFEST.json"
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".cbm",
    ".csv",
    ".joblib",
    ".model",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
FORBIDDEN_DIRS = {"__pycache__", "cache", "caches", "models", "runs"}
FORBIDDEN_NAMES = {".env", ".env.local", ".env.production", ".DS_Store"}


class VerificationError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}: {exc}") from exc


def release_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST_NAME
    )


def validate_forbidden_artifacts(root: Path, files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES:
            raise VerificationError(f"forbidden artifact: {relative.as_posix()}")
        if any(part.lower() in FORBIDDEN_DIRS for part in relative.parts[:-1]):
            raise VerificationError(f"forbidden artifact directory: {relative.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise VerificationError(f"forbidden artifact: {relative.as_posix()}")


def validate_python(root: Path, files: list[Path]) -> int:
    count = 0
    for path in files:
        if path.suffix.lower() != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, path.relative_to(root).as_posix(), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise VerificationError(
                f"Python syntax error: {path.relative_to(root).as_posix()}: {exc}"
            ) from exc
        count += 1
    return count


def validate_notebooks(root: Path, files: list[Path]) -> int:
    count = 0
    for path in files:
        if path.suffix.lower() != ".ipynb":
            continue
        notebook = read_json(path)
        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            raise VerificationError(f"invalid notebook: {path.relative_to(root).as_posix()}")
        for index, cell in enumerate(notebook["cells"]):
            if not isinstance(cell, dict):
                raise VerificationError(
                    f"invalid notebook cell: {path.relative_to(root).as_posix()}#{index}"
                )
            if cell.get("outputs"):
                raise VerificationError(
                    f"notebook contains outputs: {path.relative_to(root).as_posix()}#{index}"
                )
            if cell.get("cell_type") == "code" and cell.get("execution_count") is not None:
                raise VerificationError(
                    f"notebook has execution count: {path.relative_to(root).as_posix()}#{index}"
                )
        count += 1
    return count


def _validate_hash_rows(package: Path, rows: object, label: str) -> list[dict[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise VerificationError(f"{label} has no source files: {package}")
    checked: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise VerificationError(f"invalid {label} row: {package}")
        relative = row["path"]
        expected = row["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise VerificationError(f"invalid {label} row types: {package}")
        candidate = package / relative
        if not candidate.is_file():
            raise VerificationError(f"source closure missing file: {candidate}")
        actual = sha256(candidate)
        if actual != expected:
            raise VerificationError(f"source closure hash mismatch: {candidate}")
        checked.append({"path": relative, "sha256": actual})
    return checked


def validate_source_closures(root: Path) -> int:
    count = 0
    for manifest_path in sorted(root.rglob("SOURCE_MANIFEST.json")):
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise VerificationError(f"invalid source manifest: {manifest_path}")
        rows = _validate_hash_rows(manifest_path.parent, manifest.get("files"), "source manifest")
        payload = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
        actual_identity = hashlib.sha256(payload).hexdigest()
        if manifest.get("source_sha256") != actual_identity:
            raise VerificationError(f"source identity mismatch: {manifest_path}")
        count += 1

    for closure_path in sorted(root.rglob("SOURCE_CLOSURE.json")):
        closure = read_json(closure_path)
        files = closure.get("files") if isinstance(closure, dict) else None
        if not isinstance(files, dict) or not files:
            raise VerificationError(f"invalid source closure: {closure_path}")
        package = closure_path.parent.parent
        recovery_acceptance = package / "artifacts" / "RECOVERY_ACCEPTANCE_20260916.json"
        if recovery_acceptance.is_file():
            acceptance = read_json(recovery_acceptance)
            expected_identity = acceptance.get("source_sha256") if isinstance(acceptance, dict) else None
            names = sorted(
                path.relative_to(package).as_posix()
                for path in package.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.startswith("._")
                and path.suffix in {".py", ".json"}
                and not path.relative_to(package).as_posix().startswith(("runs/", "artifacts/"))
                and path.name not in {"runtime_config.windows.json", "LOCAL_ACCEPTANCE.json"}
            )
            rows = [{"path": name, "sha256": sha256(package / name)} for name in names]
            payload = (json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n").encode()
            actual_identity = hashlib.sha256(payload).hexdigest()
            if expected_identity != actual_identity:
                raise VerificationError(f"recovery source identity mismatch: {package}")
            count += 1
            continue
        for relative, expected in files.items():
            candidate = package / relative
            if not candidate.is_file():
                raise VerificationError(f"source closure missing file: {candidate}")
            if sha256(candidate) != expected:
                raise VerificationError(f"source closure hash mismatch: {candidate}")
        count += 1
    return count


def private_input_lines(root: Path) -> list[str]:
    path = root / "PRIVATE_INPUTS.json"
    if not path.is_file():
        raise VerificationError("PRIVATE_INPUTS.json is missing")
    document = read_json(path)
    inputs = document.get("inputs") if isinstance(document, dict) else None
    if not isinstance(inputs, list) or not inputs:
        raise VerificationError("PRIVATE_INPUTS.json has no inputs")
    lines: list[str] = []
    for item in inputs:
        if not isinstance(item, dict):
            raise VerificationError("PRIVATE_INPUTS.json contains an invalid item")
        missing = {"name", "role", "sha256", "shape", "included"} - set(item)
        if missing:
            raise VerificationError(f"private input fields missing: {sorted(missing)}")
        if item["included"] is not False:
            raise VerificationError(f"private input must be NOT_INCLUDED: {item['name']}")
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise VerificationError(f"private input SHA-256 invalid: {item['name']}")
        lines.append(f"{item['name']}: NOT_INCLUDED")
    return lines


def manifest_rows(root: Path, files: list[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]


def write_manifest(root: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "CBJ_PUBLIC_RELEASE_MANIFEST_V1",
        "scope": "cbj/ excluding RELEASE_MANIFEST.json",
        "files": rows,
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_manifest(root: Path, actual_rows: list[dict[str, object]]) -> None:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise VerificationError(f"{MANIFEST_NAME} is missing; run --write-manifest during trusted packaging")
    manifest = read_json(path)
    expected_rows = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(expected_rows, list):
        raise VerificationError(f"{MANIFEST_NAME} has invalid files")
    expected_by_path = {row.get("path"): row for row in expected_rows if isinstance(row, dict)}
    actual_by_path = {row["path"]: row for row in actual_rows}
    if set(expected_by_path) != set(actual_by_path):
        missing = sorted(set(expected_by_path) - set(actual_by_path))
        unexpected = sorted(set(actual_by_path) - set(expected_by_path))
        raise VerificationError(f"release inventory mismatch: missing={missing}, unexpected={unexpected}")
    for relative, actual in actual_by_path.items():
        expected = expected_by_path[relative]
        if expected.get("sha256") != actual["sha256"]:
            raise VerificationError(f"release hash mismatch: {relative}")
        if expected.get("bytes") != actual["bytes"]:
            raise VerificationError(f"release size mismatch: {relative}")


def verify(root: Path, write: bool) -> None:
    if not root.is_dir():
        raise VerificationError(f"release root does not exist: {root}")
    files = release_files(root)
    validate_forbidden_artifacts(root, files)
    python_count = validate_python(root, files)
    notebook_count = validate_notebooks(root, files)
    closure_count = validate_source_closures(root)
    private_lines = private_input_lines(root)
    rows = manifest_rows(root, files)
    if write:
        write_manifest(root, rows)
        print(f"WROTE {root / MANIFEST_NAME}")
    else:
        validate_manifest(root, rows)
        print(
            f"PASS files={len(files)} source_closures={closure_count} "
            f"python={python_count} notebooks={notebook_count}"
        )
    for line in private_lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="replace RELEASE_MANIFEST.json after trusted release assembly",
    )
    args = parser.parse_args(argv)
    try:
        verify(args.root.resolve(), args.write_manifest)
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
