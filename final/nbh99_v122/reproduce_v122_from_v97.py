"""Reproduce the final v122 submission from the validated v97 parent CSV."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE / "submission_v97_parent.csv"
OUTPUT = HERE / "submission_v122_reproduced.csv"
EXPECTED_SHA256 = "4733e9daa2f7ea5da108229009f3d654cd25a362936e8944924746cac9a3734c"


def main() -> None:
    with PARENT.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames != ["ID", "SUBCLASS"]:
        raise ValueError(f"unexpected columns: {fieldnames}")
    targets = [row for row in rows if row["ID"] == "TEST_1293"]
    if len(targets) != 1 or targets[0]["SUBCLASS"] != "GBMLGG":
        raise ValueError("validated v97 parent identity does not match")
    targets[0]["SUBCLASS"] = "LGG"
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        # The submitted pandas CSV used CRLF on Windows; preserve it so the
        # reproduced artifact is byte-identical, not merely row-identical.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"SHA256 mismatch: {digest}")
    print(f"created: {OUTPUT}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
