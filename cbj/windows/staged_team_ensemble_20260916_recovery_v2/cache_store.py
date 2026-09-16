"""Immutable probability artifacts with crash-safe adoption and exact lineage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


HEX = set("0123456789abcdef")
COMMON_KEYS = {
    "input_sha256", "parser_sha256", "source_sha256", "declaration_sha256",
    "fit_ids_sha256", "fit_groups_sha256", "partition_sha256",
    "feature_order_sha256", "parameters_sha256",
}


class CacheIntegrityError(RuntimeError):
    pass


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def make_cache_key(bindings, *, supervised):
    required = COMMON_KEYS | ({"fit_y_sha256"} if supervised else set())
    missing = sorted(required - set(bindings))
    if missing:
        raise ValueError(f"cache bindings missing {','.join(missing)}")
    if not supervised and "fit_y_sha256" in bindings:
        raise ValueError("label-free cache must omit fit_y_sha256")
    if any(not _valid_hash(bindings[name]) for name in required):
        raise ValueError("cache binding is not a lowercase sha256")
    payload = {"schema_version": "IMMUTABLE_CACHE_KEY_V1", "supervised": bool(supervised), "bindings": bindings}
    return _sha_bytes(_canonical(payload))


class ImmutableProbabilityCache:
    def __init__(self, root):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.ledger = self.root / "ledger"
        self.indexes = self.root / "indexes"
        for path in (self.artifacts, self.ledger, self.indexes):
            path.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, key):
        return self.artifacts / key

    def ledger_path(self, key):
        return self.ledger / f"{key}.json"

    def _write_json_once(self, path, value):
        path = Path(path)
        data = _canonical(value)
        try:
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes() != data:
                raise CacheIntegrityError(f"immutable JSON collision: {path}")

    def write(self, key, probability, bindings, *, stop_before_ledger=False):
        if key != make_cache_key(bindings, supervised="fit_y_sha256" in bindings):
            raise CacheIntegrityError("cache key does not match bindings")
        probability = np.asarray(probability, dtype=np.float64)
        if probability.ndim != 2 or not len(probability) or probability.shape[1] < 2:
            raise ValueError("probability matrix shape is invalid")
        if not np.isfinite(probability).all() or (probability < 0).any() or np.any(probability.sum(axis=1) <= 0):
            raise ValueError("probability matrix values are invalid")
        probability = probability / probability.sum(axis=1, keepdims=True)
        final = self.artifact_path(key)
        if not final.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.artifacts))
            try:
                np.save(temporary / "probability.npy", probability, allow_pickle=False)
                receipt = {
                    "schema_version": "IMMUTABLE_PROBABILITY_ARTIFACT_V1",
                    "key": key,
                    "bindings": bindings,
                    "rows": int(probability.shape[0]),
                    "classes": int(probability.shape[1]),
                    "probability_sha256": _sha_file(temporary / "probability.npy"),
                    "test_read": False,
                }
                (temporary / "COMPLETE.json").write_bytes(_canonical(receipt))
                os.replace(temporary, final)
            finally:
                if temporary.exists():
                    for child in temporary.iterdir():
                        child.unlink()
                    temporary.rmdir()
        loaded, _ = self._verify(key, bindings)
        if stop_before_ledger:
            return loaded, "WRITTEN_UNLEDGERED"
        self._commit_ledger(key)
        return loaded, "WRITTEN"

    def _verify(self, key, bindings):
        root = self.artifact_path(key)
        receipt_path = root / "COMPLETE.json"
        probability_path = root / "probability.npy"
        if not receipt_path.is_file() or not probability_path.is_file():
            raise CacheIntegrityError("cache completion files are missing")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("key") != key or receipt.get("bindings") != bindings:
            raise CacheIntegrityError("cache receipt lineage mismatch")
        if receipt.get("probability_sha256") != _sha_file(probability_path):
            raise CacheIntegrityError("cache probability hash mismatch")
        probability = np.load(probability_path, allow_pickle=False)
        if probability.shape != (receipt.get("rows"), receipt.get("classes")):
            raise CacheIntegrityError("cache probability shape mismatch")
        return probability, receipt

    def _commit_ledger(self, key):
        root = self.artifact_path(key)
        self._write_json_once(self.ledger_path(key), {
            "schema_version": "IMMUTABLE_CACHE_LEDGER_V1",
            "key": key,
            "complete_sha256": _sha_file(root / "COMPLETE.json"),
        })

    def load_or_adopt(self, key, bindings):
        probability, _ = self._verify(key, bindings)
        existed = self.ledger_path(key).exists()
        self._commit_ledger(key)
        return probability, "REUSED" if existed else "ADOPTED"

    def index_path(self, lookup_key):
        return self.indexes / f"{lookup_key}.json"

    def bind_index(self, lookup_key, full_key):
        self._write_json_once(self.index_path(lookup_key), {
            "schema_version": "CACHE_LOOKUP_INDEX_V1", "lookup_key": lookup_key, "full_key": full_key,
        })

    def resolve_index(self, lookup_key):
        path = self.index_path(lookup_key)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("lookup_key") != lookup_key or not _valid_hash(value.get("full_key")):
            raise CacheIntegrityError("cache lookup index is malformed")
        return value["full_key"]

