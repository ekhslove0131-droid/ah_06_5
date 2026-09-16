"""Finite 54-head, 541-row train-only Stack7 meta search."""

from __future__ import annotations

import hashlib
import json
import os
from numbers import Real
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

try:
    from .artifact_store import (
        canonical_sha256, file_sha256, load_inner_bank, load_or_fit_head,
        probability_sha256, sequence_sha256,
    )
    from .baseline import load_baseline_inner
    from .meta_core import blend, declarations, fit_meta, predict_meta
except ImportError:  # supported direct script import
    from artifact_store import canonical_sha256, file_sha256, load_inner_bank, load_or_fit_head, probability_sha256, sequence_sha256
    from baseline import load_baseline_inner
    from meta_core import blend, declarations, fit_meta, predict_meta


_ALPHAS = (0.10, 0.25, 0.50, 0.75, 1.00)
_POOLINGS = ("arithmetic", "geometric")
_EVIDENCE = "CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED"


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_once(path, payload):
    path = Path(path)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable search artifact changed: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.is_file():
        if temporary.read_bytes() != payload:
            raise ValueError(f"foreign partial search artifact retained: {temporary.name}")
    else:
        with temporary.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _scores(y, probability, classes):
    predicted = probability.argmax(1)
    per_class = f1_score(y, predicted, labels=np.arange(classes), average=None, zero_division=0)
    return float(np.mean(per_class)), [float(value) for value in per_class]


def _validate_bank_baseline(bank, baseline, guard, lineage):
    rows, classes = len(bank["inner_ids"]), len(bank["class_order"])
    if baseline.shape != (rows, classes) or guard.dtype != np.bool_ or guard.shape != (rows,):
        raise ValueError("baseline/bank dimensions are misaligned")
    expected = {
        "ids_sha256": sequence_sha256(bank["inner_ids"]),
        "y_sha256": sequence_sha256(bank["inner_y"]),
        "groups_sha256": sequence_sha256(bank["inner_groups"]),
        "outer_folds_sha256": sequence_sha256(bank["inner_outer_folds"]),
        "inner_folds_sha256": sequence_sha256(bank["inner_folds"]),
        "class_order_sha256": sequence_sha256(bank["class_order"]),
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise ValueError("baseline/bank row or class identity mismatch")


def _fit_identity(config, bank, declaration, fold):
    folds = bank["inner_folds"]
    fit = folds != fold
    valid = folds == fold
    return {
        "source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "bank_sha256": config["training_bank_sha256"],
        "recipe_sha256": config["v3_frozen_recipe_sha256"],
        "declaration_sha256": canonical_sha256(declaration),
        "fit_ids_sha256": sequence_sha256(bank["inner_ids"][fit]),
        "fit_y_sha256": sequence_sha256(bank["inner_y"][fit]),
        "fit_groups_sha256": sequence_sha256(bank["inner_groups"][fit]),
        "fit_folds_sha256": sequence_sha256(folds[fit]),
        "valid_ids_sha256": sequence_sha256(bank["inner_ids"][valid]),
        "valid_y_sha256": sequence_sha256(bank["inner_y"][valid]),
        "valid_groups_sha256": sequence_sha256(bank["inner_groups"][valid]),
        "valid_folds_sha256": sequence_sha256(folds[valid]),
        "model_order": bank["model_ids"].astype(str).tolist(),
        "class_order": bank["class_order"].astype(str).tolist(),
        "inner_fold": int(fold),
    }


def _score_path(scores_root, index):
    return scores_root / f"{index:04d}.json"


def _commit_path(commits_root, index):
    return commits_root / f"{index:04d}.json"


def _score_commit(index, ledger_identity, row_payload, row_sha256):
    return {
        "schema_version": "STACK7_META_SCORE_COMMIT_V1",
        "status": "BOUND_BEFORE_ROW_PUBLISH",
        "declared_index": index,
        "ledger_identity_sha256": canonical_sha256(ledger_identity),
        "row_bytes_sha256": hashlib.sha256(row_payload).hexdigest(),
        "row_sha256": row_sha256,
    }


def _record_score(scores_root, commits_root, index, row, ledger_identity, progress_callback):
    value = {**row, "declared_index": index, "ledger_identity_sha256": canonical_sha256(ledger_identity)}
    value["row_sha256"] = canonical_sha256(value)
    path = _score_path(scores_root, index)
    payload = _json_bytes(value)
    commit_path = _commit_path(commits_root, index)
    commit_payload = _json_bytes(_score_commit(index, ledger_identity, payload, value["row_sha256"]))
    if commit_path.is_file() and commit_path.read_bytes() != commit_payload:
        raise ValueError("score commit differs from recomputed missing row")
    _write_once(commit_path, commit_payload)
    existed = path.is_file()
    _write_once(path, payload)
    if not existed and progress_callback is not None:
        progress_callback({"kind": "score", "declared_index": index, "path": str(path)})
    return value


def _valid_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _read_score_commit(commits_root, index, ledger_identity):
    path = _commit_path(commits_root, index)
    if not path.is_file():
        return None
    try:
        commit = json.loads(path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"score commit changed: {path.name}") from exc
    required = {
        "schema_version", "status", "declared_index", "ledger_identity_sha256",
        "row_bytes_sha256", "row_sha256",
    }
    if (set(commit) != required
            or commit.get("schema_version") != "STACK7_META_SCORE_COMMIT_V1"
            or commit.get("status") != "BOUND_BEFORE_ROW_PUBLISH"
            or commit.get("declared_index") != index
            or commit.get("ledger_identity_sha256") != canonical_sha256(ledger_identity)
            or not _valid_sha256(commit.get("row_bytes_sha256"))
            or not _valid_sha256(commit.get("row_sha256"))):
        raise ValueError(f"score commit identity changed: {path.name}")
    return commit


def _load_existing_score(scores_root, commits_root, index, ledger_identity):
    path = _score_path(scores_root, index)
    if not path.is_file():
        _read_score_commit(commits_root, index, ledger_identity)
        return None
    commit = _read_score_commit(commits_root, index, ledger_identity)
    if commit is None:
        raise ValueError(f"score row commit missing: {path.name}")
    try:
        row_payload = path.read_bytes()
        if hashlib.sha256(row_payload).hexdigest() != commit["row_bytes_sha256"]:
            raise ValueError(f"score commit row bytes mismatch: {path.name}")
        row = json.loads(row_payload)
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and "score commit" in str(exc):
            raise
        raise ValueError(f"immutable search artifact changed: {path.name}") from exc
    checksum = row.pop("row_sha256", None)
    if (row.get("declared_index") != index
            or row.get("ledger_identity_sha256") != canonical_sha256(ledger_identity)
            or checksum != canonical_sha256(row)
            or checksum != commit["row_sha256"]):
        raise ValueError(f"immutable search artifact changed: {path.name}")
    row["row_sha256"] = checksum
    return row


def _validate_score_metrics(row, classes):
    macro = row.get("macro_f1")
    per_class = row.get("per_class_f1")
    if (isinstance(macro, bool) or not isinstance(macro, Real) or not np.isfinite(macro)
            or not 0.0 <= float(macro) <= 1.0):
        raise ValueError("score row macro F1 is invalid")
    if (not isinstance(per_class, list) or len(per_class) != classes
            or any(isinstance(value, bool) or not isinstance(value, Real)
                   or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
                   for value in per_class)):
        raise ValueError("score row per-class F1 is invalid")
    if not _valid_sha256(row.get("probability_sha256")):
        raise ValueError("score row probability hash is invalid")


def _validate_requested_score(row, expected, classes):
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"score row requested {key} binding changed")
    _validate_score_metrics(row, classes)
    if row["kind"] == "meta_blend":
        standalone = row.get("standalone_meta_macro_f1")
        standalone_per_class = row.get("standalone_meta_per_class_f1")
        if (isinstance(standalone, bool) or not isinstance(standalone, Real)
                or not np.isfinite(standalone) or not 0.0 <= float(standalone) <= 1.0
                or not isinstance(standalone_per_class, list)
                or len(standalone_per_class) != classes
                or any(isinstance(value, bool) or not isinstance(value, Real)
                       or not np.isfinite(value) or not 0.0 <= float(value) <= 1.0
                       for value in standalone_per_class)
                or not _valid_sha256(row.get("standalone_meta_probability_sha256"))):
            raise ValueError("score row standalone meta diagnostic is invalid")
    return row


def _validate_alias_rows(rows):
    by_index = {row["declared_index"]: row for row in rows}
    aliases = 0
    for row in rows:
        if row["kind"] != "meta_blend":
            continue
        alpha = row["alpha"]
        pooling = row["pooling"]
        if alpha != 1.0:
            if (row.get("alias_of") is not None
                    or row.get("endpoint_canonical_index") is not None
                    or row.get("endpoint_probability_sha256") is not None
                    or row.get("endpoint_kind") is not None):
                raise ValueError("non-endpoint score row has alias metadata")
            continue
        canonical_index = 1 + int(row["declaration_index"]) * 10 + 8
        if (row.get("endpoint_canonical_index") != canonical_index
                or row.get("endpoint_kind") != "guarded_meta_alpha_1"):
            raise ValueError("alpha=1 endpoint canonical metadata changed")
        if pooling == "arithmetic":
            if row.get("alias_of") is not None or row["declared_index"] != canonical_index:
                raise ValueError("alpha=1 canonical endpoint metadata changed")
            if row.get("endpoint_probability_sha256") != row.get("probability_sha256"):
                raise ValueError("alpha=1 canonical endpoint probability hash changed")
        elif pooling == "geometric":
            aliases += 1
            canonical = by_index.get(canonical_index)
            if row.get("alias_of") != canonical_index or canonical is None:
                raise ValueError("alpha=1 alias canonical linkage changed")
            for key in ("head_id", "macro_f1", "per_class_f1", "probability_sha256"):
                if row.get(key) != canonical.get(key):
                    raise ValueError(f"alpha=1 alias {key} changed")
            if row.get("endpoint_probability_sha256") != canonical.get("probability_sha256"):
                raise ValueError("alpha=1 alias endpoint probability hash changed")
        else:
            raise ValueError("alpha=1 pooling is invalid")
    if len(rows) == 541 and aliases != 54:
        raise ValueError("complete score ledger must contain exactly 54 alpha=1 aliases")


def _load_all_scores(scores_root, commits_root, count, ledger_identity):
    rows = []
    for index in range(count):
        row = _load_existing_score(scores_root, commits_root, index, ledger_identity)
        if row is None:
            raise ValueError("score ledger is incomplete")
        rows.append(row)
    extra = list(scores_root.glob("*.json"))
    if len(extra) != count:
        raise ValueError("score directory contains undeclared rows")
    _validate_alias_rows(rows)
    return rows


def _score_commit_set_sha256(commits_root, count, ledger_identity):
    commit_rows = []
    for index in range(count):
        commit = _read_score_commit(commits_root, index, ledger_identity)
        if commit is None:
            raise ValueError("score commit set is incomplete")
        commit_rows.append({
            "declared_index": index,
            "commit_sha256": file_sha256(_commit_path(commits_root, index)),
            "row_bytes_sha256": commit["row_bytes_sha256"],
        })
    extra = list(commits_root.glob("*.json"))
    if len(extra) != count:
        raise ValueError("score commit directory contains undeclared receipts")
    return canonical_sha256(commit_rows)


def run_search(config: dict, run_root) -> dict:
    """Run or exactly resume the fixed finite inner meta search."""
    required = {
        "meta_source_sha256", "_runtime_config_sha256", "training_bank_sha256",
        "v3_frozen_recipe_sha256",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"search config missing keys: {sorted(missing)}")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    scores_root = run_root / "scores"; scores_root.mkdir(exist_ok=True)
    commits_root = run_root / "score_commits"; commits_root.mkdir(exist_ok=True)
    fits_root = run_root / "fits"; fits_root.mkdir(exist_ok=True)
    declaration_rows = declarations()
    if len(declaration_rows) != 54:
        raise ValueError("meta declaration count changed")
    bank = load_inner_bank(config)
    baseline, guard, baseline_lineage = load_baseline_inner(config)
    _validate_bank_baseline(bank, baseline, guard, baseline_lineage)
    y = bank["inner_y"].astype(np.int64)
    folds = bank["inner_folds"].astype(np.int64)
    classes = len(bank["class_order"])
    baseline_score = float(baseline_lineage["baseline_inner_macro_f1"])
    bindings = {
        "source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "bank_sha256": config["training_bank_sha256"],
        "bank_receipt_sha256": config["training_bank_receipt_sha256"],
        "baseline_probability_sha256": baseline_lineage["baseline_probability_sha256"],
        "baseline_recipe_sha256": config["v3_frozen_recipe_sha256"],
        "ids_sha256": sequence_sha256(bank["inner_ids"]),
        "y_sha256": sequence_sha256(y),
        "groups_sha256": sequence_sha256(bank["inner_groups"]),
        "folds_sha256": sequence_sha256(folds),
        "class_order_sha256": sequence_sha256(bank["class_order"]),
        "model_order_sha256": sequence_sha256(bank["model_ids"]),
    }
    ledger_identity = {
        "schema_version": "STACK7_META_SCORE_IDENTITY_V3",
        "bindings": bindings,
        "declarations_sha256": canonical_sha256(declaration_rows),
        "alphas": list(_ALPHAS), "poolings": list(_POOLINGS),
        "validation_evidence": _EVIDENCE,
    }
    _write_once(run_root / "SCORE_IDENTITY.json", _json_bytes(ledger_identity))
    progress_callback = config.get("progress_callback")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    score_index = 0
    baseline_request = {
        "kind": "baseline_control", "head_id": None, "declaration": None,
        "alpha": 0.0, "pooling": None, "standalone_meta_macro_f1": None,
        "alias_of": None, "endpoint_canonical_index": None,
        "endpoint_probability_sha256": None, "endpoint_kind": None,
        "bindings": bindings, "validation_evidence": _EVIDENCE,
    }
    existing_baseline = _load_existing_score(scores_root, commits_root, score_index, ledger_identity)
    if existing_baseline is not None:
        _validate_requested_score(existing_baseline, baseline_request, classes)
    baseline_score, baseline_per_class = _scores(y, baseline, classes)
    if abs(baseline_score - baseline_lineage["baseline_inner_macro_f1"]) > 1e-15:
        raise ValueError("baseline score changed after bank alignment")
    baseline_probability_hash = probability_sha256(baseline)
    if existing_baseline is None:
        _record_score(scores_root, commits_root, score_index, {
            **baseline_request, "macro_f1": baseline_score, "per_class_f1": baseline_per_class,
            "probability_sha256": baseline_probability_hash,
        }, ledger_identity, progress_callback)
    elif (existing_baseline["macro_f1"] != baseline_score
            or existing_baseline["per_class_f1"] != baseline_per_class
            or existing_baseline["probability_sha256"] != baseline_probability_hash):
        raise ValueError("baseline control score row changed")
    score_index += 1
    best_inner = baseline_score

    fit_callback = config.get("fit_meta_callback", fit_meta)
    predict_callback = config.get("predict_meta_callback", predict_meta)
    if not callable(fit_callback) or not callable(predict_callback):
        raise ValueError("meta callbacks must be callable")
    head_outputs = {}
    head_receipts = {}
    completed_fit_identities = set()
    for declaration_index, declaration in enumerate(declaration_rows):
        head_id = f"META-{declaration_index:02d}-{canonical_sha256(declaration)[:16]}"
        meta_probability = np.zeros((len(y), classes), dtype=np.float64)
        coverage = np.zeros(len(y), dtype=np.uint8)
        receipts = []
        for fold in (0, 1, 2):
            fit_mask, valid_mask = folds != fold, folds == fold
            identity = _fit_identity(config, bank, declaration, fold)
            cache_complete = fits_root / canonical_sha256(identity) / "COMPLETE.json"
            was_cached = cache_complete.is_file()
            print(
                f"[stack7-meta] head={declaration_index + 1}/54 fold={fold + 1}/3 stage=start",
                flush=True,
            )
            model, valid_probability, receipt = load_or_fit_head(
                fits_root, identity,
                lambda d=declaration, m=fit_mask: fit_callback(d, bank["inner_probability"][:, m, :], y[m]),
                lambda artifact, m=valid_mask: predict_callback(artifact, bank["inner_probability"][:, m, :]),
            )
            del model
            if valid_probability.shape != (int(valid_mask.sum()), classes):
                raise ValueError("cached fold probability dimensions changed")
            meta_probability[valid_mask] = valid_probability
            coverage[valid_mask] += 1
            receipt_path = Path(receipt["model_path"]).parent / "COMPLETE.json"
            receipts.append({
                "fold": fold, "identity_sha256": receipt["identity_sha256"],
                "receipt_sha256": file_sha256(receipt_path),
                "valid_probability_sha256": receipt["valid_probability_sha256"],
                "receipt_path": str(receipt_path),
            })
            completed_fit_identities.add(receipt["identity_sha256"])
            print(
                f"[stack7-meta] head={declaration_index + 1}/54 fold={fold + 1}/3 "
                f"stage=complete cache={'cached' if was_cached else 'new'}",
                flush=True,
            )
        if not np.all(coverage == 1):
            raise ValueError("meta OOF rows were not filled exactly once")
        meta_hash = probability_sha256(meta_probability)
        head_outputs[head_id] = meta_probability
        head_receipts[head_id] = receipts
        requested_rows = []
        for alpha in _ALPHAS:
            for pooling in _POOLINGS:
                canonical_index = 1 + declaration_index * 10 + 8 if alpha == 1.0 else None
                request = {
                    "kind": "meta_blend", "head_id": head_id, "declaration": declaration,
                    "declaration_index": declaration_index, "alpha": alpha, "pooling": pooling,
                    "standalone_meta_probability_sha256": meta_hash,
                    "bindings": bindings, "fold_receipts": receipts,
                    "validation_evidence": _EVIDENCE,
                    "alias_of": canonical_index if alpha == 1.0 and pooling == "geometric" else None,
                    "endpoint_canonical_index": canonical_index,
                    "endpoint_kind": "guarded_meta_alpha_1" if alpha == 1.0 else None,
                }
                existing = _load_existing_score(
                    scores_root, commits_root, score_index, ledger_identity,
                )
                if existing is not None:
                    _validate_requested_score(existing, request, classes)
                requested_rows.append((score_index, request, existing))
                score_index += 1
        existing_rows = [row for _, _, row in requested_rows if row is not None]
        missing_rows = [item for item in requested_rows if item[2] is None]
        if missing_rows:
            standalone_score, standalone_per_class = _scores(y, meta_probability, classes)
            for row in existing_rows:
                if (row["standalone_meta_macro_f1"] != standalone_score
                        or row["standalone_meta_per_class_f1"] != standalone_per_class):
                    raise ValueError("existing standalone meta diagnostic changed")
        else:
            standalone_score = existing_rows[0]["standalone_meta_macro_f1"]
            standalone_per_class = existing_rows[0]["standalone_meta_per_class_f1"]
            if any(row["standalone_meta_macro_f1"] != standalone_score
                   or row["standalone_meta_per_class_f1"] != standalone_per_class
                   for row in existing_rows[1:]):
                raise ValueError("completed head standalone meta diagnostics disagree")
        completed_rows = {}
        for row_index, request, existing in requested_rows:
            if existing is not None:
                row = existing
            elif request["alias_of"] is not None:
                canonical = completed_rows.get(request["alias_of"])
                if canonical is None:
                    raise ValueError("alpha=1 alias is missing its canonical endpoint")
                row = _record_score(scores_root, commits_root, row_index, {
                    **request,
                    "standalone_meta_macro_f1": standalone_score,
                    "standalone_meta_per_class_f1": standalone_per_class,
                    "macro_f1": canonical["macro_f1"],
                    "per_class_f1": canonical["per_class_f1"],
                    "probability_sha256": canonical["probability_sha256"],
                    "endpoint_probability_sha256": canonical["probability_sha256"],
                }, ledger_identity, progress_callback)
            else:
                probability = blend(
                    baseline, meta_probability, request["alpha"], request["pooling"], guard,
                )
                macro, per_class = _scores(y, probability, classes)
                probability_hash = probability_sha256(probability)
                row = _record_score(scores_root, commits_root, row_index, {
                    **request,
                    "standalone_meta_macro_f1": standalone_score,
                    "standalone_meta_per_class_f1": standalone_per_class,
                    "macro_f1": macro, "per_class_f1": per_class,
                    "probability_sha256": probability_hash,
                    "endpoint_probability_sha256": (
                        probability_hash if request["endpoint_canonical_index"] is not None else None
                    ),
                }, ledger_identity, progress_callback)
            completed_rows[row_index] = row
            best_inner = max(best_inner, row["macro_f1"])
        _validate_alias_rows(list(completed_rows.values()))
        print(
            f"[stack7-meta] head={declaration_index + 1}/54 stage=complete "
            f"standalone={standalone_score:.15f} bestinner={best_inner:.15f}",
            flush=True,
        )
    if score_index != 541 or len(completed_fit_identities) != 162:
        raise ValueError("fixed search did not complete all declarations")
    rows = _load_all_scores(scores_root, commits_root, 541, ledger_identity)
    commit_set_sha = _score_commit_set_sha256(commits_root, 541, ledger_identity)
    ledger_payload = b"".join(_json_bytes(row) for row in rows)
    ledger_path = run_root / "SCORE_LEDGER.jsonl"
    _write_once(ledger_path, ledger_payload)
    ledger_sha = file_sha256(ledger_path)
    _write_once(run_root / "SCORE_COMPLETE.json", _json_bytes({
        "schema_version": "STACK7_META_SCORE_COMPLETE_V1", "status": "COMPLETE",
        "declared_rows": 541, "ledger_sha256": ledger_sha,
        "identity_sha256": canonical_sha256(ledger_identity),
        "score_commit_set_sha256": commit_set_sha,
    }))
    winner = rows[0]
    for row in rows[1:]:
        if row["macro_f1"] > winner["macro_f1"]:
            winner = row
    selection_path = run_root / "FROZEN_SELECTION.json"
    if winner["kind"] == "baseline_control":
        reproduced = baseline
        selected_receipts = []
        if probability_sha256(reproduced) != winner["probability_sha256"]:
            raise ValueError("selected inner OOF does not reproduce from saved baseline")
    else:
        selected_receipts = head_receipts[winner["head_id"]]
        if not selection_path.is_file():
            reproduced = blend(
                baseline, head_outputs[winner["head_id"]], winner["alpha"], winner["pooling"], guard,
            )
            if probability_sha256(reproduced) != winner["probability_sha256"]:
                raise ValueError("selected inner OOF does not reproduce from saved fold artifacts")
    selection = {
        "schema_version": "STACK7_META_FROZEN_SELECTION_V1", "status": "FROZEN",
        "selected_kind": winner["kind"], "selected_head_id": winner.get("head_id"),
        "selected_declaration": winner.get("declaration"), "selected_alpha": winner.get("alpha"),
        "selected_pooling": winner.get("pooling"), "selected_inner_macro_f1": winner["macro_f1"],
        "selected_probability_sha256": winner["probability_sha256"],
        "selected_fold_receipts": selected_receipts, "bindings": bindings,
        "baseline_lineage": baseline_lineage, "ledger_sha256": ledger_sha,
        "score_commit_set_sha256": commit_set_sha,
        "ledger_identity_sha256": canonical_sha256(ledger_identity),
        "validation_evidence": _EVIDENCE,
        "deployment_required": winner["kind"] != "baseline_control",
    }
    _write_once(selection_path, _json_bytes(selection))
    result = {
        "schema_version": "STACK7_META_SEARCH_COMPLETE_V1", "status": "COMPLETE",
        "declared_meta_heads": 54, "completed_fit_identities": 162,
        "declared_score_rows": 541, "winner_is_incumbent": winner["kind"] == "baseline_control",
        "deployment_required": selection["deployment_required"],
        "selected_inner_macro_f1": winner["macro_f1"],
        "selection_path": str(selection_path), "selection_sha256": file_sha256(selection_path),
        "ledger_path": str(ledger_path), "ledger_sha256": ledger_sha,
        "score_commit_set_sha256": commit_set_sha,
        "validation_evidence": _EVIDENCE, "test_read": False, "full_oof_used": False,
        "new_bias_fitted": False,
    }
    complete_path = run_root / "SEARCH_COMPLETE.json"
    _write_once(complete_path, _json_bytes(result))
    print(
        f"[stack7-meta] stage=complete rows=541 fits=162 bestinner={winner['macro_f1']:.15f}",
        flush=True,
    )
    return json.loads(complete_path.read_text())
