"""Frozen-only Stack7 meta evaluation and immutable CSV publication."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

try:
    from .artifact_store import canonical_sha256, file_sha256, load_full_bank, load_inner_bank, load_or_fit_head, probability_sha256, sequence_sha256
    from .baseline import load_baseline_inner, replay_v3_fold
    from .meta_core import blend, fit_meta, predict_meta
    from .meta_search import _fit_identity
except ImportError:  # supported direct script import
    from artifact_store import canonical_sha256, file_sha256, load_full_bank, load_inner_bank, load_or_fit_head, probability_sha256, sequence_sha256
    from baseline import load_baseline_inner, replay_v3_fold
    from meta_core import blend, fit_meta, predict_meta
    from meta_search import _fit_identity


MODEL_ORDER = (
    "ANCHOR-H1", "ANCHOR-C10", "OLD-G3-0014", "OLD-G3-0018",
    "TEAM-T2-0005", "TEAM-T2-0007", "TEAM-T4-0007",
)
BASE_SOURCE_SHA256 = "b56d607e5da1edf3f0e4c97e426182896f094ba35faf038735ceabe03b343b52"
STAGE_DEPLOYMENT_SHA256 = "398f67e2ac523eab4e485d1affa2e77e869c7dd2bfa807ddfd71a2670792b112"
V3_PREDICTIONS_SHA256 = "33c0e538a379dd3299117c7cf93546bf32a306f00ab95cc18daf81c5ba916796"
V3_SUBMISSION_SHA256 = "a80e0c7fe0b630f9f0111bfca7581ede30261f9c0ee3c101c040c1f1f4ac7061"
SAMPLE_SUBMISSION_SHA256 = "1d0e9fe0b5ab5c763eab8c97130a06712e2ac2b428299481109b447d8f2b4d84"
FULL_EVIDENCE = "CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED"
CONFIRMATION_EVIDENCE = "HELD_OUT_FROM_FITTING_PREVIOUSLY_OBSERVED"


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read_json(path, label="JSON artifact"):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _write_once(path, payload):
    path = Path(path)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable deployment artifact changed: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.is_file() and temporary.read_bytes() != payload:
        raise ValueError(f"foreign deployment partial retained: {temporary.name}")
    if not temporary.is_file():
        with temporary.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _tensor_sha(value):
    return probability_sha256(np.asarray(value, dtype=np.float64))


def deployment_fit_identity(common, feature_role, fit_probability, valid_probability):
    """Bind deployment cache identity to role and exact feature tensors."""
    if feature_role not in {"full_meta_outer_cv", "outer0_confirmation_inner_oof"}:
        raise ValueError("unknown deployment feature role")
    identity = dict(common)
    identity.update({
        "schema_version": "STACK7_META_DEPLOYMENT_FIT_IDENTITY_V1",
        "feature_role": feature_role,
        "fit_probability_tensor_sha256": _tensor_sha(fit_probability),
        "valid_probability_tensor_sha256": _tensor_sha(valid_probability),
    })
    canonical_sha256(identity)
    return identity


def load_or_fit_deployment_head(root, identity, fit_callback, valid_probability_callback):
    """Thin Task4 wrapper over the accepted immutable artifact store."""
    return load_or_fit_head(root, identity, fit_callback, valid_probability_callback)


def _valid_sha(value):
    try:
        return isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        return False


def _verified_inner_context(config):
    bank = load_inner_bank(config)
    baseline, guard, lineage = load_baseline_inner(config)
    expected_lineage = {
        "ids_sha256": sequence_sha256(bank["inner_ids"]),
        "y_sha256": sequence_sha256(bank["inner_y"]),
        "groups_sha256": sequence_sha256(bank["inner_groups"]),
        "outer_folds_sha256": sequence_sha256(bank["inner_outer_folds"]),
        "inner_folds_sha256": sequence_sha256(bank["inner_folds"]),
        "class_order_sha256": sequence_sha256(bank["class_order"]),
    }
    if any(lineage.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError("frozen baseline/inner bank identity mismatch")
    expected_bindings = {
        "source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "bank_sha256": config["training_bank_sha256"],
        "bank_receipt_sha256": config["training_bank_receipt_sha256"],
        "baseline_probability_sha256": lineage["baseline_probability_sha256"],
        "baseline_recipe_sha256": config["v3_frozen_recipe_sha256"],
        "ids_sha256": expected_lineage["ids_sha256"],
        "y_sha256": expected_lineage["y_sha256"],
        "groups_sha256": expected_lineage["groups_sha256"],
        "folds_sha256": expected_lineage["inner_folds_sha256"],
        "class_order_sha256": expected_lineage["class_order_sha256"],
        "model_order_sha256": sequence_sha256(bank["model_ids"]),
    }
    return bank, baseline, guard, lineage, expected_bindings


def _verify_selected_inner_caches(config, root, selection, winner, bank, baseline, guard):
    y = bank["inner_y"].astype(np.int64)
    classes = len(bank["class_order"])
    if winner["kind"] == "baseline_control":
        if selection.get("selected_fold_receipts") != []:
            raise ValueError("incumbent selection unexpectedly binds meta fits")
        reproduced = baseline
    else:
        selected_receipts = selection.get("selected_fold_receipts")
        if selected_receipts != winner.get("fold_receipts") or not isinstance(selected_receipts, list) or len(selected_receipts) != 3:
            raise ValueError("selected fold receipt set mismatch")
        folds = bank["inner_folds"].astype(np.int64)
        meta_probability = np.zeros((len(y), classes), dtype=np.float64)
        coverage = np.zeros(len(y), dtype=np.uint8)
        for fold in (0, 1, 2):
            valid = folds == fold
            identity = _fit_identity(config, bank, selection["selected_declaration"], fold)
            identity_sha = canonical_sha256(identity)
            target = root / "fits" / identity_sha
            complete_path = target / "COMPLETE.json"
            if not complete_path.is_file():
                raise ValueError(f"selected fit cache missing: fold {fold}")
            selected = next((row for row in selected_receipts if row.get("fold") == fold), None)
            if selected is None:
                raise ValueError(f"selected fold receipt missing: fold {fold}")
            if (selected.get("identity_sha256") != identity_sha
                    or selected.get("receipt_path") != str(complete_path)
                    or selected.get("receipt_sha256") != file_sha256(complete_path)):
                raise ValueError(f"selected fold receipt binding mismatch: fold {fold}")
            def forbidden_fit():
                raise ValueError("selected inner cache verification attempted a fit")
            def forbidden_predict(_model):
                raise ValueError("selected inner cache verification attempted prediction")
            _model, valid_probability, receipt = load_or_fit_head(
                root / "fits", identity, forbidden_fit, forbidden_predict,
            )
            if (receipt.get("valid_probability_sha256") != selected.get("valid_probability_sha256")
                    or valid_probability.shape != (int(valid.sum()), classes)):
                raise ValueError(f"selected fold probability binding mismatch: fold {fold}")
            meta_probability[valid] = valid_probability
            coverage[valid] += 1
        if not np.all(coverage == 1):
            raise ValueError("selected inner OOF coverage mismatch")
        reproduced = blend(
            baseline, meta_probability, selection["selected_alpha"],
            selection["selected_pooling"], guard,
        )
    probability_hash = probability_sha256(reproduced)
    score = float(f1_score(
        y, reproduced.argmax(1), labels=np.arange(classes), average="macro", zero_division=0,
    ))
    if (probability_hash != selection.get("selected_probability_sha256")
            or probability_hash != winner.get("probability_sha256")
            or abs(score - float(selection.get("selected_inner_macro_f1"))) > 1e-15
            or abs(score - float(winner.get("macro_f1"))) > 1e-15):
        raise ValueError("selected inner OOF does not reproduce")
    return probability_hash


def verify_frozen_selection(config):
    """Verify the complete 541-row search and independent Fix2 commit set."""
    root = Path(config["meta_search_run_root"])
    paths = {name: root / name for name in (
        "FROZEN_SELECTION.json", "SEARCH_COMPLETE.json", "SCORE_COMPLETE.json",
        "SCORE_IDENTITY.json", "SCORE_LEDGER.jsonl",
    )}
    for key in ("meta_search_selection_sha256", "meta_search_complete_sha256"):
        if key not in config: raise ValueError(f"deployment config missing key: {key}")
    if file_sha256(paths["FROZEN_SELECTION.json"]) != config["meta_search_selection_sha256"]:
        raise ValueError("frozen selection hash mismatch")
    if file_sha256(paths["SEARCH_COMPLETE.json"]) != config["meta_search_complete_sha256"]:
        raise ValueError("meta search completion hash mismatch")
    selection = _read_json(paths["FROZEN_SELECTION.json"], "frozen selection")
    complete = _read_json(paths["SEARCH_COMPLETE.json"], "search completion")
    score_complete = _read_json(paths["SCORE_COMPLETE.json"], "score completion")
    identity = _read_json(paths["SCORE_IDENTITY.json"], "score identity")
    if (selection.get("schema_version") != "STACK7_META_FROZEN_SELECTION_V1"
            or selection.get("status") != "FROZEN"
            or complete.get("status") != "COMPLETE" or complete.get("declared_score_rows") != 541
            or complete.get("completed_fit_identities") != 162
            or score_complete.get("status") != "COMPLETE" or score_complete.get("declared_rows") != 541
            or identity.get("schema_version") != "STACK7_META_SCORE_IDENTITY_V3"):
        raise ValueError("frozen search is incomplete")
    identity_sha = canonical_sha256(identity)
    if selection.get("ledger_identity_sha256") != identity_sha or score_complete.get("identity_sha256") != identity_sha:
        raise ValueError("frozen score identity mismatch")
    bank, baseline, guard, baseline_lineage, expected_bindings = _verified_inner_context(config)
    if identity.get("bindings") != expected_bindings or selection.get("bindings") != expected_bindings:
        raise ValueError("frozen search caller/inner binding mismatch")
    scores = root / "scores"; commits = root / "score_commits"; rows = []; commit_rows = []
    for index in range(541):
        row_path = scores / f"{index:04d}.json"; commit_path = commits / f"{index:04d}.json"
        row_bytes = row_path.read_bytes(); row = json.loads(row_bytes); commit = _read_json(commit_path, "score commit")
        checksum = row.pop("row_sha256", None)
        if (commit.get("schema_version") != "STACK7_META_SCORE_COMMIT_V1"
                or commit.get("status") != "BOUND_BEFORE_ROW_PUBLISH"
                or commit.get("declared_index") != index
                or commit.get("ledger_identity_sha256") != identity_sha
                or commit.get("row_bytes_sha256") != hashlib.sha256(row_bytes).hexdigest()
                or commit.get("row_sha256") != checksum
                or row.get("declared_index") != index
                or row.get("ledger_identity_sha256") != identity_sha
                or checksum != canonical_sha256(row)):
            raise ValueError(f"score commit/row mismatch: {index:04d}")
        row["row_sha256"] = checksum; rows.append(row)
        if row.get("bindings") != expected_bindings:
            raise ValueError(f"score row caller/inner binding mismatch: {index:04d}")
        commit_rows.append({"declared_index": index, "commit_sha256": file_sha256(commit_path), "row_bytes_sha256": commit["row_bytes_sha256"]})
    if len(list(scores.glob("*.json"))) != 541 or len(list(commits.glob("*.json"))) != 541:
        raise ValueError("score/commit directory count changed")
    aliases = 0
    by_index = {row["declared_index"]: row for row in rows}
    for row in rows:
        if row.get("kind") != "meta_blend" or row.get("alpha") != 1.0:
            continue
        canonical_index = 1 + int(row["declaration_index"]) * 10 + 8
        if row.get("endpoint_canonical_index") != canonical_index or row.get("endpoint_kind") != "guarded_meta_alpha_1":
            raise ValueError("alpha=1 alias endpoint metadata changed")
        if row.get("pooling") == "arithmetic":
            if row.get("alias_of") is not None or row["declared_index"] != canonical_index:
                raise ValueError("alpha=1 alias canonical metadata changed")
        elif row.get("pooling") == "geometric":
            aliases += 1; canonical = by_index.get(canonical_index)
            if row.get("alias_of") != canonical_index or canonical is None:
                raise ValueError("alpha=1 alias linkage changed")
            for key in ("head_id", "macro_f1", "probability_sha256"):
                if row.get(key) != canonical.get(key):
                    raise ValueError(f"alpha=1 alias {key} changed")
        else:
            raise ValueError("alpha=1 alias pooling changed")
    if aliases != 54:
        raise ValueError("complete score ledger must contain 54 alpha=1 aliases")
    commit_set = canonical_sha256(commit_rows)
    ledger_bytes = b"".join(_json_bytes(row) for row in rows)
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    if (paths["SCORE_LEDGER.jsonl"].read_bytes() != ledger_bytes
            or score_complete.get("ledger_sha256") != ledger_sha
            or complete.get("ledger_sha256") != ledger_sha
            or selection.get("ledger_sha256") != ledger_sha
            or score_complete.get("score_commit_set_sha256") != commit_set
            or complete.get("score_commit_set_sha256") != commit_set
            or selection.get("score_commit_set_sha256") != commit_set):
        raise ValueError("frozen ledger/commit-set binding mismatch")
    winner = rows[0]
    for row in rows[1:]:
        if row["macro_f1"] > winner["macro_f1"]: winner = row
    expected = {
        "selected_kind": winner["kind"], "selected_head_id": winner.get("head_id"),
        "selected_declaration": winner.get("declaration"), "selected_alpha": winner.get("alpha"),
        "selected_pooling": winner.get("pooling"), "selected_inner_macro_f1": winner["macro_f1"],
        "selected_probability_sha256": winner["probability_sha256"],
        "deployment_required": winner["kind"] != "baseline_control",
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen selection does not match committed winner")
    if (complete.get("selection_sha256") != file_sha256(paths["FROZEN_SELECTION.json"])
            or complete.get("deployment_required") != selection["deployment_required"]
            or complete.get("test_read") is not False or complete.get("full_oof_used") is not False):
        raise ValueError("search completion/selection binding mismatch")
    selected_inner_sha = _verify_selected_inner_caches(
        config, root, selection, winner, bank, baseline, guard,
    )
    return selection, {"selection_sha256": file_sha256(paths["FROZEN_SELECTION.json"]), "search_complete_sha256": file_sha256(paths["SEARCH_COMPLETE.json"]), "score_commit_set_sha256": commit_set, "ledger_sha256": ledger_sha, "selected_inner_probability_sha256": selected_inner_sha, "baseline_inner_probability_sha256": baseline_lineage["baseline_probability_sha256"]}


def _load_sample(config):
    path = Path(config["sample_submission_csv"])
    if file_sha256(path) != config["sample_submission_sha256"]:
        raise ValueError("sample submission hash mismatch")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["ID", "SUBCLASS"]:
            raise ValueError("sample submission columns changed")
        ids = np.asarray([row["ID"] for row in reader], dtype="U")
    if ids.shape != (int(config["expected_test_rows"]),) or len(set(ids.tolist())) != len(ids):
        raise ValueError("sample submission ID order is invalid")
    return ids


def _load_components(config, bank, test_ids):
    receipt = _read_json(config["training_bank_receipt_json"], "training bank receipt")
    lineage_rows = receipt.get("lineage", {}).get("component_receipts", [])
    lineage = {(row.get("model"), row.get("fold")): row for row in lineage_rows}
    if len(lineage_rows) != 35 or len(lineage) != 35:
        raise ValueError("training bank must bind exactly 35 component caches")
    base_source = receipt.get("lineage", {}).get("base_source_sha256")
    if int(config["expected_classes"]) == 26 and base_source != BASE_SOURCE_SHA256:
        raise ValueError("graded base source differs from frozen literal")
    root = Path(config["v3_deployment_run_root"]) / "deployment/components"
    classes = bank["class_order"].astype(str).tolist(); output = {}
    for fold in range(5):
        valid_mask = bank["full_outer_folds"] == fold; train_mask = ~valid_mask
        fold_values = {}
        for model_index, model in enumerate(MODEL_ORDER):
            target = root / model.replace("/", "_") / f"fold_{fold}"
            npz_path, receipt_path = target / "probability.npz", target / "COMPLETE.json"
            if not npz_path.is_file() or not receipt_path.is_file():
                raise ValueError(f"missing component cache: {model} fold {fold}")
            bound = lineage.get((model, fold))
            if bound is None or bound.get("npz_sha256") != file_sha256(npz_path) or bound.get("receipt_sha256") != file_sha256(receipt_path):
                raise ValueError(f"component bank-lineage mismatch: {model} fold {fold}")
            item = _read_json(receipt_path, "component receipt")
            if item.get("schema_version") != "GRADED_COMPONENT_CACHE_V1" or item.get("npz_sha256") != file_sha256(npz_path):
                raise ValueError(f"component receipt/hash mismatch: {model} fold {fold}")
            expected = {
                "component": model, "fold": fold, "source_sha256": base_source,
                "class_order": classes,
                "train_ids_sha256": sequence_sha256(bank["full_ids"][train_mask]),
                "train_y_sha256": sequence_sha256(np.asarray(classes)[bank["full_y"][train_mask]]),
                "train_groups_sha256": sequence_sha256(bank["full_groups"][train_mask]),
                "valid_ids_sha256": sequence_sha256(bank["full_ids"][valid_mask]),
                "test_ids_sha256": sequence_sha256(test_ids),
            }
            bindings = item.get("bindings", {})
            if any(bindings.get(key) != value for key, value in expected.items()):
                raise ValueError(f"component receipt binding mismatch: {model} fold {fold}")
            try:
                with np.load(npz_path, allow_pickle=False) as stored:
                    valid = np.asarray(stored["valid"], dtype=np.float64)
                    test = np.asarray(stored["test"], dtype=np.float64)
                    guard = np.asarray(stored["guard"]) if model == "ANCHOR-C10" else None
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"component probability artifact invalid: {model} fold {fold}") from exc
            if (valid.shape != (int(valid_mask.sum()), len(classes)) or test.shape != (len(test_ids), len(classes))
                    or not np.isfinite(valid).all() or not np.isfinite(test).all()
                    or (valid < 0).any() or (test < 0).any()
                    or not np.allclose(valid.sum(1), 1, rtol=0, atol=1e-8)
                    or not np.allclose(test.sum(1), 1, rtol=0, atol=1e-8)
                    or item.get("valid_probability_sha256") != probability_sha256(valid)
                    or item.get("test_probability_sha256") != probability_sha256(test)
                    or not np.array_equal(valid, bank["full_oof_probability"][model_index, valid_mask])):
                raise ValueError(f"component probability mismatch: {model} fold {fold}")
            if model == "ANCHOR-C10":
                if guard.dtype != np.bool_ or guard.shape != (len(valid) + len(test),):
                    raise ValueError(f"component guard mismatch: {model} fold {fold}")
            fold_values[model] = {"valid": valid, "test": test, "guard": guard}
        output[fold] = fold_values
    return output, lineage_rows


def _load_saved_v3(config, bank, test_ids):
    path = Path(config["v3_predictions_npz"])
    if file_sha256(path) != config["v3_predictions_sha256"]:
        raise ValueError("V3 predictions hash mismatch")
    complete_path = Path(config["v3_deployment_complete_json"])
    if file_sha256(complete_path) != config["v3_deployment_complete_sha256"]:
        raise ValueError("V3 deployment completion hash mismatch")
    complete = _read_json(complete_path, "V3 deployment completion")
    if (complete.get("predictions_sha256") != config["v3_predictions_sha256"]
            or complete.get("submission_sha256") != config["v3_submission_sha256"]
            or complete.get("adaptive_full_oof_macro_f1") != config["expected_v3_full_oof_macro_f1"]):
        raise ValueError("V3 saved output binding changed")
    required = ("ids", "y", "groups", "folds", "class_order", "oof_probability", "test_probability")
    try:
        with np.load(path, allow_pickle=False) as stored:
            values = {name: np.asarray(stored[name]) for name in required}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("V3 saved prediction members are invalid") from exc
    for key, bank_key in (("ids", "full_ids"), ("y", "full_y"), ("groups", "full_groups"), ("folds", "full_outer_folds"), ("class_order", "class_order")):
        if not np.array_equal(values[key], bank[bank_key]):
            raise ValueError(f"V3 saved {key} identity mismatch")
    if values["oof_probability"].shape != (len(bank["full_ids"]), len(bank["class_order"])) or values["test_probability"].shape != (len(test_ids), len(bank["class_order"])):
        raise ValueError("V3 saved probability dimensions changed")
    return np.asarray(values["oof_probability"], dtype=np.float64), np.asarray(values["test_probability"], dtype=np.float64), complete


def _metrics(y, probability, classes):
    predicted = probability.argmax(1)
    per_class = f1_score(y, predicted, labels=np.arange(classes), average=None, zero_division=0)
    return {"macro_f1": float(np.mean(per_class)), "per_class_f1": [float(v) for v in per_class]}


def _fit_common(config, bank, declaration, fold, fit_mask, valid_mask):
    return {
        "source_sha256": config["meta_source_sha256"], "runtime_config_sha256": config["_runtime_config_sha256"],
        "bank_sha256": config["training_bank_sha256"], "recipe_sha256": config["v3_frozen_recipe_sha256"],
        "declaration_sha256": canonical_sha256(declaration), "fold": int(fold),
        "fit_ids_sha256": sequence_sha256(bank["full_ids"][fit_mask]), "fit_y_sha256": sequence_sha256(bank["full_y"][fit_mask]),
        "fit_groups_sha256": sequence_sha256(bank["full_groups"][fit_mask]), "fit_folds_sha256": sequence_sha256(bank["full_outer_folds"][fit_mask]),
        "valid_ids_sha256": sequence_sha256(bank["full_ids"][valid_mask]), "valid_y_sha256": sequence_sha256(bank["full_y"][valid_mask]),
        "valid_groups_sha256": sequence_sha256(bank["full_groups"][valid_mask]), "valid_folds_sha256": sequence_sha256(bank["full_outer_folds"][valid_mask]),
        "model_order": bank["model_ids"].astype(str).tolist(), "class_order": bank["class_order"].astype(str).tolist(),
    }


def _confirmation_common(config, bank, inner, declaration, valid_mask):
    """Confirmation identity uses inner feature-fold roles, not outer labels."""
    return {
        "source_sha256": config["meta_source_sha256"], "runtime_config_sha256": config["_runtime_config_sha256"],
        "bank_sha256": config["training_bank_sha256"], "recipe_sha256": config["v3_frozen_recipe_sha256"],
        "declaration_sha256": canonical_sha256(declaration), "fold": 0,
        "fit_ids_sha256": sequence_sha256(inner["inner_ids"]), "fit_y_sha256": sequence_sha256(inner["inner_y"]),
        "fit_groups_sha256": sequence_sha256(inner["inner_groups"]), "fit_folds_sha256": sequence_sha256(inner["inner_folds"]),
        "valid_ids_sha256": sequence_sha256(bank["full_ids"][valid_mask]), "valid_y_sha256": sequence_sha256(bank["full_y"][valid_mask]),
        "valid_groups_sha256": sequence_sha256(bank["full_groups"][valid_mask]), "valid_folds_sha256": sequence_sha256(bank["full_outer_folds"][valid_mask]),
        "model_order": bank["model_ids"].astype(str).tolist(), "class_order": bank["class_order"].astype(str).tolist(),
    }


def _enforce_production_constants(config):
    if int(config.get("expected_classes", 0)) != 26:
        return
    expected = {
        "sample_submission_sha256": SAMPLE_SUBMISSION_SHA256,
        "v3_predictions_sha256": V3_PREDICTIONS_SHA256,
        "v3_submission_sha256": V3_SUBMISSION_SHA256,
        "v3_stage_deployment_sha256": STAGE_DEPLOYMENT_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"production exact value changed: {key}")


def _csv_bytes(test_ids, probability, class_order):
    stream = io.StringIO(newline=""); writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["ID", "SUBCLASS"])
    labels = np.asarray(class_order).astype(str)[np.asarray(probability).argmax(1)]
    writer.writerows(zip(np.asarray(test_ids).astype(str).tolist(), labels.tolist()))
    return stream.getvalue().encode("utf-8")


def publish_outputs(run_root, arrays, receipt, progress_callback=None):
    """Publish NPZ and CSV only through a durable two-file pending binding."""
    root = Path(run_root); root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "predictions.npz"; csv_path = root / "submission_stack7_frozen.csv"
    pending_path = root / "OUTPUT_PENDING.json"; complete_path = root / "COMPLETE.json"
    if complete_path.is_file():
        complete = _read_json(complete_path, "deployment completion")
        if file_sha256(npz_path) != complete.get("predictions_sha256"):
            raise ValueError("completed predictions hash mismatch")
        if file_sha256(csv_path) != complete.get("submission_sha256"):
            raise ValueError("completed submission hash mismatch")
        expected_base = dict(complete); expected_base.pop("predictions_sha256", None); expected_base.pop("submission_sha256", None)
        if expected_base != receipt:
            raise ValueError("completed deployment receipt binding changed")
        return complete
    npz_partial = root / ".predictions.npz.partial"; csv_partial = root / ".submission_stack7_frozen.csv.partial"
    if pending_path.is_file():
        pending = _read_json(pending_path, "output pending receipt")
        if pending.get("receipt_sha256") != canonical_sha256(receipt):
            raise ValueError("pending output receipt binding changed")
        for final, partial, key in ((npz_path, npz_partial, "predictions_sha256"), (csv_path, csv_partial, "submission_sha256")):
            candidate = final if final.is_file() else partial
            if not candidate.is_file() or file_sha256(candidate) != pending.get(key):
                raise ValueError("pending output artifact hash mismatch")
            if not final.is_file(): os.replace(partial, final)
            if progress_callback is not None: progress_callback("npz_published" if final == npz_path else "csv_published")
        complete = {**receipt, "predictions_sha256": pending["predictions_sha256"], "submission_sha256": pending["submission_sha256"]}
        _write_once(complete_path, _json_bytes(complete)); return complete
    if any(path.exists() for path in (npz_path, csv_path, npz_partial, csv_partial)):
        raise ValueError("unbound final output artifacts retained")
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); npz_payload = buffer.getvalue()
    csv_payload = _csv_bytes(arrays["test_ids"], arrays["test_probability"], arrays["class_order"])
    for path, payload in ((npz_partial, npz_payload), (csv_partial, csv_payload)):
        with path.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    pending = {"schema_version": "STACK7_META_OUTPUT_PENDING_V1", "status": "BOUND_BEFORE_PUBLISH", "receipt_sha256": canonical_sha256(receipt), "predictions_sha256": file_sha256(npz_partial), "submission_sha256": file_sha256(csv_partial)}
    _write_once(pending_path, _json_bytes(pending))
    os.replace(npz_partial, npz_path)
    if progress_callback is not None: progress_callback("npz_published")
    os.replace(csv_partial, csv_path)
    if progress_callback is not None: progress_callback("csv_published")
    complete = {**receipt, "predictions_sha256": pending["predictions_sha256"], "submission_sha256": pending["submission_sha256"]}
    _write_once(complete_path, _json_bytes(complete)); return complete


def run_deployment(config, run_root):
    """Run the already-frozen winner; never fit or discover a base model."""
    selection, search_binding = verify_frozen_selection(config)  # first external read gate
    run_root = Path(run_root); run_root.mkdir(parents=True, exist_ok=True)
    if not selection["deployment_required"]:
        receipt = {"schema_version": "STACK7_META_NO_DEPLOYMENT_V1", "status": "COMPLETE", "reason": "FROZEN_BASELINE_RETAINED", **search_binding, "test_read": False, "full_oof_used": False, "base_fit_count": 0, "website_submitted": False}
        _write_once(run_root / "NO_DEPLOYMENT.json", _json_bytes(receipt)); return receipt
    _enforce_production_constants(config)
    bank = load_full_bank(config); inner = load_inner_bank(config)
    if bank["model_ids"].astype(str).tolist() != list(MODEL_ORDER):
        raise ValueError("full bank model order changed")
    folds = bank["full_outer_folds"].astype(np.int64); y = bank["full_y"].astype(np.int64)
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise ValueError("full bank outer folds differ from 0..4")
    for fold in range(5):
        if set(np.unique(y[folds != fold]).tolist()) != set(range(len(bank["class_order"]))):
            raise ValueError(f"full bank training partition lacks class support: {fold}")
    test_ids = _load_sample(config)
    components, component_lineage = _load_components(config, bank, test_ids)
    saved_oof, saved_test, v3_complete = _load_saved_v3(config, bank, test_ids)
    baseline_oof = np.zeros_like(saved_oof); baseline_test_sum = np.zeros_like(saved_test); baseline_by_fold = {}
    for fold in range(5):
        values = components[fold]; raw_valid = {m: values[m]["valid"] for m in MODEL_ORDER}; raw_test = {m: values[m]["test"] for m in MODEL_ORDER}
        boundary = len(raw_valid["ANCHOR-C10"]); guard = values["ANCHOR-C10"]["guard"]
        valid = replay_v3_fold(config, raw_valid, guard[:boundary])
        test = replay_v3_fold(config, raw_test, guard[boundary:])
        mask = folds == fold; baseline_oof[mask] = valid; baseline_test_sum += test; baseline_by_fold[fold] = (valid, test, guard[:boundary], guard[boundary:])
    baseline_test = baseline_test_sum / 5.0; baseline_test /= baseline_test.sum(1, keepdims=True)
    if (np.max(np.abs(baseline_oof - saved_oof)) > 1e-12 or np.max(np.abs(baseline_test - saved_test)) > 1e-12
            or not np.array_equal(baseline_oof.argmax(1), saved_oof.argmax(1))
            or not np.array_equal(baseline_test.argmax(1), saved_test.argmax(1))):
        raise ValueError("V3 full OOF/test replay does not reproduce")
    declaration = selection["selected_declaration"]; alpha = selection["selected_alpha"]; pooling = selection["selected_pooling"]
    fit_callback = config.get("fit_meta_callback", fit_meta); predict_callback = config.get("predict_meta_callback", predict_meta)
    if not callable(fit_callback) or not callable(predict_callback): raise ValueError("meta deployment callbacks must be callable")
    oof = np.zeros_like(saved_oof); coverage = np.zeros(len(y), dtype=np.uint8); test_sum = np.zeros_like(saved_test); fit_receipts = []
    fits_root = run_root / "fits"
    for fold in range(5):
        fit_mask, valid_mask = folds != fold, folds == fold
        fit_p, valid_p = bank["full_oof_probability"][:, fit_mask, :], bank["full_oof_probability"][:, valid_mask, :]
        identity = deployment_fit_identity(_fit_common(config, bank, declaration, fold, fit_mask, valid_mask), "full_meta_outer_cv", fit_p, valid_p)
        model, meta_valid, fit_receipt = load_or_fit_deployment_head(fits_root, identity, lambda p=fit_p, m=fit_mask: fit_callback(declaration, p, y[m]), lambda artifact, p=valid_p: predict_callback(artifact, p))
        test_p = np.stack([components[fold][model_id]["test"] for model_id in MODEL_ORDER])
        meta_test = predict_callback(model, test_p)
        base_valid, base_test, guard_valid, guard_test = baseline_by_fold[fold]
        oof[valid_mask] = blend(base_valid, meta_valid, alpha, pooling, guard_valid); coverage[valid_mask] += 1
        test_sum += blend(base_test, meta_test, alpha, pooling, guard_test)
        fit_receipts.append({"feature_role": "full_meta_outer_cv", "fold": fold, "identity_sha256": fit_receipt["identity_sha256"], "receipt_sha256": file_sha256(Path(fit_receipt["model_path"]).parent / "COMPLETE.json")})
    if not np.all(coverage == 1): raise ValueError("full meta OOF rows were not filled exactly once")
    test_probability = test_sum / 5.0; test_probability /= test_probability.sum(1, keepdims=True)
    valid0 = folds == 0; fit0 = ~valid0
    if not (np.array_equal(inner["inner_ids"], bank["full_ids"][fit0]) and np.array_equal(inner["inner_y"], y[fit0]) and np.array_equal(inner["inner_groups"], bank["full_groups"][fit0])):
        raise ValueError("confirmation inner/full outer0 identities differ")
    common = _confirmation_common(config, bank, inner, declaration, valid0)
    confirmation_identity = deployment_fit_identity(common, "outer0_confirmation_inner_oof", inner["inner_probability"], bank["full_oof_probability"][:, valid0, :])
    _, confirmation_meta, confirmation_receipt = load_or_fit_deployment_head(fits_root, confirmation_identity, lambda: fit_callback(declaration, inner["inner_probability"], inner["inner_y"]), lambda artifact: predict_callback(artifact, bank["full_oof_probability"][:, valid0, :]))
    confirmation_blend = blend(baseline_by_fold[0][0], confirmation_meta, alpha, pooling, baseline_by_fold[0][2])
    fit_receipts.append({"feature_role": "outer0_confirmation_inner_oof", "fold": 0, "identity_sha256": confirmation_receipt["identity_sha256"], "receipt_sha256": file_sha256(Path(confirmation_receipt["model_path"]).parent / "COMPLETE.json")})
    full_metrics = _metrics(y, oof, len(bank["class_order"])); fold_metrics = [_metrics(y[folds == f], oof[folds == f], len(bank["class_order"])) for f in range(5)]
    confirmation = {"evidence": CONFIRMATION_EVIDENCE, "standalone_meta": _metrics(y[valid0], confirmation_meta, len(bank["class_order"])), "selected_blend": _metrics(y[valid0], confirmation_blend, len(bank["class_order"])), "used_for_selection": False}
    arrays = {
        "ids": np.asarray(bank["full_ids"], dtype="U"), "y": y, "groups": np.asarray(bank["full_groups"], dtype="U"), "folds": folds,
        "class_order": np.asarray(bank["class_order"], dtype="U"), "model_order": np.asarray(bank["model_ids"], dtype="U"),
        "oof_probability": oof, "baseline_oof_probability": baseline_oof,
        "test_ids": np.asarray(test_ids, dtype="U"), "test_probability": test_probability, "baseline_test_probability": baseline_test,
    }
    receipt = {
        "schema_version": "STACK7_META_DEPLOYMENT_COMPLETE_V1", "status": "COMPLETE",
        "validation_evidence": FULL_EVIDENCE, "selected_recipe": {"head_id": selection["selected_head_id"], "declaration": declaration, "alpha": alpha, "pooling": pooling, "inner_macro_f1": selection["selected_inner_macro_f1"]},
        "adaptive_full_oof": full_metrics, "fold_metrics": fold_metrics, "confirmation": confirmation,
        "fit_counts": {"base": 0, "full_meta": 5, "confirmation_meta": 1}, "fit_receipts": fit_receipts,
        "input_hashes": {"training_bank_sha256": config["training_bank_sha256"], "training_bank_receipt_sha256": config["training_bank_receipt_sha256"], "sample_submission_sha256": config["sample_submission_sha256"], "v3_predictions_sha256": config["v3_predictions_sha256"], "v3_deployment_complete_sha256": config["v3_deployment_complete_sha256"]},
        "source_hashes": {"meta_source_sha256": config["meta_source_sha256"], "runtime_config_sha256": config["_runtime_config_sha256"], "base_source_sha256": v3_complete.get("base_source_sha256"), "v3_source_sha256": config["v3_source_sha256"], "v3_frozen_recipe_sha256": config["v3_frozen_recipe_sha256"]},
        "cache_hashes": {"component_receipts_sha256": canonical_sha256(component_lineage), **search_binding},
        "test_used_for_selection": False, "full_oof_used_for_recipe_selection": False, "confirmation_used_for_selection": False,
        "base_refit": False, "website_submitted": False, "prior_csv_preserved": True,
    }
    return publish_outputs(run_root, arrays, receipt, config.get("progress_callback"))
