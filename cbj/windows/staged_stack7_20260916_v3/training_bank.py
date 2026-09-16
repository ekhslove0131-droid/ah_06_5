"""Prepare an immutable train-only probability bank for seven completed models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np


MODEL_ORDER = (
    "ANCHOR-H1", "ANCHOR-C10", "OLD-G3-0014", "OLD-G3-0018",
    "TEAM-T2-0005", "TEAM-T2-0007", "TEAM-T4-0007",
)
TRAIN_SHA256 = "92418b8441d058cfc68e939dd88725610750be4bc8edc51253cffc72fc4fc0ab"
REFERENCE_SHA256 = "ab952812c5b779bbaee826bc9163579268cffbf50dd567fb2162c5600bec08e9"
EVIDENCE_SHA256 = "461c5930b0dd094f13c9659eb720676b624efac9ccceddf4f067f59659dd914f"
BASE_SOURCE_SHA256 = "b56d607e5da1edf3f0e4c97e426182896f094ba35faf038735ceabe03b343b52"


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _read(path): return json.loads(Path(path).read_text())


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _canonical(value): return hashlib.sha256(_json_bytes(value)).hexdigest()
def _sequence(value): return _canonical(np.asarray(value).tolist())
def _probability_hash(value): return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _verify_probability(value, rows, classes, name):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (rows, classes) or not np.isfinite(value).all() or (value < 0).any()
            or not np.allclose(value.sum(1), 1, atol=1e-8, rtol=0)):
        raise ValueError(f"{name} probability is invalid")
    return value


def _component_paths(context, model, fold):
    root = Path(context["component_root"]) / context.get("component_subpath", "deployment/components")
    target = root / model.replace("/", "_") / f"fold_{fold}"
    return target / "probability.npz", target / "COMPLETE.json"


def _load_component_valid(context, model, outer):
    fold = int(outer["fold"]); path, receipt_path = _component_paths(context, model, fold)
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError(f"missing component cache: {model} fold {fold}")
    receipt = _read(receipt_path)
    if receipt.get("npz_sha256") != _sha(path):
        raise ValueError(f"component artifact hash mismatch: {model} fold {fold}")
    data = context["data"]; train = np.asarray(outer["train_indices"], dtype=np.int64)
    valid = np.asarray(outer["valid_indices"], dtype=np.int64)
    expected = {
        "component": model,
        "declaration_sha256": context["declaration_hashes"][model],
        "fold": fold,
        "source_sha256": context["base_source_sha256"],
        "train_ids_sha256": context["sequence_sha256"](data["ids"][train]),
        "train_y_sha256": context["sequence_sha256"](data["y_label"][train]),
        "train_groups_sha256": context["sequence_sha256"](data["groups"][train]),
        "valid_ids_sha256": context["sequence_sha256"](data["ids"][valid]),
        "class_order": list(data["class_order"]),
    }
    bindings = receipt.get("bindings", {})
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError(f"component receipt binding mismatch: {model} fold {fold}")
    # Hashing the complete NPZ is allowed. Only the train-valid member is read;
    # the existing test member is intentionally never selected or copied.
    with np.load(path, allow_pickle=False) as stored:
        valid_probability = np.asarray(stored["valid"], dtype=np.float64)
    valid_probability = _verify_probability(
        valid_probability, len(valid), len(data["class_order"]), f"{model} fold {fold}",
    )
    if receipt.get("valid_probability_sha256") != _probability_hash(valid_probability):
        raise ValueError(f"component valid probability hash mismatch: {model} fold {fold}")
    lineage = {
        "model": model, "fold": fold, "npz_sha256": receipt["npz_sha256"],
        "receipt_sha256": _sha(receipt_path), "valid_probability_sha256": receipt["valid_probability_sha256"],
    }
    return valid_probability, lineage


def _assemble_bank(context):
    data, partitions, reference, evidence = (
        context["data"], context["partitions"], context["reference"], context["evidence"],
    )
    classes = len(data["class_order"]); rows = len(data["ids"])
    if classes < 2 or rows == 0 or set(np.unique(data["folds"]).tolist()) != set(range(5)):
        raise ValueError("full reference folds are invalid")
    for field, data_value, reference_value in (
        ("ids", data["ids"], reference["ids"]), ("y", data["y_int"], reference["y"]),
        ("groups", data["groups"], reference["groups"]), ("folds", data["folds"], reference["folds"]),
        ("class_order", np.asarray(data["class_order"]), reference["class_order"]),
    ):
        if not np.array_equal(np.asarray(data_value), np.asarray(reference_value)):
            raise ValueError(f"full reference {field} identity mismatch")
    full_group_fold = {}
    for group, fold in zip(reference["groups"], reference["folds"]):
        key, value = str(group), int(fold)
        if full_group_fold.setdefault(key, value) != value:
            raise ValueError("full reference group crosses outer folds")
    outer_rows = partitions.get("outer", [])
    if len(outer_rows) != 5 or {int(row["fold"]) for row in outer_rows} != set(range(5)):
        raise ValueError("full OOF coverage partition mismatch")
    all_indices = np.arange(rows, dtype=np.int64)
    for outer_row in outer_rows:
        fold = int(outer_row["fold"])
        expected_valid = np.flatnonzero(np.asarray(reference["folds"]) == fold)
        expected_train = np.flatnonzero(np.asarray(reference["folds"]) != fold)
        if (not np.array_equal(np.asarray(outer_row["valid_indices"], dtype=np.int64), expected_valid)
                or not np.array_equal(np.asarray(outer_row["train_indices"], dtype=np.int64), expected_train)
                or len(np.union1d(outer_row["valid_indices"], outer_row["train_indices"])) != len(all_indices)):
            raise ValueError(f"outer partition/reference fold mismatch: {fold}")
    outer0 = next(row for row in outer_rows if int(row["fold"]) == 0)
    inner_indices = np.asarray(outer0["train_indices"], dtype=np.int64)
    reference_inner = np.flatnonzero(np.asarray(reference["folds"]) != 0)
    if not np.array_equal(inner_indices, reference_inner):
        raise ValueError("inner/reference partition identity mismatch")
    for field, actual, expected in (
        ("ids", evidence["ids"], reference["ids"][reference_inner]),
        ("y", evidence["y"], reference["y"][reference_inner]),
        ("groups", evidence["groups"], reference["groups"][reference_inner]),
        ("outer_folds", evidence["outer_folds"], reference["folds"][reference_inner]),
        ("class_order", evidence["class_order"], reference["class_order"]),
    ):
        if not np.array_equal(np.asarray(actual), np.asarray(expected)):
            raise ValueError(f"inner {field} identity mismatch")
    inner_folds = np.asarray(evidence["inner_folds"], dtype=np.int16)
    if inner_folds.shape != (len(inner_indices),) or set(np.unique(inner_folds).tolist()) != {0, 1, 2}:
        raise ValueError("inner fold identity mismatch")
    derived_inner_folds = np.full(len(inner_indices), -1, dtype=np.int16)
    derived_coverage = np.zeros(len(inner_indices), dtype=np.uint8)
    inner_local = {int(value): index for index, value in enumerate(inner_indices)}
    for inner_row in outer0.get("inner", []):
        positions = np.asarray([inner_local[int(value)] for value in inner_row["valid_indices"]], dtype=np.int64)
        derived_inner_folds[positions] = int(inner_row["fold"]); derived_coverage[positions] += 1
    if not np.all(derived_coverage == 1) or not np.array_equal(inner_folds, derived_inner_folds):
        raise ValueError("inner fold assignment differs from frozen partitions")
    group_fold = {}
    for group, fold in zip(evidence["groups"], inner_folds):
        key, value = str(group), int(fold)
        if group_fold.setdefault(key, value) != value: raise ValueError("inner group crosses folds")

    candidate_ids = np.asarray(evidence["candidate_ids"]).astype(str).tolist()
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate evidence IDs are duplicated")
    candidate_map = {name: index for index, name in enumerate(candidate_ids)}
    required_candidates = list(MODEL_ORDER[2:])
    if not set(required_candidates).issubset(candidate_map):
        raise ValueError("candidate evidence order/membership is incomplete")
    candidate_probability = np.asarray(evidence["candidate_probabilities"], dtype=np.float64)
    if candidate_probability.shape != (len(candidate_ids), len(inner_indices), classes):
        raise ValueError("candidate evidence dimensions mismatch")
    inner = np.zeros((7, len(inner_indices), classes), dtype=np.float64)
    inner_coverage = np.zeros((2, len(inner_indices)), dtype=np.uint8)
    local = inner_local
    for inner_row in outer0.get("inner", []):
        fit = np.asarray(inner_row["train_indices"], dtype=np.int64)
        valid = np.asarray(inner_row["valid_indices"], dtype=np.int64)
        positions = np.asarray([local[int(value)] for value in valid], dtype=np.int64)
        partition_sha = context["canonical_sha256"](inner_row)
        for model_index, (model, short) in enumerate((("ANCHOR-H1", "H1"), ("ANCHOR-C10", "C10"))):
            namespace = f"outer0-inner{inner_row['fold']}-{short}"
            value = context["anchor_probability"](
                model, inner_row, fit, valid, namespace,
                context["canonical_sha256"]([partition_sha, short]),
            )
            inner[model_index, positions] = _verify_probability(value, len(valid), classes, namespace)
            inner_coverage[model_index, positions] += 1
    if not np.all(inner_coverage == 1): raise ValueError("inner anchor coverage mismatch")
    for offset, model in enumerate(required_candidates, start=2):
        inner[offset] = _verify_probability(
            candidate_probability[candidate_map[model]], len(inner_indices), classes, model,
        )

    full = np.zeros((7, rows, classes), dtype=np.float64)
    coverage = np.zeros((7, rows), dtype=np.uint8); component_lineage = []
    for outer in outer_rows:
        valid = np.asarray(outer["valid_indices"], dtype=np.int64)
        for model_index, model in enumerate(MODEL_ORDER):
            value, lineage = _load_component_valid(context, model, outer)
            full[model_index, valid] = value; coverage[model_index, valid] += 1
            component_lineage.append(lineage)
    if not np.all(coverage == 1): raise ValueError("full OOF coverage mismatch")
    bank = {
        "model_ids": np.asarray(MODEL_ORDER, dtype="U"),
        "class_order": np.asarray(data["class_order"], dtype="U"),
        "inner_ids": np.asarray(evidence["ids"], dtype="U"), "inner_y": np.asarray(evidence["y"], dtype=np.int64),
        "inner_groups": np.asarray(evidence["groups"], dtype="U"),
        "inner_outer_folds": np.asarray(evidence["outer_folds"], dtype=np.int16),
        "inner_folds": inner_folds, "inner_probability": inner,
        "full_ids": np.asarray(reference["ids"], dtype="U"), "full_y": np.asarray(reference["y"], dtype=np.int64),
        "full_groups": np.asarray(reference["groups"], dtype="U"),
        "full_outer_folds": np.asarray(reference["folds"], dtype=np.int16),
        "full_oof_probability": full,
    }
    if any("test" in key.lower() for key in bank): raise AssertionError("train-only bank contains a test field")
    lineage = {
        "model_order": list(MODEL_ORDER), "component_receipts": component_lineage,
        "anchor_receipts": context.get("anchor_receipts", []),
        "base_source_sha256": context["base_source_sha256"],
        "test_probability_read": False, "test_data_read": False,
    }
    return bank, lineage


def _verify_npz_exact(path, expected):
    raw = Path(path).read_bytes(); marker = raw.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(raw): raise ValueError("training bank NPZ changed")
    comment_length = int.from_bytes(raw[marker + 20:marker + 22], "little")
    if marker + 22 + comment_length != len(raw): raise ValueError("training bank NPZ changed")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None: raise ValueError("training bank NPZ changed")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != set(expected): raise ValueError("training bank fields changed")
        for key, value in expected.items():
            if not np.array_equal(stored[key], value): raise ValueError("training bank arrays changed")


def _save_bank(output_root, bank, lineage, identity):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    bank_path = output_root / "TRAINING_BANK.npz"; receipt_path = output_root / "TRAINING_BANK.json"
    pending_path = output_root / "TRAINING_BANK.PENDING.json"
    temporary = bank_path.with_name(f".{bank_path.name}.partial")
    array_sha256 = {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for key, value in bank.items()}
    binding = {
        "identity": identity, "lineage": lineage, "array_sha256": array_sha256,
        "keys": sorted(bank), "model_order": list(MODEL_ORDER),
        "inner_shape": list(bank["inner_probability"].shape),
        "full_shape": list(bank["full_oof_probability"].shape),
        "test_probability_read": False, "test_data_read": False,
    }
    if bank_path.is_file():
        if not pending_path.is_file():
            raise ValueError("unbound training bank NPZ retained; pending provenance is missing")
        pending = _read(pending_path)
        if pending.get("identity") != identity:
            raise ValueError("training bank producer identity differs")
        if any(pending.get(key) != value for key, value in binding.items() if key != "identity"):
            raise ValueError("training bank pending provenance differs")
        if pending.get("bank_sha256") != _sha(bank_path):
            raise ValueError("training bank pending hash differs")
        _verify_npz_exact(bank_path, bank)
    else:
        existing_partial = temporary.is_file()
        if temporary.is_file():
            _verify_npz_exact(temporary, bank)
        elif pending_path.is_file():
            raise ValueError("pending training bank provenance exists without its bound NPZ")
        else:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **bank); stream.flush(); os.fsync(stream.fileno())
        bank_sha256 = _sha(temporary)
        pending = {
            "schema_version": "STACK7_TRAINING_BANK_PENDING_V3", "status": "BOUND_BEFORE_PUBLISH",
            "bank_sha256": bank_sha256, **binding,
        }
        pending_payload = _json_bytes(pending)
        pending_temp = pending_path.with_name(f".{pending_path.name}.partial")
        if pending_path.is_file():
            current = _read(pending_path)
            if current.get("identity") != identity: raise ValueError("training bank producer identity differs")
            if pending_path.read_bytes() != pending_payload: raise ValueError("training bank pending provenance differs")
        elif pending_temp.is_file():
            if pending_temp.read_bytes() != pending_payload:
                raise ValueError("training bank pending partial differs; preserving foreign bytes")
            os.replace(pending_temp, pending_path)
        elif existing_partial:
            raise ValueError("unbound partial training bank NPZ retained; durable pending provenance is missing")
        else:
            with pending_temp.open("wb") as stream:
                stream.write(pending_payload); stream.flush(); os.fsync(stream.fileno())
            os.replace(pending_temp, pending_path)
        os.replace(temporary, bank_path)
    pending_sha256 = _sha(pending_path)
    receipt = {
        "schema_version": "STACK7_TRAINING_BANK_V2", "status": "COMPLETE",
        "bank_sha256": _sha(bank_path), "pending_receipt_sha256": pending_sha256, **binding,
    }
    payload = _json_bytes(receipt); partial_receipt = receipt_path.with_name(f".{receipt_path.name}.partial")
    if receipt_path.is_file():
        current = _read(receipt_path)
        if current.get("identity") != identity: raise ValueError("training bank producer identity differs")
        if receipt_path.read_bytes() != payload: raise ValueError("training bank receipt changed")
    elif partial_receipt.is_file():
        if partial_receipt.read_bytes() != payload: raise ValueError("partial training bank receipt changed")
        os.replace(partial_receipt, receipt_path)
    else:
        with partial_receipt.open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(partial_receipt, receipt_path)
    return receipt


def _allowlist_source_identity(root):
    root = Path(root); rows = []
    for name in [line.strip() for line in (root / "SOURCE_ALLOWLIST.txt").read_text().splitlines() if line.strip()]:
        path = root / name
        if not path.is_file(): raise ValueError(f"allowlisted source missing: {name}")
        rows.append({"path": name, "sha256": _sha(path)})
    return hashlib.sha256(_json_bytes(rows)).hexdigest(), rows


def _require_base_source(current, prepared, configured):
    if not (current == prepared == configured == BASE_SOURCE_SHA256):
        raise ValueError("graded V2 base source differs from the frozen literal")
    return current


def _readonly_anchor_artifact_map(config, *, old_source_sha, old_parser_sha, train_sha, make_cache_key):
    artifacts = Path(config["old_run_root"]) / "cache/outer_0/anchors/artifacts"
    if not artifacts.is_dir():
        raise ValueError("old anchor cache artifacts are missing")
    rows, receipts = {}, []
    complete_paths = sorted(artifacts.glob("*/COMPLETE.json"))
    if len(complete_paths) != 15:
        raise ValueError(f"old anchor cache must contain exactly 15 fits, got {len(complete_paths)}")
    for complete in complete_paths:
        receipt = _read(complete); bindings = receipt.get("bindings", {})
        if (bindings.get("source_sha256") != old_source_sha
                or bindings.get("input_sha256") != train_sha
                or bindings.get("parser_sha256") != old_parser_sha):
            raise ValueError("old anchor input/parser/source identity changed")
        key = receipt.get("key")
        if key != complete.parent.name or key != make_cache_key(bindings, supervised=True):
            raise ValueError("old anchor cache key does not match receipt bindings")
        probability_path = complete.parent / "probability.npy"
        if not probability_path.is_file() or receipt.get("probability_sha256") != _sha(probability_path):
            raise ValueError("old anchor probability hash changed")
        probability = np.load(probability_path, allow_pickle=False)
        if probability.shape != (receipt.get("rows"), receipt.get("classes")):
            raise ValueError("old anchor probability shape changed")
        lookup = (
            bindings["declaration_sha256"], bindings["fit_ids_sha256"], bindings["fit_y_sha256"],
            bindings["fit_groups_sha256"], bindings["partition_sha256"], bindings["parameters_sha256"],
        )
        if lookup in rows: raise ValueError("old anchor cache contains an ambiguous fit identity")
        rows[lookup] = probability
        receipts.append({
            "key": key, "complete_sha256": _sha(complete),
            "probability_sha256": receipt["probability_sha256"],
        })
    return rows, receipts


def _verified_context(config):
    if config.get("automatic_website_submission") is not False:
        raise ValueError("automatic website submission must remain disabled")
    v2_root = Path(config["v2_root"]); v2_run = Path(config["v2_run_root"])
    v2_config_path = Path(config["v2_runtime_config"]); prepared = _read(v2_run / "PREPARED.json")
    sys.path.insert(0, str(v2_root))
    from anchor_pipeline import anchor_model_recipes
    from atomic_state import read_json
    from grade_runner import (
        OLD_PARSER_SHA256, OLD_SOURCE_SHA256, _declarations, _old_anchor_probability, _source_identity,
    )
    from base_receipts import canonical_sha256, sequence_sha256
    from cache_store import make_cache_key
    from partition_contract import build_partitions
    from source_closure import sha256_file
    from workflow import _load_training

    current_v2_source, _ = _source_identity(v2_root)
    _require_base_source(current_v2_source, prepared["source_sha256"], config["base_source_sha256"])
    if sha256_file(v2_config_path) != prepared["runtime_config_sha256"]:
        raise ValueError("graded V2 runtime config changed")
    v2_config = read_json(v2_config_path)
    if v2_config.get("train_sha256") != TRAIN_SHA256 or sha256_file(v2_config["train_csv"]) != TRAIN_SHA256:
        raise ValueError("train input hash changed")
    reference_path = Path(v2_config["reference_oof"])
    if sha256_file(reference_path) != REFERENCE_SHA256: raise ValueError("reference input hash changed")
    evidence_path = Path(config["evidence_npz"])
    if sha256_file(evidence_path) != EVIDENCE_SHA256 or EVIDENCE_SHA256 != config["evidence_sha256"]:
        raise ValueError("inner evidence hash changed")
    evidence_receipt = read_json(config["evidence_receipt_json"])
    if evidence_receipt.get("evidence_sha256") != EVIDENCE_SHA256: raise ValueError("inner evidence receipt changed")

    v3_source_sha, _ = _allowlist_source_identity(config["v3_source_root"])
    if v3_source_sha != config["v3_source_sha256"]: raise ValueError("V3 source identity changed")
    if sha256_file(config["v3_runtime_config"]) != config["v3_runtime_config_sha256"]:
        raise ValueError("V3 runtime config changed")
    v3_complete = read_json(Path(config["v3_deployment_run_root"]) / "COMPLETE.json")
    if (v3_complete.get("status") != "COMPLETE" or v3_complete.get("base_source_sha256") != current_v2_source
            or v3_complete.get("meta_source_sha256") != v3_source_sha
            or v3_complete.get("runtime_config_sha256") != config["v3_runtime_config_sha256"]):
        raise ValueError("V3 deployment receipt binding changed")

    data = _load_training(v2_config)
    partitions = build_partitions(data["ids"], data["y_label"], data["groups"], data["folds"], data["class_order"])
    with np.load(reference_path, allow_pickle=False) as stored:
        reference = {name: np.asarray(stored[name]) for name in ("ids", "y", "groups", "folds", "class_order")}
    with np.load(evidence_path, allow_pickle=False) as stored:
        names = ("ids", "y", "groups", "outer_folds", "inner_folds", "class_order", "candidate_ids", "candidate_probabilities")
        evidence = {name: np.asarray(stored[name]) for name in names}
    if (len(evidence["ids"]) != 4959 or len(evidence["candidate_ids"]) != 124
            or evidence["candidate_probabilities"].shape != (124, 4959, 26)
            or len(reference["ids"]) != 6201 or len(reference["class_order"]) != 26):
        raise ValueError("training bank input dimensions changed")
    _, _, declarations = _declarations(v2_root)
    anchor_recipes = anchor_model_recipes()
    anchor_map, old_anchor_receipts = _readonly_anchor_artifact_map(
        v2_config, old_source_sha=OLD_SOURCE_SHA256, old_parser_sha=OLD_PARSER_SHA256,
        train_sha=TRAIN_SHA256, make_cache_key=make_cache_key,
    )
    declaration_hashes = {
        "ANCHOR-H1": anchor_recipes["H1"]["config_hash"],
        "ANCHOR-C10": anchor_recipes["C10"]["config_hash"],
    }
    for name in MODEL_ORDER[2:]: declaration_hashes[name] = declarations[name]["config_hash"]

    def anchor_probability(model, inner, fit, valid, namespace, partition_sha):
        short = model.removeprefix("ANCHOR-")
        value = _old_anchor_probability(anchor_map, anchor_recipes[short], data, fit, namespace, partition_sha)
        return _verify_probability(value, len(valid), 26, namespace)

    return {
        "data": data, "partitions": partitions, "reference": reference, "evidence": evidence,
        "anchor_probability": anchor_probability, "anchor_receipts": old_anchor_receipts,
        "declaration_hashes": declaration_hashes, "base_source_sha256": current_v2_source,
        "component_root": config["v3_deployment_run_root"], "component_subpath": "deployment/components",
        "sequence_sha256": sequence_sha256, "canonical_sha256": canonical_sha256,
        "identity": {
            "train_sha256": TRAIN_SHA256, "reference_sha256": REFERENCE_SHA256,
            "evidence_sha256": EVIDENCE_SHA256, "v2_source_sha256": current_v2_source,
            "v2_runtime_config_sha256": prepared["runtime_config_sha256"],
            "v3_source_sha256": v3_source_sha, "v3_runtime_config_sha256": config["v3_runtime_config_sha256"],
        },
    }


def preflight(config):
    try:
        context = _verified_context(config); bank, lineage = _assemble_bank(context)
    except Exception as error:
        return {"schema_version": "STACK7_PREFLIGHT_V1", "status": "BLOCKED", "ready": False, "blocker": str(error)}
    return {
        "schema_version": "STACK7_PREFLIGHT_V1", "status": "READY", "ready": True,
        "models": list(MODEL_ORDER), "inner_shape": list(bank["inner_probability"].shape),
        "full_shape": list(bank["full_oof_probability"].shape),
        "component_receipts": len(lineage["component_receipts"]), "anchor_receipts": len(lineage["anchor_receipts"]),
        "test_probability_read": False, "test_data_read": False,
    }


def prepare_bank(config, output_root):
    source_sha, source_rows = _allowlist_source_identity(Path(__file__).parent)
    if source_sha != config["meta_source_sha256"]: raise ValueError("stack7 source identity changed")
    context = _verified_context(config); bank, lineage = _assemble_bank(context)
    identity = {
        **context["identity"], "meta_source_sha256": source_sha,
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "source_files": source_rows,
    }
    return _save_bank(output_root, bank, lineage, identity)
