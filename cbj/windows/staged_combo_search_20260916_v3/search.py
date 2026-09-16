"""Finite, resumable C0/C1/C2 train-only probability composition search."""

from __future__ import annotations

import json
import hashlib
import io
import math
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from alias_resolution import execution_row
from completed_replay import adopt_completed_search

from combo_core import (
    apply_bias, canonical_sha256, evaluate, normalize, pool, probability_sha256,
    recipe_id, simplex, translate_v2_selection,
)


C0_ALPHAS = (0.05, 0.10, 0.20, 0.35, 0.50)
POOLING = ("arithmetic", "geometric")
BIAS_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_joblib_bytes(value):
    canonical = json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    stream = io.BytesIO(); joblib.dump(canonical, stream, compress=3)
    return stream.getvalue()


def _save_frozen_recipe(run_root, payload, metadata):
    run_root = Path(run_root)
    frozen_path = run_root / "FROZEN_RECIPE.joblib"
    receipt_path = run_root / "FROZEN_RECIPE.json"
    temporary = frozen_path.with_name(f".{frozen_path.name}.partial")
    expected_bytes = _canonical_joblib_bytes(payload)

    def verify_payload(path):
        if Path(path).read_bytes() != expected_bytes:
            raise ValueError("partial frozen recipe changed")

    had_frozen = frozen_path.is_file()
    if had_frozen:
        verify_payload(frozen_path); candidate_path = frozen_path
    else:
        if temporary.is_file():
            verify_payload(temporary)
        else:
            with temporary.open("wb") as stream:
                stream.write(expected_bytes); stream.flush(); os.fsync(stream.fileno())
        candidate_path = temporary
    receipt = {
        "schema_version": "FROZEN_COMBO_RECIPE_RECEIPT_V1",
        **metadata,
        "joblib_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
    if receipt_path.is_file():
        if receipt_path.read_bytes() != _json_bytes(receipt):
            raise ValueError("frozen recipe receipt changed")
    if not had_frozen:
        os.replace(temporary, frozen_path)
    if not receipt_path.is_file():
        receipt_temporary = receipt_path.with_name(f".{receipt_path.name}.partial")
        if receipt_temporary.is_file():
            if receipt_temporary.read_bytes() != _json_bytes(receipt):
                raise ValueError("partial frozen recipe receipt changed")
            os.replace(receipt_temporary, receipt_path)
        else:
            _atomic_json(receipt_path, receipt)
    return receipt


class Ledger:
    def __init__(self, path, identity, *, read_only=False):
        self.read_only = read_only
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.path.with_suffix(".IDENTITY.json")
        if self.identity_path.is_file():
            if json.loads(self.identity_path.read_text()) != identity:
                raise ValueError("grid ledger identity changed")
        else:
            _atomic_json(self.identity_path, identity)
        self.rows = []
        if self.path.is_file():
            with self.path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip(): self.rows.append(json.loads(line))

    def record(self, index, row):
        if index < len(self.rows):
            existing = self.rows[index]
            if existing.get("recipe_id") != row.get("recipe_id") or existing.get("declared_index") != index:
                raise ValueError("grid resume prefix changed")
            return existing, True
        if self.read_only:
            raise ValueError('completed replay cannot append new score rows')
        if index != len(self.rows):
            raise ValueError("grid ledger append is not contiguous")
        value = {**row, "declared_index": index}
        with self.path.open("ab") as stream:
            stream.write(_json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        self.rows.append(value)
        return value, False

    def complete(self, expected, unique_effective):
        if len(self.rows) != expected:
            raise ValueError("grid ledger is incomplete")
        receipt = {
            "schema_version": "COMBO_GRID_COMPLETE_V1", "declared_rows": expected,
            "unique_effective_rows": int(unique_effective),
            "alias_rows": int(expected - unique_effective),
            "ledger_sha256": __import__("hashlib").sha256(self.path.read_bytes()).hexdigest(),
        }
        _atomic_json(self.path.with_suffix(".COMPLETE.json"), receipt)
        return receipt


def _macro(y, probability):
    return float(f1_score(y, probability.argmax(1), labels=np.arange(probability.shape[1]), average="macro", zero_division=0))


def _guard(probability, baseline, guard):
    output = normalize(probability).copy(); output[np.asarray(guard, dtype=bool)] = baseline[np.asarray(guard, dtype=bool)]
    return normalize(output)


def _row(recipe, probability, y, *, family_signature, alias_of=None):
    return {
        "recipe_id": recipe_id(recipe), "recipe": recipe,
        "macro_f1": _macro(y, probability), "probability_sha256": probability_sha256(probability),
        "family_signature": sorted(set(family_signature)), "alias_of": alias_of,
    }


def _verify_frozen_recipe(recipe, nodes, leaves, fold_ids, expected_probability, y):
    expected_probability = normalize(expected_probability)
    reproduced = evaluate(recipe, nodes, leaves, mode="search", fold_ids=fold_ids)
    if reproduced.shape != expected_probability.shape:
        raise ValueError("frozen recipe does not reproduce retained best")
    max_abs = float(np.max(np.abs(reproduced - expected_probability)))
    argmax_identical = bool(np.array_equal(reproduced.argmax(1), expected_probability.argmax(1)))
    reproduced_score = _macro(y, reproduced)
    expected_score = _macro(y, expected_probability)
    if max_abs > 1e-12 or not argmax_identical or reproduced_score != expected_score:
        raise ValueError("frozen recipe does not reproduce retained best")
    return {
        "frozen_reproduction_max_abs": max_abs,
        "frozen_reproduction_argmax_identical": argmax_identical,
        "frozen_reproduction_macro_f1": reproduced_score,
        "frozen_reproduced_probability_sha256": probability_sha256(reproduced),
        "frozen_retained_probability_sha256": probability_sha256(expected_probability),
    }


def _select_parent_rows(rows, incumbent, maximum=6):
    if maximum != 6:
        raise ValueError("parent maximum is frozen at six")
    selected = [incumbent]
    hashes = {incumbent["probability_sha256"]}
    families = set()
    ranked = sorted((row for row in rows if row["recipe_id"] != incumbent["recipe_id"]), key=lambda row: (-row["macro_f1"], row["recipe_id"]))
    # First pass: the best unique probability from each unseen family.
    for row in ranked:
        family = tuple(row["family_signature"])
        if row["probability_sha256"] in hashes or family in families:
            continue
        selected.append(row); hashes.add(row["probability_sha256"]); families.add(family)
        if len(selected) == maximum: return selected
    # Second pass: fill by score while still rejecting exact prediction duplicates.
    for row in ranked:
        if row["probability_sha256"] in hashes:
            continue
        selected.append(row); hashes.add(row["probability_sha256"])
        if len(selected) == maximum: break
    return selected


def _crossfit_biases(v2_root, probability, y, inner_folds, cache_root):
    from combo_log_bias import fit_c10_log_bias, optimizer_receipt
    optimizer = optimizer_receipt()
    identity = {
        "schema_version": "COMBO_BIAS_CACHE_V1",
        "probability_sha256": probability_sha256(probability),
        "y_sha256": canonical_sha256(np.asarray(y).tolist()),
        "inner_folds_sha256": canonical_sha256(np.asarray(inner_folds).tolist()),
        "optimizer_sha256": optimizer["source_sha256"],
        "optimizer_version": optimizer["optimizer_version"],
    }
    key = canonical_sha256(identity)
    path = Path(cache_root) / f"{key}.json"
    if path.is_file():
        value = json.loads(path.read_text())
        if value.get("identity") != identity:
            raise ValueError("bias cache identity mismatch")
        biases = {name: np.asarray(row, dtype=np.float64) for name, row in value["fold_biases"].items()}
        return biases, np.asarray(value["deployment_bias"], dtype=np.float64), value["deployment_receipt"]
    biases = {}
    for fold in sorted(np.unique(inner_folds).tolist()):
        mask = inner_folds != fold
        bias, _ = fit_c10_log_bias(probability[mask], y[mask])
        biases[str(fold)] = bias
    deployment_bias, receipt = fit_c10_log_bias(probability, y)
    _atomic_json(path, {
        "identity": identity,
        "fold_biases": {name: value.tolist() for name, value in biases.items()},
        "deployment_bias": deployment_bias.tolist(),
        "deployment_receipt": {"compatible_receipt": receipt, "optimizer_receipt": optimizer},
    })
    return biases, deployment_bias, {"compatible_receipt": receipt, "optimizer_receipt": optimizer}


def _biased_probability(base, fold_biases, strength, inner_folds):
    if strength == 0:
        return base
    output = np.zeros_like(base)
    for fold in sorted(np.unique(inner_folds).tolist()):
        mask = inner_folds == fold
        output[mask] = apply_bias(base[mask], fold_biases[str(fold)] * strength)
    return output


def _evidence(path):
    with np.load(path, allow_pickle=False) as stored:
        required = (
            "ids", "y", "groups", "outer_folds", "inner_folds", "class_order", "guard",
            "candidate_ids", "candidate_families", "candidate_probabilities",
            "anchor_ids", "anchor_probabilities", "stage_probability",
        )
        if any(name not in stored for name in required):
            raise ValueError("prepared evidence fields are incomplete")
        return {name: np.asarray(stored[name]) for name in required}


def _c0_probability(recipe, leaves, baseline, guard):
    if recipe["kind"] == "incumbent":
        return baseline
    return _guard(pool(
        [leaves[recipe["reference_leaf"]], leaves[recipe["candidate_leaf"]]],
        [1 - recipe["alpha"], recipe["alpha"]], recipe["pooling"],
    ), baseline, guard)


def run_search(config, run_root):
    run_root = Path(run_root); run_root.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(Path(config["evidence_npz"]).read_bytes()).hexdigest() != config["evidence_sha256"]:
        raise ValueError("evidence NPZ hash mismatch")
    if hashlib.sha256(Path(config["grid_manifest_json"]).read_bytes()).hexdigest() != config["grid_manifest_sha256"]:
        raise ValueError("grid manifest hash mismatch")
    evidence_receipt = read_json(config["evidence_receipt_json"])
    if evidence_receipt.get("evidence_sha256") != config["evidence_sha256"]:
        raise ValueError("evidence receipt hash mismatch")
    complete_path = run_root / "SEARCH_COMPLETE.json"
    if complete_path.is_file():
        complete = read_json(complete_path)
        if (complete.get("evidence_sha256") != config["evidence_sha256"]
                or complete.get("meta_source_sha256") != config["meta_source_sha256"]
                or complete.get("runtime_config_sha256") != config["_runtime_config_sha256"]):
            raise ValueError("completed search provenance changed")
        frozen_path = run_root / "FROZEN_RECIPE.joblib"; frozen_receipt_path = run_root / "FROZEN_RECIPE.json"
        if (hashlib.sha256(frozen_path.read_bytes()).hexdigest() != complete.get("frozen_recipe_joblib_sha256")
                or hashlib.sha256(frozen_receipt_path.read_bytes()).hexdigest() != complete.get("frozen_recipe_receipt_sha256")):
            raise ValueError("completed frozen recipe changed")
        for name, receipt in zip(("C0", "C1", "C2"), complete.get("round_receipts", [])):
            ledger_path = run_root / f"ledgers/{name}.jsonl"
            if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != receipt.get("ledger_sha256"):
                raise ValueError("completed score ledger changed")
        return complete
    evidence = _evidence(config["evidence_npz"])
    ids, y = evidence["ids"], evidence["y"].astype(np.int64)
    inner_folds = evidence["inner_folds"].astype(np.int16)
    guard = evidence["guard"].astype(bool)
    for name, value in (("ids", ids), ("y", y), ("groups", evidence["groups"]),
                        ("outer_folds", evidence["outer_folds"]), ("inner_folds", inner_folds),
                        ("class_order", evidence["class_order"])):
        if evidence_receipt.get(f"{name}_sha256") != canonical_sha256(np.asarray(value).tolist()):
            raise ValueError(f"evidence {name} identity hash mismatch")
    if evidence["guard"].dtype != np.bool_:
        raise ValueError("evidence guard dtype is not bool")
    baseline = normalize(evidence["stage_probability"])
    if baseline.shape != (len(ids), len(evidence["class_order"])):
        raise ValueError("stage baseline shape mismatch")
    leaves = {"stage:latest": baseline, "mask:guard": guard}
    candidate_family = {}
    for index, candidate_id in enumerate(evidence["candidate_ids"].tolist()):
        leaves[f"candidate:{candidate_id}"] = normalize(evidence["candidate_probabilities"][index])
        candidate_family[str(candidate_id)] = str(evidence["candidate_families"][index])
    for index, anchor_id in enumerate(evidence["anchor_ids"].tolist()):
        leaves[f"anchor:{anchor_id}"] = normalize(evidence["anchor_probabilities"][index])
    if len(candidate_family) != 124:
        raise ValueError("combo search requires all 124 complete candidates")
    if hashlib.sha256(Path(config["stage_selection_joblib"]).read_bytes()).hexdigest() != evidence_receipt.get("stage_selection_sha256"):
        raise ValueError("stage selection differs from evidence receipt")
    stage_selection = joblib.load(config["stage_selection_joblib"])
    translated = translate_v2_selection(stage_selection)
    graph_nodes = dict(translated["nodes"])
    stage_root = translated["root"]
    def leaf_node(leaf_id):
        node_id = "LEAF-" + canonical_sha256(leaf_id)
        graph_nodes.setdefault(node_id, {"kind": "leaf", "leaf_id": leaf_id})
        return node_id

    identity = {
        "schema_version": "COMBO_SEARCH_IDENTITY_V1",
        "evidence_sha256": config["evidence_sha256"], "rows": len(ids),
        "candidate_count": len(candidate_family), "grid_manifest_sha256": config["grid_manifest_sha256"],
        "meta_source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
    }
    identity, replay_receipt = adopt_completed_search(config, run_root, identity)
    c0 = Ledger(run_root / "ledgers/C0.jsonl", {**identity, "round": "C0"}, read_only=bool(replay_receipt))
    declarations = [{"kind": "incumbent", "reference_leaf": "stage:latest"}]
    for candidate_id in sorted(candidate_family):
        for reference in ("anchor:A1_GEOMETRIC_GUARD", "anchor:A2_C10_BIAS_GEOMETRIC_GUARD", "stage:latest"):
            for pooling_name in POOLING:
                for alpha in C0_ALPHAS:
                    declarations.append({
                        "kind": "c0_mix", "candidate_leaf": f"candidate:{candidate_id}",
                        "candidate_id": candidate_id, "reference_leaf": reference,
                        "pooling": pooling_name, "alpha": alpha,
                    })
    if len(declarations) != 3721:
        raise AssertionError("C0 declaration count changed")
    seen_probability = {row["probability_sha256"]: row["recipe_id"] for row in c0.rows}
    for index, recipe in enumerate(declarations):
        rid = recipe_id(recipe)
        if index < len(c0.rows):
            c0.record(index, {"recipe_id": rid, "recipe": recipe}); continue
        probability = _c0_probability(recipe, leaves, baseline, guard)
        probability_hash = probability_sha256(probability)
        alias = seen_probability.get(probability_hash)
        row = _row(
            recipe, probability, y,
            family_signature=[] if recipe["kind"] == "incumbent" else [candidate_family[recipe["candidate_id"]]],
            alias_of=alias,
        )
        c0.record(index, row)
        seen_probability.setdefault(probability_hash, row["recipe_id"])
    c0_receipt = c0.complete(3721, len({row["probability_sha256"] for row in c0.rows}))
    incumbent_row = c0.rows[0]
    selected_rows = _select_parent_rows(c0.rows, incumbent_row)
    row_index = {row['recipe_id']: row for row in c0.rows}

    def materialize(row, source_parents):
        executed = execution_row(row, row_index)
        recipe = executed["recipe"]
        if recipe["kind"] == "incumbent":
            probability = _c0_probability(recipe, leaves, baseline, guard)
            node_id = stage_root
        elif recipe["kind"] == "c0_mix":
            probability = _c0_probability(recipe, leaves, baseline, guard)
            reference_node = stage_root if recipe["reference_leaf"] == "stage:latest" else leaf_node(recipe["reference_leaf"])
            node_id = row["recipe_id"]
            graph_nodes[node_id] = {
                "kind": "mix", "sources": [reference_node, leaf_node(recipe["candidate_leaf"])],
                "weights": [1 - recipe["alpha"], recipe["alpha"]], "pooling": recipe["pooling"],
            }
        else:
            sources = [source_parents[value]["probability"] for value in recipe["source_recipe_ids"]]
            pooled = _guard(pool(sources, recipe["weights"], recipe["pooling"]), baseline, guard)
            fold_biases, deployment_bias, bias_receipt = _crossfit_biases(
                config["v2_root"], pooled, y, inner_folds, run_root / "bias_cache",
            )
            probability = _guard(_biased_probability(
                pooled, fold_biases, recipe["bias_strength"], inner_folds,
            ), baseline, guard)
            mix_id = row["recipe_id"] + "-MIX"
            active = [(source_parents[value]["node_id"], float(weight))
                      for value, weight in zip(recipe["source_recipe_ids"], recipe["weights"]) if float(weight) > 0]
            graph_nodes[mix_id] = {
                "kind": "mix", "sources": [value for value, _ in active],
                "weights": [value for _, value in active], "pooling": recipe["pooling"],
            }
            node_id = row["recipe_id"]
            graph_nodes[node_id] = {
                "kind": "crossfit_bias", "source": mix_id, "strength": recipe["bias_strength"],
                "fold_biases": {key: value.tolist() for key, value in fold_biases.items()},
                "deployment_bias": deployment_bias.tolist(), "bias_receipt": bias_receipt,
            }
            if executed['recipe_id'] != row['recipe_id']:
                graph_nodes[node_id]['canonical_execution_recipe_id'] = executed['recipe_id']
        if probability_sha256(probability) != row["probability_sha256"]:
            raise ValueError("selected parent probability does not reproduce")
        return {"row": row, "probability": probability, "node_id": node_id}

    parents = {}; archive = {}
    for row in selected_rows:
        parent = materialize(row, parents)
        parents[row["recipe_id"]] = parent; archive[row["recipe_id"]] = parent
    _save_parents(run_root, "C0", selected_rows, parents, graph_nodes, ids, evidence["class_order"])

    round_receipts = [c0_receipt]
    all_rows = list(c0.rows)
    global_best = sorted(all_rows, key=lambda row: (-row["macro_f1"], row["recipe_id"]))[0]
    if global_best["recipe_id"] not in archive:
        archive[global_best["recipe_id"]] = materialize(global_best, parents)
    for round_index in (1, 2):
        parent_ids = tuple(row["recipe_id"] for row in selected_rows)
        ledger = Ledger(run_root / f"ledgers/C{round_index}.jsonl", {
            **identity, "round": f"C{round_index}", "parent_ids": list(parent_ids),
        }, read_only=bool(replay_receipt))
        declared_index = 0
        effective_cache = {row.get("effective_key"): row for row in ledger.rows if row.get("effective_key")}
        probability_alias = {row["probability_sha256"]: row["recipe_id"] for row in ledger.rows}
        for weights in simplex(slots=len(parent_ids)):
            active = tuple((rid, weight) for rid, weight in zip(parent_ids, weights) if weight > 0)
            for pooling_name in POOLING:
                effective_pooling = pooling_name if len(active) > 1 else "single_source"
                effective_key = canonical_sha256({"active": active, "pooling": effective_pooling})
                pooled = None; fold_biases = deployment_bias = bias_receipt = None
                for strength in BIAS_STRENGTHS:
                    recipe = {
                        "kind": "meta_mix", "round": round_index,
                        "source_recipe_ids": list(parent_ids), "weights": list(weights),
                        "pooling": pooling_name, "bias_strength": strength,
                    }
                    rid = recipe_id(recipe)
                    if declared_index < len(ledger.rows):
                        ledger.record(declared_index, {"recipe_id": rid, "recipe": recipe}); declared_index += 1; continue
                    alias_effective_key = canonical_sha256({"pool": effective_key, "bias_strength": strength})
                    if alias_effective_key in effective_cache:
                        prior = effective_cache[alias_effective_key]
                        row = {**prior, "recipe_id": rid, "recipe": recipe, "alias_of": prior["recipe_id"]}
                    else:
                        if pooled is None:
                            pooled = _guard(pool(
                                [parents[rid_value]["probability"] for rid_value in parent_ids], weights, pooling_name,
                            ), baseline, guard)
                            fold_biases, deployment_bias, bias_receipt = _crossfit_biases(
                                config["v2_root"], pooled, y, inner_folds, run_root / "bias_cache",
                            )
                        probability = _guard(_biased_probability(pooled, fold_biases, strength, inner_folds), baseline, guard)
                        psha = probability_sha256(probability)
                        alias = probability_alias.get(psha)
                        active_ids = {rid_value for rid_value, weight in zip(parent_ids, weights) if weight > 0}
                        families = sorted({family for row0 in selected_rows if row0["recipe_id"] in active_ids
                                           for family in row0["family_signature"]})
                        row = _row(recipe, probability, y, family_signature=families, alias_of=alias)
                        row["effective_key"] = alias_effective_key
                        probability_alias.setdefault(psha, row["recipe_id"])
                        effective_cache[alias_effective_key] = row
                    ledger.record(declared_index, row); declared_index += 1
        expected = math.comb(len(parent_ids) + 3, 4) * 2 * 5
        receipt = ledger.complete(expected, len({row["probability_sha256"] for row in ledger.rows}))
        round_receipts.append(receipt); all_rows.extend(ledger.rows)
        row_index.update({row['recipe_id']: row for row in ledger.rows})
        round_best = sorted(ledger.rows, key=lambda row: (-row["macro_f1"], row["recipe_id"]))[0]
        if (-round_best["macro_f1"], round_best["recipe_id"]) < (-global_best["macro_f1"], global_best["recipe_id"]):
            global_best = round_best
        if global_best["recipe_id"] not in archive:
            archive[global_best["recipe_id"]] = materialize(global_best, parents)
        selected_rows = _select_parent_rows(ledger.rows, global_best)
        new_parents = {}
        for row in selected_rows:
            if row["recipe_id"] in archive:
                parent = archive[row["recipe_id"]]
            else:
                parent = materialize(row, parents); archive[row["recipe_id"]] = parent
            new_parents[row["recipe_id"]] = parent
        parents = new_parents
        _save_parents(run_root, f"C{round_index}", selected_rows, parents, graph_nodes, ids, evidence["class_order"])

    best = sorted(all_rows, key=lambda row: (-row["macro_f1"], row["recipe_id"]))[0]
    if best["recipe_id"] not in archive:
        raise ValueError("global best recipe was not retained")
    frozen_graph = _prune_graph(archive[best["recipe_id"]]["node_id"], graph_nodes)
    frozen_graph.update(_prune_graph(stage_root, graph_nodes))
    frozen_recipe = {
        "schema_version": "FROZEN_COMBO_RECIPE_V1", "root": archive[best["recipe_id"]]["node_id"],
        "guard_baseline_leaf": "stage:latest", "guard_mask_leaf": "mask:guard",
        "stage_graph_root": stage_root,
    }
    frozen_reproduction = _verify_frozen_recipe(
        frozen_recipe, frozen_graph, leaves, inner_folds,
        archive[best["recipe_id"]]["probability"], y,
    )
    frozen_receipt = _save_frozen_recipe(
        run_root,
        {"recipe": frozen_recipe, "nodes": frozen_graph},
        {
            "best_recipe_id": best["recipe_id"],
            "best_inner_macro_f1": best["macro_f1"],
            "active_node_count": len(frozen_graph),
            **frozen_reproduction,
        },
    )
    _atomic_json(run_root / "SEARCH_COMPLETE.json", {
        "schema_version": "COMBO_SEARCH_COMPLETE_V1", "status": "COMPLETE",
        "best_recipe_id": best["recipe_id"], "best_inner_macro_f1": best["macro_f1"],
        "incumbent_inner_macro_f1": _macro(y, baseline),
        "declared_score_rows": sum(value["declared_rows"] for value in round_receipts),
        "round_receipts": round_receipts,
        "evidence_sha256": config["evidence_sha256"],
        "stage_deployment_sha256": evidence_receipt["stage_deployment_sha256"],
        "meta_source_sha256": config["meta_source_sha256"],
        "runtime_config_sha256": config["_runtime_config_sha256"],
        "frozen_recipe_joblib_sha256": frozen_receipt["joblib_sha256"],
        "frozen_recipe_receipt_sha256": hashlib.sha256((run_root / "FROZEN_RECIPE.json").read_bytes()).hexdigest(),
        "score_producer_source_sha256": identity['meta_source_sha256'],
        "score_producer_config_sha256": identity['runtime_config_sha256'],
        "completed_score_replay": bool(replay_receipt),
        "origin_receipt_sha256": hashlib.sha256((run_root / 'ORIGIN.json').read_bytes()).hexdigest() if replay_receipt else None,
        "minimum_f1_gate": None, "test_read": False, "full_oof_used": False,
    })
    return read_json(run_root / "SEARCH_COMPLETE.json")


def _save_parents(run_root, round_name, rows, probabilities, nodes, ids, class_order):
    root = Path(run_root) / "parents" / round_name; root.mkdir(parents=True, exist_ok=True)
    ordered = [row["recipe_id"] for row in rows]
    expected = {
        "ids": np.asarray(ids, dtype="U"), "class_order": np.asarray(class_order, dtype="U"),
        "recipe_ids": np.asarray(ordered, dtype="U"),
        "probabilities": np.stack([probabilities[value]["probability"] for value in ordered]),
    }
    probability_path = root / "probabilities.npz"
    recipe_path = root / "recipes.joblib"
    expected_recipe_bytes = _canonical_joblib_bytes({"rows": rows, "nodes": nodes})
    if probability_path.is_file():
        with np.load(probability_path, allow_pickle=False) as stored:
            if set(stored.files) != set(expected) or any(not np.array_equal(stored[name], value) for name, value in expected.items()):
                raise ValueError("parent probability checkpoint changed")
    if recipe_path.is_file():
        if recipe_path.read_bytes() != expected_recipe_bytes:
            raise ValueError("parent recipe checkpoint changed")
    if probability_path.is_file() and recipe_path.is_file():
        return
    probability_temp = probability_path.with_name(f".{probability_path.name}.partial")
    recipe_temp = recipe_path.with_name(f".{recipe_path.name}.partial")
    if not probability_path.is_file():
        if probability_temp.is_file():
            with np.load(probability_temp, allow_pickle=False) as stored:
                if set(stored.files) != set(expected) or any(not np.array_equal(stored[name], value) for name, value in expected.items()):
                    raise ValueError("partial parent probability checkpoint changed")
        else:
            with probability_temp.open("wb") as stream: np.savez_compressed(stream, **expected)
        os.replace(probability_temp, probability_path)
    if not recipe_path.is_file():
        if recipe_temp.is_file():
            if recipe_temp.read_bytes() != expected_recipe_bytes:
                raise ValueError("partial parent recipe checkpoint changed")
        else:
            with recipe_temp.open("wb") as stream:
                stream.write(expected_recipe_bytes); stream.flush(); os.fsync(stream.fileno())
        os.replace(recipe_temp, recipe_path)


def _prune_graph(root, nodes):
    keep = {}
    def visit(node_id):
        if node_id in keep:
            return
        node = nodes[node_id]; keep[node_id] = node
        if node["kind"] in {"fixed_bias", "crossfit_bias"}:
            visit(node["source"])
        elif node["kind"] == "mix":
            for source, weight in zip(node["sources"], node["weights"]):
                if float(weight) > 0: visit(source)
    visit(root)
    return keep


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
