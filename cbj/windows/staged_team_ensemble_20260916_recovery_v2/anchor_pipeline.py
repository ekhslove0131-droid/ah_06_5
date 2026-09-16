"""Two alternative anchor endpoints and frozen historical replay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
import warnings

from mrmr import mrmr_classif

from contracts import CLASS_ORDER
from feature_transforms import (
    FeaturePair,
    _base_state,
    _hstack,
    _matrix_hash,
    _mcmp_block_pair,
    _mcmp_rows,
    _numeric41,
    _tfidf_pair,
    _validate_context,
)
from log_bias import apply_log_bias, fit_c10_log_bias
from support_guard import geometric_pool, guarded_probability, support_guard_mask


H1_BLOCKS = ["gene", "type", "gene_change", "aa", "aa_from", "aa_to", "global", "site", "exact"]
C10_BLOCKS = ["gene", "type", "aa", "aa_from", "aa_to", "gene_change", "site", "exact"]


def anchor_model_recipes():
    """Return immutable effective recipes for the two anchor components."""
    return {
        "H1": {
            "family": "ANCHOR_H1",
            "candidate_id": "H1_MRMR_GLOBAL_RAW12",
            "config_hash": hashlib.sha256(b"H1_MRMR_GLOBAL_RAW12_V1").hexdigest(),
            "kind": "xgboost",
            "device_kind": "GPU_REQUIRED",
            "weighting": "sqrt_inverse",
            "params": {
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
                "n_jobs": 4,
                "tree_method": "hist",
                "device": "cuda:0",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
                "min_child_weight": 1.0,
            },
        },
        "C10": {
            "family": "ANCHOR_C10",
            "candidate_id": "C10_TFIDF_LOGISTIC",
            "config_hash": hashlib.sha256(b"C10_TFIDF_LOGISTIC_V1").hexdigest(),
            "kind": "logistic",
            "device_kind": "CPU_INTENTIONAL",
            "weighting": "sqrt_inverse",
            "params": {
                "C": 10.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 2000,
                "tol": 1e-5,
                "fit_intercept": True,
            },
        },
    }


def _anchor_state(name, context, fitted, **extra):
    declaration = {
        "candidate_id": name,
        "config_hash": hashlib.sha256(name.encode("utf-8")).hexdigest(),
        "family": "ANCHOR",
        "parameters": {},
    }
    return _base_state(declaration, context, fitted) | extra


def _h1_selected_mask(matrix, names, y):
    candidate = np.asarray([
        index for index, name in enumerate(names)
        if name.startswith(("type|", "gene_change|"))
    ], dtype=np.int64)
    candidate_set = set(candidate.tolist())
    protected = np.asarray([
        index for index in range(len(names)) if index not in candidate_set
    ], dtype=np.int64)
    working = matrix[:, candidate].copy().tocsr()
    working.sum_duplicates()
    working.eliminate_zeros()
    if len(working.data) and not np.all(working.data == 1):
        raise ValueError("H1 candidate columns must be binary")
    count = np.asarray(working.sum(axis=0)).ravel()
    active = (count > 0) & (count < len(y))
    candidate = candidate[active]
    working = working[:, active]
    if not len(candidate):
        raise ValueError("H1 has no variable derived candidates")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        relevance = f_classif(working, y)[0]
    relevance = np.nan_to_num(
        relevance, nan=0.0, posinf=np.finfo(float).max, neginf=0.0,
    )
    order = np.lexsort((candidate, -relevance))
    pool = candidate[order[:min(2500, len(candidate))]]
    selected_k = min(1200, len(pool))
    selected_names = mrmr_classif(
        X=pd.DataFrame(matrix[:, pool].toarray(), columns=[str(index) for index in pool]),
        y=pd.Series(y),
        K=selected_k,
        n_jobs=1,
        relevance="f",
        redundancy="c",
        denominator="mean",
        show_progress=False,
    )
    chosen = [int(value) for value in selected_names]
    # mrmr-selection can stop early on tiny synthetic or fully redundant data.
    # Preserve its returned rank, then fill only from the fixed F-ranked pool.
    chosen_set = set(chosen)
    chosen.extend(int(index) for index in pool if int(index) not in chosen_set and len(chosen) < selected_k)
    if len(chosen) != selected_k or len(set(chosen)) != selected_k or not set(chosen).issubset(set(pool.tolist())):
        raise ValueError("invalid H1 mRMR selection")
    selected = np.sort(np.r_[protected, np.asarray(chosen, dtype=np.int64)])
    fitted = {
        "candidate_prefixes": ["type|", "gene_change|"],
        "prefilter_max": 2500,
        "selected_k_limit": 1200,
        "candidate_count": int(len(candidate)),
        "prefilter_pool": int(len(pool)),
        "selected_k": int(selected_k),
        "protected_count": int(len(protected)),
        "selected_indices": selected.tolist(),
        "selected_names": [names[index] for index in selected],
        "mrmr_ranked_indices": chosen,
    }
    return selected, fitted


def prepare_anchor_features(train, valid, context):
    """Fit H1/C10 transformations on one fitting partition only."""
    _validate_context(train, valid, context)
    train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)

    h1_raw_train, h1_raw_valid, h1_encoder = _mcmp_block_pair(
        train_rows, valid_rows, context.train_groups, H1_BLOCKS, 3,
    )
    selected, selection = _h1_selected_mask(
        h1_raw_train, h1_encoder["names"], context.train_y,
    )
    h1_train = _hstack(h1_raw_train[:, selected], np.asarray(context.numeric41_train, dtype=np.float32))
    h1_valid = _hstack(h1_raw_valid[:, selected], np.asarray(context.numeric41_valid, dtype=np.float32))
    h1_names = selection["selected_names"] + [f"numeric41::{index}" for index in range(41)]
    h1_fitted = {
        "encoder": h1_encoder,
        "selection": selection,
        "numeric41": {"transform": "raw", "width": 41},
    }
    h1_state = _anchor_state(
        "H1_MRMR_GLOBAL_RAW12", context, h1_fitted,
        selected_k=selection["selected_k"],
        prefilter_pool=selection["prefilter_pool"],
    )
    h1_pair = FeaturePair(
        train=h1_train,
        valid=h1_valid,
        feature_names=h1_names,
        state=h1_state,
        sample_weight_multiplier=np.ones(len(train), dtype=np.float32),
        train_sha256=_matrix_hash(h1_train),
        valid_sha256=_matrix_hash(h1_valid),
    )

    c10_pair, c10_symbolic_train, c10_symbolic_valid = prepare_c10_features(
        train, valid, context, rows=(train_rows, valid_rows),
    )
    guard_mask, guard_receipt = support_guard_mask(c10_symbolic_train, c10_symbolic_valid)
    return {
        "H1": h1_pair,
        "C10": c10_pair,
        "guard_mask": guard_mask,
        "guard_receipt": guard_receipt,
    }


def prepare_c10_features(train, valid, context, *, rows=None):
    """Prepare only C10, avoiding H1/mRMR work on A2 sub-inner folds."""
    _validate_context(train, valid, context)
    train_rows, valid_rows = rows if rows is not None else _mcmp_rows(train, valid, context.gene_cols)
    symbolic_train, symbolic_valid, encoder = _mcmp_block_pair(
        train_rows, valid_rows, context.train_groups, C10_BLOCKS, 3,
    )
    tfidf_train, tfidf_valid, tfidf_state = _tfidf_pair(symbolic_train, symbolic_valid)
    numeric_train, numeric_valid, numeric_state = _numeric41(context)
    combined_train = _hstack(tfidf_train, numeric_train)
    combined_valid = _hstack(tfidf_valid, numeric_valid)
    names = encoder["names"] + [f"numeric41::{index}" for index in range(41)]
    fitted = {
        "encoder": encoder,
        "tfidf": tfidf_state,
        "numeric41": numeric_state,
        "model": {"C": 10.0, "solver": "lbfgs", "penalty": "l2", "max_iter": 2000, "tol": 1e-5},
    }
    state = _anchor_state("C10_TFIDF_LOGISTIC", context, fitted, C=10.0)
    pair = FeaturePair(
        train=combined_train, valid=combined_valid, feature_names=names, state=state,
        sample_weight_multiplier=np.ones(len(train), dtype=np.float32),
        train_sha256=_matrix_hash(combined_train), valid_sha256=_matrix_hash(combined_valid),
    )
    return pair, symbolic_train, symbolic_valid


def fit_anchor_components(train, valid, context, *, capabilities=None):
    """Fit H1 on CUDA and C10 on intentional CPU for one boundary."""
    from model_adapters import fit_model, predict_checked
    features = prepare_anchor_features(train, valid, context)
    recipes = anchor_model_recipes()
    artifacts = {}
    probabilities = {}
    for name in ("H1", "C10"):
        artifacts[name] = fit_model(
            recipes[name], features[name], context.train_y, None, context.class_order,
            capabilities=capabilities,
        )
        probabilities[name] = predict_checked(artifacts[name], features[name].valid, context.class_order)
    return {
        "features": features,
        "artifacts": artifacts,
        "probabilities": probabilities,
        "guard_mask": features["guard_mask"],
        "guard_receipt": features["guard_receipt"],
    }


def build_anchor_predictions(h1_probability, c10_probability, guard_mask, bias_fit_probability, bias_fit_y):
    bias, receipt = fit_c10_log_bias(bias_fit_probability, bias_fit_y)
    a1 = guarded_probability(geometric_pool(h1_probability, c10_probability), h1_probability, guard_mask)
    corrected_c10 = apply_log_bias(c10_probability, bias)
    a2 = guarded_probability(geometric_pool(h1_probability, corrected_c10), h1_probability, guard_mask)
    return {
        "A1_GEOMETRIC_GUARD": a1,
        "A2_C10_BIAS_GEOMETRIC_GUARD": a2,
        "bias_receipt": receipt,
    }


def build_anchor_oof(
    h1_probability,
    c10_probability,
    guard_mask,
    subinner_c10_oof,
    subinner_y,
    *,
    branch,
):
    """Build one held-out anchor from bias evidence outside that held-out part.

    The interface intentionally has no parameter for labels belonging to the
    rows being predicted.  ``subinner_*`` must be exact-once OOF evidence from
    the fitting parent and is used only by A2.
    """
    allowed = {"A1_GEOMETRIC_GUARD", "A2_C10_BIAS_GEOMETRIC_GUARD"}
    if branch not in allowed:
        raise ValueError("unknown anchor branch")
    h1_probability = np.asarray(h1_probability, dtype=np.float64)
    c10_probability = np.asarray(c10_probability, dtype=np.float64)
    if branch == "A1_GEOMETRIC_GUARD":
        probability = guarded_probability(
            geometric_pool(h1_probability, c10_probability), h1_probability, guard_mask,
        )
        bias_receipt = None
    else:
        bias, bias_receipt = fit_c10_log_bias(subinner_c10_oof, subinner_y)
        probability = guarded_probability(
            geometric_pool(h1_probability, apply_log_bias(c10_probability, bias)),
            h1_probability,
            guard_mask,
        )
    stable = np.ascontiguousarray(probability, dtype="<f8")
    return {
        "branch": branch,
        "probability": probability,
        "receipt": {
            "schema_version": "ANCHOR_PREDICTION_RECEIPT_V1",
            "branch": branch,
            "rows": int(len(probability)),
            "classes": int(probability.shape[1]),
            "probability_sha256": hashlib.sha256(stable.tobytes()).hexdigest(),
            "guarded_rows": int(np.asarray(guard_mask, dtype=bool).sum()),
            "bias": bias_receipt,
            "outer_or_same_validation_labels_accepted": False,
        },
    }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_historical_anchors(root):
    root = Path(root)
    files = {
        "A1_GEOMETRIC_GUARD": (
            "geometric_support_guard_best_oof.npz",
            "3b668367ca6d48f9c05f2a507458b6e0d530322722c99422980aba353722084a",
            "1f8f42adfac600311cf7f661e99c30c9cd2029eee2affcc46759c8b7edaacf96",
        ),
        "A2_C10_BIAS_GEOMETRIC_GUARD": (
            "bias_corrected_geometric_guard_oof.npz",
            "f4ede5b52937a5c8fc5902413da2e6a15b0add0fb22dc20a6f80a639cd90e517",
            "26c51cae601da997e52329d68637aa626fc1fadd9e0ced79fc6deff9d77c528f",
        ),
    }
    output = {}
    identity = None
    reference_ids = reference_folds = None
    for name, (filename, expected_sha, expected_argmax_sha) in files.items():
        path = root / filename
        if _sha(path) != expected_sha:
            raise ValueError("historical anchor OOF hash mismatch")
        with np.load(path, allow_pickle=False) as stored:
            current_identity = tuple(stored[key].tolist() for key in ("ids", "y", "groups", "folds", "class_order"))
            if stored["class_order"].tolist() != CLASS_ORDER:
                raise ValueError("historical anchor class order mismatch")
            probability = np.asarray(stored["prob"], dtype=np.float64)
            y = np.asarray(stored["y"], dtype=np.int64)
            current_ids = np.asarray(stored["ids"])
            current_folds = np.asarray(stored["folds"])
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError("historical anchor identities differ")
        score = float(f1_score(y, probability.argmax(axis=1), labels=np.arange(26), average="macro", zero_division=0))
        argmax = np.ascontiguousarray(probability.argmax(axis=1), dtype="<i8")
        argmax_sha = hashlib.sha256(argmax.tobytes()).hexdigest()
        if argmax_sha != expected_argmax_sha:
            raise ValueError("historical anchor argmax mismatch")
        reference_ids, reference_folds = current_ids, current_folds
        output[name] = {
            "macro_f1": score,
            "sha256": expected_sha,
            "argmax_sha256": argmax_sha,
            "rows": len(y),
            "classes": 26,
        }
    guard_path = root.parent / "goal055_support_guard_seed43_v2" / "guard_mask.npz"
    expected_guard_sha = "4db5857f00e06ea227849c60637e4731547d24a4e35e369845320f76918ac94f"
    if _sha(guard_path) != expected_guard_sha:
        raise ValueError("historical support guard hash mismatch")
    with np.load(guard_path, allow_pickle=False) as stored:
        if not np.array_equal(stored["ids"], reference_ids) or not np.array_equal(stored["folds"], reference_folds):
            raise ValueError("historical support guard identity mismatch")
        guard = np.asarray(stored["guard"])
    if guard.dtype != np.bool_ or int(guard.sum()) != 94:
        raise ValueError("historical support guard row count mismatch")
    output["guarded_rows"] = 94
    output["guard_mask_sha256"] = expected_guard_sha
    return output
