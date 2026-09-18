"""Fold-fitted T1-T6 feature transformations with explicit lineage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler

from mrmr import mrmr_classif

from raw_views import exact_cell_token_rows, literal_gene_binary, mutation_severity, recurrent_tokens
from vendor import mcmp_repr


class ResourceBlockedError(RuntimeError):
    pass


@dataclass
class FeatureContext:
    gene_cols: list[str]
    train_ids: np.ndarray
    valid_ids: np.ndarray
    train_groups: np.ndarray
    train_y: np.ndarray
    class_order: list[str]
    numeric41_train: np.ndarray
    numeric41_valid: np.ndarray
    max_dense_bytes: int = 12 * 1024**3


@dataclass
class FeaturePair:
    train: object
    valid: object
    feature_names: list[str]
    state: dict
    sample_weight_multiplier: np.ndarray
    train_sha256: str
    valid_sha256: str
    deferred_numeric_train: np.ndarray | None = None
    deferred_numeric_valid: np.ndarray | None = None
    class_reweights: np.ndarray | None = None
    symbolic_width: int = 0


def _sequence_hash(values) -> str:
    payload = json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _matrix_hash(matrix) -> str:
    digest = hashlib.sha256()
    if sparse.issparse(matrix):
        value = matrix.tocsr().astype(np.float32)
        digest.update(b"CSR_FLOAT32_V1")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.indptr.astype("<i8", copy=False).tobytes())
        digest.update(value.indices.astype("<i8", copy=False).tobytes())
        digest.update(value.data.astype("<f4", copy=False).tobytes())
    else:
        value = np.ascontiguousarray(np.asarray(matrix, dtype="<f4"))
        digest.update(b"DENSE_FLOAT32_V1")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_context(train, valid, context: FeatureContext):
    if list(train.columns) != context.gene_cols or list(valid.columns) != context.gene_cols:
        raise ValueError("gene columns are missing or reordered")
    if len(train) != len(context.train_ids) or len(valid) != len(context.valid_ids):
        raise ValueError("feature rows and IDs are misaligned")
    if len(context.train_y) != len(train) or len(context.train_groups) != len(train):
        raise ValueError("training labels/groups are misaligned")
    if np.asarray(context.numeric41_train).shape != (len(train), 41) or np.asarray(context.numeric41_valid).shape != (len(valid), 41):
        raise ValueError("numeric41 shape mismatch")
    if set(np.asarray(context.train_y).tolist()) != set(context.class_order):
        raise ValueError("training classes differ from class order")


def _numeric41(context: FeatureContext):
    train = np.asarray(context.numeric41_train, dtype=np.float32)
    valid = np.asarray(context.numeric41_valid, dtype=np.float32)
    if not np.isfinite(train).all() or not np.isfinite(valid).all() or (train < 0).any() or (valid < 0).any():
        raise ValueError("numeric41 must be finite and nonnegative")
    scaler = StandardScaler()
    train_out = scaler.fit_transform(np.log1p(train)).astype(np.float32) / math.sqrt(41)
    valid_out = scaler.transform(np.log1p(valid)).astype(np.float32) / math.sqrt(41)
    state = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weight": 1 / math.sqrt(41),
    }
    return train_out, valid_out, state


def _hstack(left, right):
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.hstack([sparse.csr_matrix(left), sparse.csr_matrix(right)], format="csr", dtype=np.float32)
    return np.column_stack([left, right]).astype(np.float32)


def _mcmp_rows(train, valid, genes):
    train_rows, _ = mcmp_repr.build_feature_rows(train, genes)
    valid_rows, _ = mcmp_repr.build_feature_rows(valid, genes)
    return train_rows, valid_rows


def _mcmp_block_pair(train_rows, valid_rows, groups, blocks, min_support):
    encoder = mcmp_repr.Encoder(blocks, min_support)
    train_matrix = encoder.fit_transform(train_rows, groups).astype(np.float32).tocsr()
    valid_matrix = encoder.transform(valid_rows).astype(np.float32).tocsr()
    return train_matrix, valid_matrix, encoder.to_dict()


def _tfidf_pair(train_matrix, valid_matrix):
    transformer = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    train_out = transformer.fit_transform(train_matrix).astype(np.float32).tocsr()
    valid_out = transformer.transform(valid_matrix).astype(np.float32).tocsr()
    return train_out, valid_out, {
        "idf": transformer.idf_.tolist(),
        "norm": "l2", "use_idf": True, "smooth_idf": True, "sublinear_tf": True,
    }


def _select_genes(train, context, selector):
    binary = literal_gene_binary(train, context.gene_cols)
    limit = min(500, len(context.gene_cols))
    if selector == "chi2_top500":
        scores, _ = chi2(binary, context.train_y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        order = np.argsort(-scores, kind="stable")[:limit]
        method_state = {"kind": "chi2", "scores": scores.tolist()}
    elif selector == "mrmr_top500":
        selected = mrmr_classif(
            X=pd.DataFrame(binary, columns=context.gene_cols),
            y=pd.Series(context.train_y), K=limit, show_progress=False, n_jobs=1,
        )
        selected = list(selected)
        selected_set = set(selected)
        selected.extend(gene for gene in context.gene_cols if gene not in selected_set and len(selected) < limit)
        order = np.asarray([context.gene_cols.index(name) for name in selected], dtype=np.int64)
        method_state = {"kind": "mrmr", "selected_order": list(selected)}
    else:
        raise ValueError("unsupported gene selector")
    genes = [context.gene_cols[index] for index in order]
    return genes, method_state


def _ordinal_pair(train, valid, genes):
    hotspots = recurrent_tokens(train, genes, min_count=3)
    def transform(frame):
        output = np.empty((len(frame), len(genes)), dtype=np.float32)
        for column_index, gene in enumerate(genes):
            output[:, column_index] = [mutation_severity(value, hotspots[gene]) for value in frame[gene].tolist()]
        return output
    return transform(train), transform(valid), {gene: sorted(values) for gene, values in hotspots.items()}


def _type_pair(train, valid, context, selected_genes=None):
    train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
    if selected_genes is not None:
        prefixes = tuple(f"type|{gene}|" for gene in selected_genes)
        train_rows = [{key: value for key, value in row.items() if key.startswith(prefixes)} for row in train_rows]
        valid_rows = [{key: value for key, value in row.items() if key.startswith(prefixes)} for row in valid_rows]
    return _mcmp_block_pair(train_rows, valid_rows, context.train_groups, ["type"], 1)


def _our_tfidf_pair(train, valid, context):
    train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
    blocks = ["gene", "type", "aa", "aa_from", "aa_to", "gene_change", "site", "exact"]
    raw_train, raw_valid, encoder = _mcmp_block_pair(train_rows, valid_rows, context.train_groups, blocks, 3)
    out_train, out_valid, tfidf = _tfidf_pair(raw_train, raw_valid)
    return out_train, out_valid, {"encoder": encoder, "tfidf": tfidf}


def _base_state(declaration, context, fitted):
    return {
        "schema_version": "TEAM_FEATURE_STATE_V1",
        "candidate_id": declaration["candidate_id"],
        "config_hash": declaration["config_hash"],
        "family": declaration["family"],
        "parameters": declaration["parameters"],
        "fit_ids_sha256": _sequence_hash(context.train_ids),
        "fit_y_sha256": _sequence_hash(context.train_y),
        "fit_groups_sha256": _sequence_hash(context.train_groups),
        "valid_ids_sha256": _sequence_hash(context.valid_ids),
        "fitted": fitted,
    }


def fit_transform_candidate(declaration: dict, train: pd.DataFrame, valid: pd.DataFrame, context: FeatureContext) -> FeaturePair:
    _validate_context(train, valid, context)
    family = declaration.get("family")
    parameters = declaration.get("parameters", {})
    sample_multiplier = np.ones(len(train), dtype=np.float32)
    deferred_train = deferred_valid = None

    if family == "T1":
        train_binary = literal_gene_binary(train, context.gene_cols)
        valid_binary = literal_gene_binary(valid, context.gene_cols)
        names = [f"literal_presence::{gene}" for gene in context.gene_cols]
        if parameters["burden"] == "raw_mutated_gene_count":
            train_binary = np.column_stack([train_binary, train_binary.sum(axis=1)]).astype(np.float32)
            valid_binary = np.column_stack([valid_binary, valid_binary.sum(axis=1)]).astype(np.float32)
            names.append("burden::raw_mutated_gene_count")
        elif parameters["burden"] != "none":
            raise ValueError("unsupported T1 burden")
        if parameters["storage"] == "csr_implicit_missing":
            X_train, X_valid = sparse.csr_matrix(train_binary), sparse.csr_matrix(valid_binary)
        elif parameters["storage"] == "dense_explicit_zero":
            X_train, X_valid = train_binary, valid_binary
        else:
            raise ValueError("unsupported T1 storage")
        fitted = {"gene_cols": context.gene_cols, "storage": parameters["storage"], "burden": parameters["burden"]}

    elif family == "T2":
        presence_train = sparse.csr_matrix(literal_gene_binary(train, context.gene_cols))
        presence_valid = sparse.csr_matrix(literal_gene_binary(valid, context.gene_cols))
        burden_train = np.asarray(presence_train.sum(axis=1)).ravel().astype(np.float32)
        burden_valid = np.asarray(presence_valid.sum(axis=1)).ravel().astype(np.float32)
        pieces_train, pieces_valid = [presence_train], [presence_valid]
        names = [f"presence::{gene}" for gene in context.gene_cols]
        if parameters["view"] == "presence_exact_hash_burden":
            hasher = FeatureHasher(n_features=32768, input_type="string", alternate_sign=False)
            pieces_train.append(hasher.transform(exact_cell_token_rows(train, context.gene_cols)).astype(np.float32))
            pieces_valid.append(hasher.transform(exact_cell_token_rows(valid, context.gene_cols)).astype(np.float32))
            names.extend(f"variant_hash::{index}" for index in range(32768))
        elif parameters["view"] != "presence_burden":
            raise ValueError("unsupported T2 view")
        pieces_train.append(sparse.csr_matrix(np.column_stack([burden_train, np.log1p(burden_train)])))
        pieces_valid.append(sparse.csr_matrix(np.column_stack([burden_valid, np.log1p(burden_valid)])))
        names.extend(["burden::count", "burden::log1p"])
        X_train = sparse.hstack(pieces_train, format="csr", dtype=np.float32)
        X_valid = sparse.hstack(pieces_valid, format="csr", dtype=np.float32)
        fitted = {"gene_cols": context.gene_cols, "view": parameters["view"], "hash_dim": 32768, "alternate_sign": False}

    elif family == "T3":
        selected, selector_state = _select_genes(train, context, parameters["gene_selector"])
        numeric_train, numeric_valid, numeric_state = _numeric41(context)
        if parameters["encoding"] == "legacy_ordinal_annotation":
            base_train, base_valid, hotspots = _ordinal_pair(train, valid, selected)
            base_names = [f"ordinal::{gene}" for gene in selected]
            encoding_state = {"kind": "legacy_ordinal_annotation", "hotspots": hotspots}
        elif parameters["encoding"] == "categorical_gene_type_binary":
            base_train, base_valid, encoder = _type_pair(train, valid, context, selected)
            base_names = encoder["names"]
            encoding_state = {"kind": "categorical_gene_type_binary", "encoder": encoder}
        else:
            raise ValueError("unsupported T3 encoding")
        X_train, X_valid = _hstack(base_train, numeric_train), _hstack(base_valid, numeric_valid)
        names = base_names + [f"numeric41::{index}" for index in range(41)]
        fitted = {"selected_genes": selected, "selector": selector_state, "encoding": encoding_state, "numeric41": numeric_state}

    elif family == "T4":
        numeric_train, numeric_valid, numeric_state = _numeric41(context)
        if parameters["input"] == "our_B0_B1_support3_TFIDF":
            base_train, base_valid, input_state = _our_tfidf_pair(train, valid, context)
            base_names = input_state["encoder"]["names"]
        elif parameters["input"] == "kim_ordinal_chi2top500":
            selected, selector_state = _select_genes(train, context, "chi2_top500")
            base_train, base_valid, hotspots = _ordinal_pair(train, valid, selected)
            base_names = [f"ordinal::{gene}" for gene in selected]
            input_state = {"selected_genes": selected, "selector": selector_state, "hotspots": hotspots}
        else:
            raise ValueError("unsupported T4 input")
        if parameters["compressor"] == "SVD64":
            compressor = TruncatedSVD(n_components=64, n_iter=7, random_state=42)
            latent_train = compressor.fit_transform(base_train).astype(np.float32)
            latent_valid = compressor.transform(base_valid).astype(np.float32)
            scaler = StandardScaler()
            latent_train = scaler.fit_transform(latent_train).astype(np.float32) / 8.0
            latent_valid = scaler.transform(latent_valid).astype(np.float32) / 8.0
            X_train, X_valid = _hstack(latent_train, numeric_train), _hstack(latent_valid, numeric_valid)
            names = [f"latent64::{index}" for index in range(64)] + [f"numeric41::{index}" for index in range(41)]
            fitted = {
                "input": input_state, "compressor": "SVD64", "components": compressor.components_.tolist(),
                "latent_mean": scaler.mean_.tolist(), "latent_scale": scaler.scale_.tolist(), "numeric41": numeric_state,
            }
        elif parameters["compressor"] == "AE64":
            dense_required_bytes = int((base_train.shape[0] + base_valid.shape[0]) * base_train.shape[1] * 4 * 3)
            if dense_required_bytes > context.max_dense_bytes:
                raise ResourceBlockedError(
                    f"AE64 dense input requires {dense_required_bytes} bytes, limit {context.max_dense_bytes}"
                )
            if sparse.issparse(base_train):
                base_train = base_train.toarray().astype(np.float32)
                base_valid = base_valid.toarray().astype(np.float32)
            scaler = StandardScaler(with_mean=True)
            X_train = scaler.fit_transform(base_train).astype(np.float32)
            X_valid = scaler.transform(base_valid).astype(np.float32)
            names = list(base_names)
            fitted = {
                "input": input_state,
                "ae_input_mean": scaler.mean_.tolist(),
                "ae_input_scale": scaler.scale_.tolist(),
                "numeric41": numeric_state,
                "dense_required_bytes": dense_required_bytes,
                "dense_limit_bytes": context.max_dense_bytes,
            }
            deferred_train, deferred_valid = numeric_train, numeric_valid
        else:
            raise ValueError("unsupported T4 compressor")

    elif family == "T5":
        if parameters["view"] == "our_B0_B1_support3_TFIDF_numeric41":
            base_train, base_valid, input_state = _our_tfidf_pair(train, valid, context)
            numeric_train, numeric_valid, numeric_state = _numeric41(context)
            X_train, X_valid = _hstack(base_train, numeric_train), _hstack(base_valid, numeric_valid)
            names = input_state["encoder"]["names"] + [f"numeric41::{index}" for index in range(41)]
            fitted = {"input": input_state, "numeric41": numeric_state}
        elif parameters["view"] == "kim_presence_hash_burden":
            derived = {**declaration, "family": "T2", "parameters": {"view": "presence_exact_hash_burden"}}
            pair = fit_transform_candidate(derived, train, valid, context)
            X_train, X_valid, names = pair.train, pair.valid, pair.feature_names
            fitted = {"input": pair.state["fitted"]}
        else:
            raise ValueError("unsupported T5 view")
        if parameters["group_weight"] == "inverse_train_group_size":
            values, counts = np.unique(context.train_groups, return_counts=True)
            count_by_group = dict(zip(values.tolist(), counts.tolist()))
            sample_multiplier = np.asarray([1 / count_by_group[value] for value in context.train_groups], dtype=np.float32)
            sample_multiplier /= sample_multiplier.mean()
        elif parameters["group_weight"] != "row_uniform":
            raise ValueError("unsupported T5 group weight")
        fitted["group_weight"] = parameters["group_weight"]

    elif family == "T6":
        numeric_train, numeric_valid, numeric_state = _numeric41(context)
        if parameters["view"] == "gene_binary":
            base_train = literal_gene_binary(train, context.gene_cols)
            base_valid = literal_gene_binary(valid, context.gene_cols)
            base_names = [f"presence::{gene}" for gene in context.gene_cols]
            input_state = {"kind": "gene_binary", "gene_cols": context.gene_cols}
        elif parameters["view"] == "gene_type_binary":
            base_train, base_valid, encoder = _type_pair(train, valid, context)
            base_names = encoder["names"]
            input_state = {"kind": "gene_type_binary", "encoder": encoder}
        else:
            raise ValueError("unsupported T6 view")
        if parameters["per_gene_missing_flag"]:
            missing_train = (train[context.gene_cols].isna() | train[context.gene_cols].astype(str).apply(lambda col: col.str.strip().eq(""))).to_numpy(dtype=np.float32)
            missing_valid = (valid[context.gene_cols].isna() | valid[context.gene_cols].astype(str).apply(lambda col: col.str.strip().eq(""))).to_numpy(dtype=np.float32)
            base_train, base_valid = _hstack(base_train, missing_train), _hstack(base_valid, missing_valid)
            base_names += [f"missing::{gene}" for gene in context.gene_cols]
        X_train, X_valid = _hstack(base_train, numeric_train), _hstack(base_valid, numeric_valid)
        names = base_names + [f"numeric41::{index}" for index in range(41)]
        fitted = {"input": input_state, "missing_flags": bool(parameters["per_gene_missing_flag"]), "numeric41": numeric_state}

    else:
        raise ValueError("unsupported team feature family")

    state = _base_state(declaration, context, fitted)
    if family == "T4" and parameters["compressor"] == "AE64":
        state["deferred_compressor"] = "AE64"
        state["deferred_output_width"] = 64
    return FeaturePair(
        train=X_train,
        valid=X_valid,
        feature_names=list(names),
        state=state,
        sample_weight_multiplier=sample_multiplier,
        train_sha256=_matrix_hash(X_train),
        valid_sha256=_matrix_hash(X_valid),
        deferred_numeric_train=deferred_train,
        deferred_numeric_valid=deferred_valid,
    )
