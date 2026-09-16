"""Fold-fitted feature pipelines for the existing G1-G4 grid families."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from feature_transforms import (
    FeatureContext,
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
from vendor import mcmp_repr


COARSE = ["gene", "type", "aa", "aa_from", "aa_to"]
FINE = ["gene_change", "site", "exact"]


def _position_rows(frame, genes, base_rows, groups, width, min_support):
    additions = []
    support = defaultdict(set)
    for row_index, (_, row) in enumerate(frame[genes].iterrows()):
        values = {}
        for gene in genes:
            for token in mcmp_repr.split_tokens(row[gene]):
                kind, _, _, _, _ = mcmp_repr.classify_token(token)
                site, _ = mcmp_repr.safe_site(token)
                if site is None:
                    continue
                start = int(site.split("-", 1)[0])
                key = (gene, kind, (start - 1) // width)
                values[key] = 1.0
                support[key].add(str(groups[row_index]))
        additions.append(values)
    allowed = {key for key, seen in support.items() if len(seen) >= min_support}
    output = []
    for base, values in zip(base_rows, additions):
        merged = {key: value for key, value in base.items() if mcmp_repr.block_of(key) in COARSE}
        for (gene, kind, bin_index), value in values.items():
            if (gene, kind, bin_index) in allowed:
                merged[f"position_bin|{gene}|{kind}|{bin_index}"] = value
        output.append(merged)
    return output, allowed


def _g1_pair(train, valid, context, width, support):
    train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
    prepared_train, allowed_bins = _position_rows(
        train, context.gene_cols, train_rows, context.train_groups, width, support
    )
    prepared_valid = []
    for row, (_, raw) in zip(valid_rows, valid[context.gene_cols].iterrows()):
        merged = {key: value for key, value in row.items() if mcmp_repr.block_of(key) in COARSE}
        for gene in context.gene_cols:
            for token in mcmp_repr.split_tokens(raw[gene]):
                kind, _, _, _, _ = mcmp_repr.classify_token(token)
                site, _ = mcmp_repr.safe_site(token)
                if site is None:
                    continue
                bin_index = (int(site.split("-", 1)[0]) - 1) // width
                if (gene, kind, bin_index) in allowed_bins:
                    merged[f"position_bin|{gene}|{kind}|{bin_index}"] = 1.0
        prepared_valid.append(merged)
    vectorizer = DictVectorizer(sparse=True, sort=True)
    raw_train = vectorizer.fit_transform(prepared_train).astype(np.float32).tocsr()
    mapping = {name: index for index, name in enumerate(vectorizer.get_feature_names_out().tolist())}
    def transform(rows):
        data, indices, indptr = [], [], [0]
        for row in rows:
            pairs = sorted((mapping[key], float(value)) for key, value in row.items() if key in mapping)
            indices.extend(index for index, _ in pairs); data.extend(value for _, value in pairs); indptr.append(len(data))
        return sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32), np.asarray(indptr, np.int32)), shape=(len(rows), len(mapping)))
    raw_valid = transform(prepared_valid)
    out_train, out_valid, tfidf = _tfidf_pair(raw_train, raw_valid)
    return out_train, out_valid, vectorizer.get_feature_names_out().tolist(), {
        "position_bin_width": width,
        "min_group_support": support,
        "allowed_position_bins": [list(value) for value in sorted(allowed_bins)],
        "tfidf": tfidf,
    }


def _nb_ratios(binary_matrix, y, classes, alpha):
    width = binary_matrix.shape[1]
    if alpha is None:
        return np.ones((len(classes), width), dtype=np.float32)
    result = np.empty((len(classes), width), dtype=np.float32)
    binary = (binary_matrix > 0).astype(np.float64).tocsr()
    for class_index, label in enumerate(classes):
        positive = np.asarray(binary[np.asarray(y) == label].sum(axis=0)).ravel()
        negative = np.asarray(binary[np.asarray(y) != label].sum(axis=0)).ravel()
        positive = alpha + positive; negative = alpha + negative
        result[class_index] = (np.log(positive / positive.sum()) - np.log(negative / negative.sum())).astype(np.float32)
    return result


def fit_transform_old_candidate(declaration, train, valid, context: FeatureContext) -> FeaturePair:
    _validate_context(train, valid, context)
    family = declaration["family"]
    parameters = declaration["parameters"]
    numeric_train, numeric_valid, numeric_state = _numeric41(context)
    class_reweights = None
    symbolic_width = 0

    if family == "G1":
        symbolic_train, symbolic_valid, names, fitted = _g1_pair(
            train, valid, context, int(parameters["position_bin_width"]), int(parameters["min_group_support"])
        )
        X_train, X_valid = _hstack(symbolic_train, numeric_train), _hstack(symbolic_valid, numeric_valid)
        names += [f"numeric41::{index}" for index in range(41)]
        fitted["numeric41"] = numeric_state

    elif family == "G2":
        train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
        coarse_train, coarse_valid, coarse_encoder = _mcmp_block_pair(train_rows, valid_rows, context.train_groups, COARSE, 1)
        fine_train, fine_valid, fine_encoder = _mcmp_block_pair(
            train_rows, valid_rows, context.train_groups, FINE, int(parameters["fine_min_group_support"])
        )
        coarse_train, coarse_valid, coarse_tfidf = _tfidf_pair(coarse_train, coarse_valid)
        fine_train, fine_valid, fine_tfidf = _tfidf_pair(fine_train, fine_valid)
        weight = float(parameters["fine_block_weight"])
        symbolic_train = sparse.hstack([coarse_train, fine_train * weight], format="csr", dtype=np.float32)
        symbolic_valid = sparse.hstack([coarse_valid, fine_valid * weight], format="csr", dtype=np.float32)
        X_train, X_valid = _hstack(symbolic_train, numeric_train), _hstack(symbolic_valid, numeric_valid)
        names = coarse_encoder["names"] + fine_encoder["names"] + [f"numeric41::{index}" for index in range(41)]
        fitted = {
            "coarse_encoder": coarse_encoder, "fine_encoder": fine_encoder,
            "coarse_tfidf": coarse_tfidf, "fine_tfidf": fine_tfidf,
            "fine_weight": weight, "numeric41": numeric_state,
        }

    elif family == "G3":
        train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
        blocks = COARSE if parameters["view"] == "B0" else COARSE + FINE
        support = 1 if parameters["view"] == "B0" else 3
        raw_train, raw_valid, encoder = _mcmp_block_pair(train_rows, valid_rows, context.train_groups, blocks, support)
        symbolic_train, symbolic_valid, tfidf = _tfidf_pair(raw_train, raw_valid)
        symbolic_width = symbolic_train.shape[1]
        reweight = parameters["reweight"]
        alpha = None if reweight == "none" else float(reweight.replace("nb_alpha", ""))
        class_reweights = _nb_ratios(raw_train, context.train_y, context.class_order, alpha)
        X_train, X_valid = _hstack(symbolic_train, numeric_train), _hstack(symbolic_valid, numeric_valid)
        names = encoder["names"] + [f"numeric41::{index}" for index in range(41)]
        fitted = {"encoder": encoder, "tfidf": tfidf, "reweight": reweight, "alpha": alpha, "numeric41": numeric_state}

    elif family == "G4":
        train_rows, valid_rows = _mcmp_rows(train, valid, context.gene_cols)
        raw_train, raw_valid, encoder = _mcmp_block_pair(train_rows, valid_rows, context.train_groups, COARSE, 1)
        symbolic_train, symbolic_valid, tfidf = _tfidf_pair(raw_train, raw_valid)
        rank = int(parameters["components"])
        if rank >= min(symbolic_train.shape):
            raise ValueError("G4 components must be smaller than both matrix dimensions")
        if parameters["decomposition"] == "TruncatedSVD":
            decomposition = TruncatedSVD(n_components=rank, n_iter=7, random_state=42)
            latent_train = decomposition.fit_transform(symbolic_train)
            latent_valid = decomposition.transform(symbolic_valid)
            component_state = decomposition.components_.tolist()
        elif parameters["decomposition"] == "NMF":
            decomposition = NMF(
                n_components=rank, init="nndsvda", solver="cd", beta_loss="frobenius",
                max_iter=1000, tol=1e-4, random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                latent_train = decomposition.fit_transform(symbolic_train)
                latent_valid = decomposition.transform(symbolic_valid)
            component_state = decomposition.components_.tolist()
        else:
            raise ValueError("unsupported G4 decomposition")
        scaler = StandardScaler()
        latent_train = scaler.fit_transform(latent_train).astype(np.float32) / math.sqrt(rank)
        latent_valid = scaler.transform(latent_valid).astype(np.float32) / math.sqrt(rank)
        X_train, X_valid = _hstack(latent_train, numeric_train), _hstack(latent_valid, numeric_valid)
        names = [f"latent{rank}::{index}" for index in range(rank)] + [f"numeric41::{index}" for index in range(41)]
        fitted = {
            "encoder": encoder, "tfidf": tfidf, "decomposition": parameters["decomposition"],
            "components": component_state, "latent_mean": scaler.mean_.tolist(),
            "latent_scale": scaler.scale_.tolist(), "numeric41": numeric_state,
        }
    else:
        raise ValueError("old candidate transformer supports only G1-G4")

    state = _base_state(declaration, context, fitted)
    return FeaturePair(
        train=X_train, valid=X_valid, feature_names=names, state=state,
        sample_weight_multiplier=np.ones(len(train), dtype=np.float32),
        train_sha256=_matrix_hash(X_train), valid_sha256=_matrix_hash(X_valid),
        class_reweights=class_reweights, symbolic_width=symbolic_width,
    )
