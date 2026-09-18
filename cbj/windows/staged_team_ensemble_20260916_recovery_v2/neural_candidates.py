"""Fold-fitted input contracts for GPU-only G5 FM and G6 token MLP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse

from feature_transforms import FeatureContext, _numeric41, _type_pair, _validate_context
from raw_views import literal_gene_binary
from vendor import mcmp_repr


@dataclass
class PreparedNeuralCandidate:
    train: dict
    valid: dict
    state: dict


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _missing(frame, genes):
    return (
        frame[genes].isna()
        | frame[genes].astype(str).apply(lambda column: column.str.strip().eq(""))
    ).to_numpy(dtype=np.float32)


def _row_normalize(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _fit_token_state(train, context):
    gene_support = defaultdict(set)
    kinds, aa_from, aa_to = set(), set(), set()
    positions = []
    for row_index, (_, row) in enumerate(train[context.gene_cols].iterrows()):
        group = str(context.train_groups[row_index])
        for gene in context.gene_cols:
            tokens = mcmp_repr.split_tokens(row[gene])
            if tokens:
                gene_support[gene].add(group)
            for token in tokens:
                kind, _, _, source, target = mcmp_repr.classify_token(token)
                kinds.add(kind)
                if source is not None:
                    aa_from.add(source)
                if target is not None:
                    aa_to.add(target)
                site, _ = mcmp_repr.safe_site(token)
                if site is not None:
                    positions.append(np.log1p(int(site.split("-", 1)[0])))
    genes = sorted(gene for gene, groups in gene_support.items() if len(groups) >= 2)
    position_mean = float(np.mean(positions)) if positions else 0.0
    position_scale = float(np.std(positions)) if positions and np.std(positions) > 0 else 1.0
    return {
        "gene_vocabulary": genes,
        "kind_vocabulary": sorted(kinds),
        "aa_from_vocabulary": sorted(aa_from),
        "aa_to_vocabulary": sorted(aa_to),
        "position_mean": position_mean,
        "position_scale": position_scale,
        "gene_min_group_support": 2,
    }


def _index(values):
    return {value: index + 1 for index, value in enumerate(values)}


def _tokenize(frame, genes, fitted):
    gene_index = _index(fitted["gene_vocabulary"])
    kind_index = _index(fitted["kind_vocabulary"])
    source_index = _index(fitted["aa_from_vocabulary"])
    target_index = _index(fitted["aa_to_vocabulary"])
    output = []
    for _, row in frame[genes].iterrows():
        encoded = []
        for gene in genes:
            for token in mcmp_repr.split_tokens(row[gene]):
                kind, _, _, source, target = mcmp_repr.classify_token(token)
                site, _ = mcmp_repr.safe_site(token)
                if site is None:
                    position = 0.0
                    unknown_position = 1.0
                else:
                    raw_position = np.log1p(int(site.split("-", 1)[0]))
                    position = float((raw_position - fitted["position_mean"]) / fitted["position_scale"])
                    unknown_position = 0.0
                encoded.append({
                    "gene": gene_index.get(gene, 0),
                    "kind": kind_index.get(kind, 0),
                    "aa_from": source_index.get(source, 0),
                    "aa_to": target_index.get(target, 0),
                    "position": position,
                    "unknown_position": unknown_position,
                })
        output.append(encoded)
    return output


def prepare_neural_candidate(declaration, train: pd.DataFrame, valid: pd.DataFrame, context: FeatureContext) -> PreparedNeuralCandidate:
    _validate_context(train, valid, context)
    family = declaration["family"]
    parameters = declaration["parameters"]
    numeric_train, numeric_valid, numeric_state = _numeric41(context)
    common = {
        "schema_version": "NEURAL_CANDIDATE_INPUT_V1",
        "candidate_id": declaration["candidate_id"],
        "config_hash": declaration["config_hash"],
        "family": family,
        "parameters": parameters,
        "fit_ids_sha256": _hash(context.train_ids.tolist()),
        "fit_y_sha256": _hash(context.train_y.tolist()),
        "fit_groups_sha256": _hash(context.train_groups.tolist()),
        "device_kind": "GPU_REQUIRED",
        "token_truncation": False,
    }
    if family == "G5":
        if parameters["view"] == "gene_binary":
            symbolic_train = literal_gene_binary(train, context.gene_cols)
            symbolic_valid = literal_gene_binary(valid, context.gene_cols)
            fitted = {"view": "gene_binary", "names": list(context.gene_cols)}
        elif parameters["view"] == "gene_type_binary":
            symbolic_train, symbolic_valid, encoder = _type_pair(train, valid, context)
            fitted = {"view": "gene_type_binary", "encoder": encoder}
        else:
            raise ValueError("unsupported G5 view")
        train_payload = {
            "symbolic": _row_normalize(symbolic_train),
            "missing": _missing(train, context.gene_cols),
            "numeric41": numeric_train,
        }
        valid_payload = {
            "symbolic": _row_normalize(symbolic_valid),
            "missing": _missing(valid, context.gene_cols),
            "numeric41": numeric_valid,
        }
        architecture = {
            "kind": "multiclass_second_order_fm",
            "classes": len(context.class_order),
            "interaction_rank": int(parameters["interaction_rank"]),
            "weight_decay": float(parameters["weight_decay"]),
            "learning_rate": 0.001,
            "epochs": 80,
            "batch_rows": 128,
            "seed": 42,
            "optimizer": "AdamW",
        }
    elif family == "G6":
        fitted = _fit_token_state(train, context)
        train_payload = {"tokens": _tokenize(train, context.gene_cols, fitted), "numeric41": numeric_train}
        valid_payload = {"tokens": _tokenize(valid, context.gene_cols, fitted), "numeric41": numeric_valid}
        architecture = {
            "kind": "two_layer_token_mlp",
            "classes": len(context.class_order),
            "token_embedding_dim": int(parameters["token_embedding_dim"]),
            "pool": parameters["pool"],
            "dropout": float(parameters["dropout"]),
            "weight_decay": 0.01,
            "learning_rate": 0.001,
            "epochs": 60,
            "batch_rows": 32,
            "seed": 42,
            "optimizer": "AdamW",
        }
    else:
        raise ValueError("neural input preparation supports only G5/G6")
    state = {**common, "fitted": fitted, "numeric41": numeric_state, "architecture": architecture}
    return PreparedNeuralCandidate(train=train_payload, valid=valid_payload, state=state)

