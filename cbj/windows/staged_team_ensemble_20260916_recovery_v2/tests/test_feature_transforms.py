from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from feature_transforms import FeatureContext, ResourceBlockedError, fit_transform_candidate


def fixture(rows=90, genes=70):
    gene_cols = [f"G{i:03d}" for i in range(genes)]
    values = {}
    for j, gene in enumerate(gene_cols):
        column = []
        for i in range(rows):
            if (i + j) % 17 == 0:
                column.append(None)
            elif (i * (j + 1)) % 11 == 0:
                column.append(f"A{j + 1}V")
            elif (i + 2 * j) % 13 == 0:
                column.append(f"A{j + 1}A")
            else:
                column.append("WT")
        values[gene] = column
    frame = pd.DataFrame(values)
    split = int(rows * 0.8)
    train = frame.iloc[:split].reset_index(drop=True)
    valid = frame.iloc[split:].reset_index(drop=True)
    classes = ["C0", "C1", "C2"]
    y = np.asarray([classes[i % 3] for i in range(len(train))])
    groups = np.asarray([f"GR{i:03d}" for i in range(len(train))])
    numeric_train = np.arange(len(train) * 41, dtype=np.float32).reshape(len(train), 41) % 7
    numeric_valid = np.arange(len(valid) * 41, dtype=np.float32).reshape(len(valid), 41) % 7
    context = FeatureContext(
        gene_cols=gene_cols,
        train_ids=np.asarray([f"TR{i:03d}" for i in range(len(train))]),
        valid_ids=np.asarray([f"VA{i:03d}" for i in range(len(valid))]),
        train_groups=groups,
        train_y=y,
        class_order=classes,
        numeric41_train=numeric_train,
        numeric41_valid=numeric_valid,
    )
    return train, valid, context


def declaration(family, **parameters):
    return {
        "candidate_id": f"TEAM-{family}-TEST",
        "config_hash": "a" * 64,
        "source_kind": "TEAM",
        "family": family,
        "parameters": parameters,
        "fixed": {},
    }


class FeatureTransformTests(unittest.TestCase):
    def test_t1_preserves_csr_missing_vs_dense_zero_and_optional_burden(self):
        train, valid, context = fixture(30, 5)
        csr_pair = fit_transform_candidate(
            declaration("T1", storage="csr_implicit_missing", burden="raw_mutated_gene_count", class_weight="none"),
            train, valid, context,
        )
        dense_pair = fit_transform_candidate(
            declaration("T1", storage="dense_explicit_zero", burden="none", class_weight="sqrt_inverse"),
            train, valid, context,
        )
        self.assertTrue(sparse.isspmatrix_csr(csr_pair.train))
        self.assertEqual(csr_pair.train.shape[1], 6)
        self.assertIsInstance(dense_pair.train, np.ndarray)
        self.assertEqual(dense_pair.train.shape[1], 5)
        np.testing.assert_array_equal(csr_pair.train[:, -1].toarray().ravel(), dense_pair.train.sum(axis=1))

    def test_t2_hash_view_adds_exact_32768_block_without_learning_valid_tokens(self):
        train, valid, context = fixture(30, 5)
        plain = fit_transform_candidate(
            declaration("T2", view="presence_burden", min_samples_leaf=2, max_features="sqrt"),
            train, valid, context,
        )
        hashed = fit_transform_candidate(
            declaration("T2", view="presence_exact_hash_burden", min_samples_leaf=2, max_features="sqrt"),
            train, valid, context,
        )
        self.assertEqual(plain.train.shape[1], 7)
        self.assertEqual(hashed.train.shape[1], 5 + 32768 + 2)
        changed = valid.copy(); changed.iloc[0, 0] = "VALID_ONLY_TOKEN"
        changed_pair = fit_transform_candidate(
            declaration("T2", view="presence_exact_hash_burden", min_samples_leaf=2, max_features="sqrt"),
            train, changed, context,
        )
        self.assertEqual(hashed.state["fitted"], changed_pair.state["fitted"])
        self.assertNotEqual(hashed.valid_sha256, changed_pair.valid_sha256)

    def test_t3_selectors_and_t6_missing_global_are_fold_train_fitted(self):
        train, valid, context = fixture()
        ordinal = fit_transform_candidate(
            declaration("T3", encoding="legacy_ordinal_annotation", gene_selector="chi2_top500", model="XGB200_depth3_lr0.1"),
            train, valid, context,
        )
        typed = fit_transform_candidate(
            declaration("T3", encoding="categorical_gene_type_binary", gene_selector="mrmr_top500", model="ExtraTrees500_entropy_leaf2"),
            train, valid, context,
        )
        t6 = fit_transform_candidate(
            declaration("T6", view="gene_type_binary", per_gene_missing_flag=True, model="XGB200_depth3_lr0.1"),
            train, valid, context,
        )
        self.assertEqual(len(ordinal.state["fitted"]["selected_genes"]), 70)
        self.assertEqual(len(typed.state["fitted"]["selected_genes"]), 70)
        self.assertEqual(ordinal.train.shape[1], 70 + 41)
        self.assertGreater(typed.train.shape[1], 41)
        self.assertGreaterEqual(t6.train.shape[1], 70 + 70 + 41)
        changed = valid.copy(); changed.iloc[:, :] = "VALID_ONLY"
        replay = fit_transform_candidate(
            declaration("T6", view="gene_type_binary", per_gene_missing_flag=True, model="XGB200_depth3_lr0.1"),
            train, changed, context,
        )
        self.assertEqual(t6.state["fitted"], replay.state["fitted"])

    def test_t4_svd_and_ae_contract_and_t5_group_weights(self):
        train, valid, context = fixture()
        svd = fit_transform_candidate(
            declaration("T4", input="our_B0_B1_support3_TFIDF", compressor="SVD64", head="logreg_C1"),
            train, valid, context,
        )
        ae = fit_transform_candidate(
            declaration("T4", input="kim_ordinal_chi2top500", compressor="AE64", head="LightGBM_fixed"),
            train, valid, context,
        )
        groups = context.train_groups.copy(); groups[1] = groups[0]
        grouped_context = copy.copy(context); grouped_context.train_groups = groups
        weighted = fit_transform_candidate(
            declaration("T5", view="kim_presence_hash_burden", group_weight="inverse_train_group_size", model="ExtraTrees500_entropy_leaf5"),
            train, valid, grouped_context,
        )
        self.assertEqual(svd.train.shape[1], 64 + 41)
        self.assertEqual(ae.state["deferred_compressor"], "AE64")
        self.assertEqual(ae.state["deferred_output_width"], 64)
        self.assertAlmostEqual(weighted.sample_weight_multiplier[0], weighted.sample_weight_multiplier[1])
        self.assertLess(weighted.sample_weight_multiplier[0], weighted.sample_weight_multiplier[2])
        self.assertAlmostEqual(float(weighted.sample_weight_multiplier.mean()), 1.0)

    def test_t4_ae_densification_fails_before_allocation_when_memory_is_insufficient(self):
        train, valid, context = fixture()
        context.max_dense_bytes = 1
        with self.assertRaisesRegex(ResourceBlockedError, "AE64 dense input"):
            fit_transform_candidate(
                declaration("T4", input="our_B0_B1_support3_TFIDF", compressor="AE64", head="logreg_C1"),
                train, valid, context,
            )


if __name__ == "__main__":
    unittest.main()
