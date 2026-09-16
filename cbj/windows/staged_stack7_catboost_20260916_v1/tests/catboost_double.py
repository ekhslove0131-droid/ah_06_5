"""Specific in-process CatBoost boundary double for Mac synthetic tests."""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import numpy as np


class FakeCatBoostClassifier:
    fit_calls = 0

    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.effective_parameters = dict(parameters)
        self.classes_ = np.asarray([], dtype=np.int64)
        self.tree_count_ = 0
        self.n_features_in_ = 0
        self.feature_names_ = []
        self._centroids = None

    def fit(self, matrix, labels, sample_weight=None):
        type(self).fit_calls += 1
        matrix = np.asarray(matrix, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        weights = np.ones(len(labels), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(labels)
        self.tree_count_ = int(self.parameters["iterations"])
        self.n_features_in_ = int(matrix.shape[1])
        self.feature_names_ = [str(index) for index in range(matrix.shape[1])]
        self._centroids = np.stack([
            np.average(matrix[labels == value], axis=0, weights=weights[labels == value])
            for value in self.classes_
        ])
        return self

    def get_params(self):
        return dict(self.parameters)

    def get_all_params(self):
        return dict(self.effective_parameters)

    def predict_proba(self, matrix, thread_count=None):
        if thread_count != 4:
            raise AssertionError("CPU inference must request thread_count=4")
        matrix = np.asarray(matrix, dtype=np.float64)
        distance = ((matrix[:, None, :] - self._centroids[None, :, :]) ** 2).mean(axis=2)
        score = np.exp(-(distance - distance.min(axis=1, keepdims=True)))
        return score / score.sum(axis=1, keepdims=True)


def admitted_gpu():
    return {
        "index": 0,
        "name": "Synthetic GPU",
        "uuid": "GPU-SYNTHETIC",
        "driver_version": "test",
        "memory_total_mib": 8192,
        "memory_free_mib": 6144,
    }


@contextmanager
def patched_backend():
    FakeCatBoostClassifier.fit_calls = 0
    with mock.patch("meta_core._load_catboost", return_value=(FakeCatBoostClassifier, lambda: 1)), \
            mock.patch("meta_core._query_gpu0", return_value=admitted_gpu()):
        yield
