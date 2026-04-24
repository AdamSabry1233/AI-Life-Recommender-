"""Unit tests for the per-user cluster layer (S2)."""
from __future__ import annotations

import numpy as np
import pytest

from clusters.cluster_fit import fit_user_clusters
from clusters.conditioning import condition
from clusters.posterior import entropy, posterior

pytestmark = [pytest.mark.unit]


class TestFit:
    def test_short_history_caps_k(self):
        V = np.eye(2, dtype="float32")
        centroids = fit_user_clusters(V, k=5)
        assert centroids.shape[0] == 2

    def test_clusters_separate_two_modes(self):
        rng = np.random.RandomState(0)
        A = rng.randn(20, 8).astype("float32") + np.array([5.0] + [0] * 7, dtype="float32")
        B = rng.randn(20, 8).astype("float32") + np.array([-5.0] + [0] * 7, dtype="float32")
        V = np.vstack([A, B])
        centroids = fit_user_clusters(V, k=2)
        # one centroid should be near +5 on axis 0, the other near -5
        signs = sorted(np.sign(centroids[:, 0]).tolist())
        assert signs == [-1.0, 1.0]


class TestPosterior:
    def test_sums_to_one(self):
        C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        p = posterior(np.array([1, 0, 0], dtype="float32"), C)
        assert pytest.approx(1.0, abs=1e-6) == float(p.sum())

    def test_concentrates_on_matching_cluster(self):
        C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        p = posterior(np.array([1, 0, 0], dtype="float32"), C, temp=0.1)
        assert int(np.argmax(p)) == 0 and p[0] > 0.9

    def test_entropy_zero_on_peak(self):
        p = np.array([1.0, 0.0, 0.0])
        assert entropy(p) == pytest.approx(0.0, abs=1e-6)

    def test_entropy_max_on_uniform(self):
        p = np.full(4, 0.25)
        assert entropy(p) == pytest.approx(2.0, abs=1e-6)  # log2(4) = 2


class TestCondition:
    def test_preserves_set_of_ids(self):
        mf = {1: 0.9, 2: 0.5, 3: 0.1}
        vecs = {1: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5]), 3: np.array([0.0, 1.0])}
        out = condition(mf, vecs, np.array([1.0, 0.0]), alpha=0.5)
        assert {i for i, _ in out} == {1, 2, 3}

    def test_alpha_zero_uses_cluster_proximity_only(self):
        mf = {1: 10.0, 2: -10.0}
        vecs = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
        out = condition(mf, vecs, np.array([0.0, 1.0]), alpha=0.0)
        assert out[0][0] == 2
