"""Unit tests for the semantic retrieval layer (S1)."""
from __future__ import annotations

import numpy as np
import pytest

from semantic.fusion import fuse
from semantic.profile import user_profile

pytestmark = [pytest.mark.unit]


class TestProfile:
    def test_empty_history_returns_zero_vector(self):
        v = user_profile(np.zeros((0, 8), dtype="float32"))
        assert v.shape == (8,) and np.all(v == 0)

    def test_unit_norm_on_nonzero_input(self):
        V = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
        v = user_profile(V)
        assert pytest.approx(1.0, abs=1e-6) == float(np.linalg.norm(v))

    def test_weighted_mean_biases_toward_higher_weight(self):
        V = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
        v = user_profile(V, weights=np.array([0.1, 0.9]))
        assert v[1] > v[0]


class TestFusion:
    def test_fuse_preserves_ids_from_both_sources(self):
        mf = {1: 0.9, 2: 0.4}
        sm = {2: 0.8, 3: 0.2}
        out = fuse(mf, sm, alpha=0.5)
        assert {i for i, _ in out} == {1, 2, 3}

    def test_fuse_sorts_descending(self):
        mf = {1: 1.0, 2: 0.5, 3: 0.0}
        sm = {1: 1.0, 2: 0.5, 3: 0.0}
        out = fuse(mf, sm, alpha=0.5)
        assert [i for i, _ in out] == [1, 2, 3]

    def test_alpha_zero_ignores_mf(self):
        mf = {1: 10.0, 2: -10.0}
        sm = {1: 0.0,  2: 1.0}
        out = fuse(mf, sm, alpha=0.0)
        assert out[0][0] == 2

    def test_alpha_one_ignores_semantic(self):
        mf = {1: 10.0, 2: -10.0}
        sm = {1: 0.0,  2: 1.0}
        out = fuse(mf, sm, alpha=1.0)
        assert out[0][0] == 1
