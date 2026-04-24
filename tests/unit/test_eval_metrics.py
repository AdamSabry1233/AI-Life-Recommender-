"""Unit tests for offline metric helpers."""
from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import coverage, diversity, ndcg_at_k, novelty, recall_at_k

pytestmark = [pytest.mark.unit]


def test_ndcg_perfect():
    assert ndcg_at_k({1, 2, 3}, [1, 2, 3, 4, 5], k=3) == pytest.approx(1.0)


def test_ndcg_empty_relevant():
    assert ndcg_at_k(set(), [1, 2, 3], k=3) == 0.0


def test_recall():
    assert recall_at_k({1, 2, 3}, [1, 4, 5, 2], k=4) == pytest.approx(2 / 3)


def test_coverage_counts_distinct_items():
    catalog = {1, 2, 3, 4}
    assert coverage([[1, 2], [2, 3]], catalog) == 0.75


def test_diversity_zero_for_identical_vectors():
    v = np.array([1.0, 0.0], dtype="float32")
    assert diversity([1, 2], {1: v, 2: v}, k=2) == pytest.approx(0.0, abs=1e-5)


def test_diversity_one_for_orthogonal():
    v1 = np.array([1.0, 0.0], dtype="float32")
    v2 = np.array([0.0, 1.0], dtype="float32")
    assert diversity([1, 2], {1: v1, 2: v2}, k=2) == pytest.approx(1.0, abs=1e-5)


def test_novelty_higher_for_rarer_items():
    pop = {1: 100, 2: 1}
    n = 100
    assert novelty([2], pop, n, k=1) > novelty([1], pop, n, k=1)
