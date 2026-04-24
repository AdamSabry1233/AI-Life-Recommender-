"""Offline recommender metrics — kept flat, numpy-only."""
from __future__ import annotations

import numpy as np


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(relevant: set[int], ranked: list[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def coverage(all_ranked_lists: list[list[int]], catalog: set[int]) -> float:
    seen = set().union(*all_ranked_lists) if all_ranked_lists else set()
    return len(seen & catalog) / len(catalog) if catalog else 0.0


def diversity(ranked: list[int], item_vectors: dict[int, np.ndarray], k: int = 10) -> float:
    top = [i for i in ranked[:k] if i in item_vectors]
    if len(top) < 2:
        return 0.0
    V = np.stack([item_vectors[i] for i in top])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    sim = V @ V.T
    n = len(top)
    return float(1.0 - (sim.sum() - n) / (n * (n - 1)))  # 1 - mean pairwise cosine


def novelty(ranked: list[int], item_popularity: dict[int, int], n_users: int, k: int = 10) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return float(np.mean([-np.log2((item_popularity.get(i, 0) + 1) / (n_users + 1)) for i in top]))
