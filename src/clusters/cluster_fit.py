"""Fit K concurrent taste clusters per user via KMeans on their item embeddings."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def fit_user_clusters(item_vectors: np.ndarray, k: int = 3, seed: int = 42) -> np.ndarray:
    """Return K cluster centroids (shape [k, D]). Falls back to k=1 if history is short."""
    n = item_vectors.shape[0]
    if n == 0:
        return np.zeros((1, 1), dtype="float32")
    k_eff = min(k, n)
    km = KMeans(n_clusters=k_eff, random_state=seed, n_init=5).fit(item_vectors)
    return km.cluster_centers_.astype("float32")
