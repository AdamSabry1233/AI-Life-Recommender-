"""User profile vector: weighted mean of embeddings for the user's rated items."""
from __future__ import annotations

import numpy as np


def user_profile(item_vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean (or rating-weighted mean) of the user's rated item embeddings, L2-normalized."""
    if item_vectors.shape[0] == 0:
        return np.zeros(item_vectors.shape[1], dtype="float32")
    v = np.average(item_vectors, axis=0, weights=weights).astype("float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
