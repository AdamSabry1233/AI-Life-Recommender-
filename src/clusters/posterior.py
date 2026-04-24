"""Softmax posterior over a user's taste clusters given the current context vector."""
from __future__ import annotations

import numpy as np


def posterior(context_vec: np.ndarray, centroids: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Return p(cluster | context) via softmax over cosine similarity."""
    c = context_vec / (np.linalg.norm(context_vec) + 1e-12)
    C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    logits = (C @ c) / max(temp, 1e-6)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits; 0 = fully concentrated, log2(K) = uniform."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())
