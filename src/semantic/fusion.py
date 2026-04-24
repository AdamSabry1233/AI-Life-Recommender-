"""Fuse MF collaborative scores with semantic similarities into one ranked list."""
from __future__ import annotations

import numpy as np


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)


def fuse(
    mf_scores: dict[int, float],
    sem_scores: dict[int, float],
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Return [(item_id, fused_score), ...] sorted desc. alpha weights MF."""
    ids = sorted(set(mf_scores) | set(sem_scores))
    mf = np.array([mf_scores.get(i, 0.0) for i in ids], dtype="float32")
    sm = np.array([sem_scores.get(i, 0.0) for i in ids], dtype="float32")
    fused = alpha * _minmax(mf) + (1 - alpha) * _minmax(sm)
    order = np.argsort(-fused)
    return [(ids[i], float(fused[i])) for i in order]
