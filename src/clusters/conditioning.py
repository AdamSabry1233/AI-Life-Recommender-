"""Re-rank candidates by proximity to the active cluster centroid."""
from __future__ import annotations

import numpy as np


def condition(
    mf_scores: dict[int, float],
    item_vectors: dict[int, np.ndarray],
    active_centroid: np.ndarray,
    alpha: float = 0.5,
) -> list[tuple[int, float]]:
    """Return [(item_id, blended), ...] sorted desc. alpha weights MF vs cluster proximity."""
    c = active_centroid / (np.linalg.norm(active_centroid) + 1e-12)
    ids = [i for i in mf_scores if i in item_vectors]
    if not ids:
        return sorted(mf_scores.items(), key=lambda x: -x[1])
    mf = np.array([mf_scores[i] for i in ids], dtype="float32")
    V = np.stack([item_vectors[i] for i in ids])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    sim = V @ c
    mf_n = (mf - mf.min()) / (mf.max() - mf.min() + 1e-12)
    sim_n = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
    blended = alpha * mf_n + (1 - alpha) * sim_n
    order = np.argsort(-blended)
    return [(ids[i], float(blended[i])) for i in order]
