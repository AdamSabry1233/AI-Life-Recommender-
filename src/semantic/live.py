"""Runtime semantic-augmentation hook used inside the live agent pipeline.

Re-ranks a domain agent's MF candidate list by fusing collaborative scores with
sentence-transformer cosine similarity between (intent + preferences + memory)
and each candidate's metadata text. Toggled via USE_SEMANTIC_RETRIEVAL=1.
"""
from __future__ import annotations
import os
from typing import Optional

import numpy as np

_MODEL = None
_ALPHA = float(os.getenv("S1_ALPHA", "0.5"))


def enabled() -> bool:
    return os.getenv("USE_SEMANTIC_RETRIEVAL", "0").lower() in ("1", "true", "yes")


def _embedder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def augment(candidates: list[dict], query: str, alpha: Optional[float] = None) -> list[dict]:
    """Return candidates re-ranked by fused (MF, semantic) score.

    `query` should already be a concatenation of intent + user preferences + memory.
    Each candidate dict must carry `name`, `tags`, and `score`.
    """
    if not candidates or not query.strip():
        return candidates
    a = _ALPHA if alpha is None else alpha
    m = _embedder()
    q_vec = m.encode([query], normalize_embeddings=True).astype("float32")[0]
    texts = [f"{c.get('name','')} {c.get('tags','')}".strip() for c in candidates]
    cand_vecs = m.encode(texts, normalize_embeddings=True).astype("float32")
    sem = cand_vecs @ q_vec
    mf = np.array([c["score"] for c in candidates], dtype="float32")
    def _mm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)
    fused = a * _mm(mf) + (1.0 - a) * _mm(sem)
    order = np.argsort(-fused)
    out = []
    for i in order:
        c = dict(candidates[i])
        c["fused"] = float(fused[i])
        out.append(c)
    return out
