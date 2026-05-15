"""Runtime cluster-conditioning hook used inside the live agent pipeline.

Treats each onboarding-answer field (genres, dietary, book preferences, lifestyle)
as one of K = up-to-4 latent taste facets for the user. The current intent
selects the active facet via softmax over cosine similarity; MF candidates are
re-ranked by proximity to the active facet centroid. Toggled via
USE_CLUSTER_CONDITIONING=1.

This is the live, demoable version of S2 — the offline evaluation in
src/clusters/* uses KMeans on actual user interaction embeddings. The live
system has no per-user MF interaction history, so onboarding facets serve as
the K cluster proxies.
"""
from __future__ import annotations
import os
from typing import Optional

import numpy as np

_MODEL = None
_ALPHA = float(os.getenv("S2_ALPHA", "0.5"))


def enabled() -> bool:
    return os.getenv("USE_CLUSTER_CONDITIONING", "0").lower() in ("1", "true", "yes")


def _embedder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _split_facets(prefs: str) -> list[str]:
    """Pull individual onboarding facets out of the joined preferences string."""
    if not prefs:
        return []
    parts: list[str] = []
    for chunk in prefs.replace(";", ",").split(","):
        s = chunk.strip(" .\n\t")
        if s:
            parts.append(s)
    return parts[:4]  # cap at K = 4


def condition(candidates: list[dict], prefs: str, intent: str,
              alpha: Optional[float] = None) -> tuple[list[dict], dict]:
    """Return (re-ranked candidates, debug info).

    debug info carries:
      - 'facets'     : the parsed onboarding facets
      - 'posterior'  : softmax posterior over those facets given the intent
      - 'active'     : the active facet string
      - 'entropy'    : Shannon entropy of the posterior in bits
    """
    facets = _split_facets(prefs)
    if not candidates or len(facets) < 2:
        return candidates, {"facets": facets, "active": None, "posterior": None, "entropy": None}
    a = _ALPHA if alpha is None else alpha
    m = _embedder()
    centroids = m.encode(facets, normalize_embeddings=True).astype("float32")
    ctx = m.encode([intent or " "], normalize_embeddings=True).astype("float32")[0]
    sims = centroids @ ctx
    logits = sims - sims.max()
    post = np.exp(logits); post /= post.sum()
    active = centroids[int(np.argmax(post))]
    H = float(-(post * np.log2(np.clip(post, 1e-12, 1.0))).sum())

    texts = [f"{c.get('name','')} {c.get('tags','')}".strip() for c in candidates]
    cand_vecs = m.encode(texts, normalize_embeddings=True).astype("float32")
    prox = cand_vecs @ active
    mf = np.array([c["score"] for c in candidates], dtype="float32")
    def _mm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)
    blended = a * _mm(mf) + (1.0 - a) * _mm(prox)
    order = np.argsort(-blended)
    out = []
    for i in order:
        c = dict(candidates[i])
        c["blended"] = float(blended[i])
        out.append(c)

    debug = {
        "facets": facets,
        "posterior": post.tolist(),
        "active": facets[int(np.argmax(post))],
        "entropy": H,
    }
    return out, debug
