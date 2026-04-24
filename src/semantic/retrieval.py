"""Semantic item retriever. Embed item metadata once, retrieve by cosine."""
from __future__ import annotations
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """Wraps a sentence-transformers model + FAISS IndexFlatIP over item texts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.ids: np.ndarray | None = None
        self.vectors: np.ndarray | None = None

    def fit(self, item_ids: list[int], texts: list[str]) -> None:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.vectors = vecs.astype("float32")
        self.ids = np.asarray(item_ids, dtype=np.int64)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(self.vectors)

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True).astype("float32")[0]

    def search(self, query: np.ndarray, k: int = 30) -> list[tuple[int, float]]:
        assert self.index is not None, "call fit() first"
        if query.ndim == 1:
            query = query[None, :]
        D, I = self.index.search(query.astype("float32"), k)
        return [(int(self.ids[i]), float(d)) for i, d in zip(I[0], D[0])]
