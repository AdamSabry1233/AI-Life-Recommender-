"""Shared tiny helper to embed item metadata into float32 vectors."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def embed_items(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
