"""
Text → vector embeddings.

Backend selected via INFERENCE_BACKEND env var:
  groq / any non-ollama  →  sentence-transformers (all-MiniLM-L6-v2, 384-dim, runs in-process)
  ollama (default)       →  OllamaEmbeddings (4096-dim, requires Ollama running)
"""

import os
import numpy as np


def _make_embedding_fn():
    if os.getenv("INFERENCE_BACKEND") == "groq":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url="http://host.docker.internal:11434",
        )


class Embedder:
    def __init__(self, model: str = "llama3.1", base_url: str = "http://host.docker.internal:11434"):
        self._emb = _make_embedding_fn()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            test    = self._emb.embed_query("test")
            self._dim = len(test)
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.array(self._emb.embed_query(text), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-10)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        vecs  = np.array(self._emb.embed_documents(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        return vecs / norms
