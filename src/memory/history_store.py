"""
FAISS-backed interaction history store.

Each entry stores a full interaction (intent + all 4 agent outputs) as a
normalized embedding so cosine similarity search works via inner product.

Layout on disk:
    data/memory/faiss.index   ← FAISS flat index
    data/memory/metadata.json ← parallel list of interaction dicts
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np

from memory.embedder import Embedder


class HistoryStore:
    def __init__(self, store_dir: str = "data/memory", embedder: Embedder = None):
        self.store_dir  = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "faiss.index"
        self.meta_path  = self.store_dir / "metadata.json"
        self.embedder   = embedder
        self.index      = None
        self.metadata: list[dict] = []
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index    = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text())
        else:
            self.index = None   # built lazily on first save (need dim from embedder)

    def _init_index(self, dim: int):
        # IndexFlatIP = exact inner product search (cosine sim on normalized vecs)
        self.index = faiss.IndexFlatIP(dim)

    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2, default=str))

    # ── public API ────────────────────────────────────────────────────────────

    def save(self, state: dict):
        """
        Embed the intent from state and store the full interaction.

        Parameters
        ----------
        state : completed AgentState dict (after all agents + orchestrator ran)
        """
        intent = state.get("intent", "")
        vec    = self.embedder.embed(intent).reshape(1, -1)

        if self.index is None:
            self._init_index(vec.shape[1])

        self.index.add(vec)
        self.metadata.append({
            "intent":    intent,
            "timestamp": int(time.time()),
            "movie":     state.get("movie_rec")  or {},
            "food":      state.get("food_rec")   or {},
            "book":      state.get("book_rec")   or {},
            "habit":     state.get("habit_rec")  or {},
            "summary":   state.get("final_response") or "",
        })
        self._persist()

    def search(self, intent: str, k: int = 3) -> list[dict]:
        """
        Return the k most similar past interactions for a given intent string.
        Returns [] if the store is empty.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        vec    = self.embedder.embed(intent).reshape(1, -1)
        k      = min(k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = self.metadata[idx].copy()
                entry["similarity"] = round(float(score), 3)
                results.append(entry)
        return results

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def format_context(past: list[dict]) -> str:
        """Format retrieved interactions into a prompt-ready memory block."""
        if not past:
            return ""
        lines = ["Relevant past interactions:"]
        for p in past:
            lines.append(f'\n  Past intent: "{p["intent"]}"')
            if p.get("movie"):
                lines.append(f'    Movie enjoyed: {p["movie"].get("name", "")}')
            if p.get("food"):
                lines.append(f'    Food enjoyed:  {p["food"].get("name", "")}')
            if p.get("book"):
                lines.append(f'    Book enjoyed:  {p["book"].get("name", "")}')
            if p.get("habit"):
                lines.append(f'    Habit used:    {p["habit"].get("habit", "")}')
        return "\n".join(lines)
