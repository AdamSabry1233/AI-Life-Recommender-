"""
Phase 3 — MF Inference Wrapper

Loads mf_best.pt and the lookup CSVs produced by scripts/build_lookup_tables.py,
then exposes get_top_n(domain_id, user_local_idx, n) for use by the pipeline.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

DOMAIN_MOVIES = 0
DOMAIN_FOOD   = 1
DOMAIN_BOOKS  = 2

# ── model (must match train_mf.py exactly) ────────────────────────────────────

class MultiDomainMF(nn.Module):
    def __init__(self, n_users, n_items, n_domains, n_factors):
        super().__init__()
        self.user_emb    = nn.Embedding(n_users,   n_factors)
        self.item_emb    = nn.Embedding(n_items,   n_factors)
        self.domain_emb  = nn.Embedding(n_domains, n_factors)
        self.user_bias   = nn.Embedding(n_users, 1)
        self.item_bias   = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item, domain):
        u        = self.user_emb(user)
        i        = self.item_emb(item)
        d        = self.domain_emb(domain)
        user_ctx = u + d
        dot      = (user_ctx * i).sum(dim=1)
        bias     = (self.user_bias(user).squeeze(1)
                  + self.item_bias(item).squeeze(1)
                  + self.global_bias)
        return dot + bias

# ── recommender ───────────────────────────────────────────────────────────────

class MultiDomainRecommender:
    """
    Loads a trained MF checkpoint and lookup tables, then scores items for a
    given (domain, demo_user) pair.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to mf_best.pt
    data_dir : str | Path
        Project data root (expects data/processed/*_lookup.csv files)
    """

    def __init__(self, checkpoint_path="mf_best.pt", data_dir="data"):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model = MultiDomainMF(
            ckpt["n_users"], ckpt["n_items"], ckpt["n_domains"], ckpt["n_factors"]
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.domain_meta = ckpt["domain_meta"]

        # For each domain: local_embedding_idx → processed_item_idx (0-based)
        self.idx2processed = {
            d: {v: k for k, v in meta["item2idx"].items()}
            for d, meta in self.domain_meta.items()
        }

        self._load_lookups(Path(data_dir) / "processed")

    def _load_lookups(self, proc_dir: Path):
        lookup_files = {
            DOMAIN_MOVIES: ("movie_lookup.csv", "title",  "genre_str"),
            DOMAIN_FOOD:   ("food_lookup.csv",  "name",   "tags"),
            DOMAIN_BOOKS:  ("book_lookup.csv",  "title",  None),
        }

        self.lookups = {}
        for domain_id, (fname, name_col, tag_col) in lookup_files.items():
            path = proc_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run scripts/build_lookup_tables.py first."
                )
            df = pd.read_csv(path, index_col="item_idx")
            self.lookups[domain_id] = (df, name_col, tag_col)

    # ── public API ────────────────────────────────────────────────────────────

    def get_top_n(self, domain_id: int, user_local_idx: int = 0, n: int = 10) -> list[dict]:
        """
        Return the top-n items for a given domain and local user index.

        Parameters
        ----------
        domain_id       : DOMAIN_MOVIES | DOMAIN_FOOD | DOMAIN_BOOKS
        user_local_idx  : 0-based index within that domain's user set
        n               : number of candidates to return

        Returns
        -------
        List of dicts: {name, tags, score}
        """
        meta = self.domain_meta[domain_id]
        user_offset = meta["user_offset"]
        item_offset = meta["item_offset"]
        n_items     = meta["n_items"]

        user_global  = user_offset + user_local_idx
        items_global = torch.arange(item_offset, item_offset + n_items)
        users        = torch.full((n_items,), user_global,  dtype=torch.long)
        domains      = torch.full((n_items,), domain_id,    dtype=torch.long)

        with torch.no_grad():
            scores = self.model(users, items_global, domains).numpy()

        top_local_idxs = np.argsort(scores)[::-1][:n]

        results = []
        df, name_col, tag_col = self.lookups[domain_id]
        for local_idx in top_local_idxs:
            processed_idx = self.idx2processed[domain_id].get(int(local_idx))
            row = df.loc[processed_idx] if processed_idx in df.index else None
            results.append({
                "name":  row[name_col] if row is not None else f"Item {local_idx}",
                "tags":  row[tag_col]  if (row is not None and tag_col) else "",
                "score": float(scores[local_idx]),
            })
        return results
