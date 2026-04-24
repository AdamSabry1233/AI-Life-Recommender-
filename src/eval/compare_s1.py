"""Compare MF-only vs MF+S1 (semantic retrieval with user-profile anchor).

Single-domain MovieLens 100K demo. Baseline = Cornac MF trained on a random
80/20 time-aware split. +S1 = fused MF + semantic with a mean-pooled user-profile
embedding as the semantic query.

Run:  python -m eval.compare_s1
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import MF

from eval.ml100k import load_items, load_ratings
from eval.metrics import coverage, diversity, ndcg_at_k, novelty, recall_at_k
from semantic.fusion import fuse
from semantic.profile import user_profile
from semantic.retrieval import SemanticRetriever

OUT = Path(__file__).resolve().parents[2] / "results"
OUT.mkdir(exist_ok=True)
K = 10
SEED = 42


def _split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(SEED)
    mask = rng.rand(len(df)) < test_frac
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def run():
    ratings = load_ratings()
    items = load_items()
    train_df, test_df = _split(ratings)

    train_tuples = list(train_df[["user", "item", "rating"]].itertuples(index=False, name=None))
    train_ds = Dataset.from_uir(train_tuples)
    mf = MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=SEED, verbose=False)
    mf.fit(train_ds)

    uid_map, iid_map = train_ds.uid_map, train_ds.iid_map
    inv_iid = {v: k for k, v in iid_map.items()}
    catalog = set(iid_map.keys())

    items_in = items[items["item"].isin(catalog)].reset_index(drop=True)
    ret = SemanticRetriever()
    ret.fit(items_in["item"].tolist(), items_in["text"].tolist())
    item_vec = {int(r.item): ret.vectors[i] for i, r in enumerate(items_in.itertuples())}

    pop = ratings.groupby("item").size().to_dict()
    n_users = int(ratings["user"].nunique())

    # per-user train history (for profile vector) and held-out relevance
    train_by_user = train_df[train_df["rating"] >= 4.0].groupby("user")["item"].apply(list).to_dict()
    rel = test_df[test_df["rating"] >= 4.0].groupby("user")["item"].apply(set).to_dict()

    rows, all_mf, all_s1 = [], [], []
    for u_raw, relevant in rel.items():
        if u_raw not in uid_map:
            continue
        u = uid_map[u_raw]
        # MF scores across the full training catalog
        mf_s = {inv_iid[i]: float(mf.score(u, i)) for i in range(len(iid_map))}
        mf_ranked = [i for i, _ in sorted(mf_s.items(), key=lambda x: -x[1])]

        # user-profile semantic query
        hist = [i for i in train_by_user.get(u_raw, []) if i in item_vec]
        if hist:
            q = user_profile(np.stack([item_vec[i] for i in hist]))
        else:
            q = item_vec[mf_ranked[0]] if mf_ranked else np.zeros(ret.vectors.shape[1], dtype="float32")
        sem_s = dict(ret.search(q, k=len(mf_s)))
        s1_ranked = [i for i, _ in fuse(mf_s, sem_s, alpha=0.5)]

        rows.append({
            "mf_ndcg":   ndcg_at_k(relevant, mf_ranked, K),
            "s1_ndcg":   ndcg_at_k(relevant, s1_ranked, K),
            "mf_recall": recall_at_k(relevant, mf_ranked, K),
            "s1_recall": recall_at_k(relevant, s1_ranked, K),
            "mf_div":    diversity(mf_ranked, item_vec, K),
            "s1_div":    diversity(s1_ranked, item_vec, K),
            "mf_nov":    novelty(mf_ranked, pop, n_users, K),
            "s1_nov":    novelty(s1_ranked, pop, n_users, K),
        })
        all_mf.append(mf_ranked[:K])
        all_s1.append(s1_ranked[:K])

    per_user = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "NDCG@10":     [per_user["mf_ndcg"].mean(),   per_user["s1_ndcg"].mean()],
        "Recall@10":   [per_user["mf_recall"].mean(), per_user["s1_recall"].mean()],
        "Diversity":   [per_user["mf_div"].mean(),    per_user["s1_div"].mean()],
        "Novelty":     [per_user["mf_nov"].mean(),    per_user["s1_nov"].mean()],
        "Coverage":    [coverage(all_mf, catalog),    coverage(all_s1, catalog)],
    }, index=["MF", "MF + S1"])

    summary.round(4).to_csv(OUT / "compare_s1.csv")
    print(summary.round(4).to_string())
    return summary


if __name__ == "__main__":
    run()
