"""Compare MF-only vs MF+S2 (per-user taste clusters + context-conditioned re-rank).

MovieLens 100K. Each user's train-history item embeddings get clustered into K
centroids via KMeans. At test time the most-recent train item's embedding acts
as the context proxy; the posterior picks the active cluster, and MF scores are
re-ranked by proximity to that centroid.

Run:  python -m eval.compare_s2
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import MF

from clusters.cluster_fit import fit_user_clusters
from clusters.conditioning import condition
from clusters.posterior import posterior
from eval.embed import embed_items
from eval.ml100k import load_items, load_ratings
from eval.metrics import coverage, diversity, ndcg_at_k, novelty, recall_at_k

OUT = Path(__file__).resolve().parents[2] / "results"
OUT.mkdir(exist_ok=True)
K_CLUSTERS = 3
K_TOP = 10
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
    vectors = embed_items(items_in["text"].tolist())
    item_vec = {int(r.item): vectors[i] for i, r in enumerate(items_in.itertuples())}

    pop = ratings.groupby("item").size().to_dict()
    n_users = int(ratings["user"].nunique())

    # chronological train history per user, filtered to positive ratings
    train_sorted = train_df.sort_values("ts")
    train_by_user = (
        train_sorted[train_sorted["rating"] >= 4.0].groupby("user")["item"].apply(list).to_dict()
    )
    rel = test_df[test_df["rating"] >= 4.0].groupby("user")["item"].apply(set).to_dict()

    rows, all_mf, all_s2 = [], [], []
    for u_raw, relevant in rel.items():
        if u_raw not in uid_map:
            continue
        u = uid_map[u_raw]
        mf_s = {inv_iid[i]: float(mf.score(u, i)) for i in range(len(iid_map))}
        mf_ranked = [i for i, _ in sorted(mf_s.items(), key=lambda x: -x[1])]

        hist = [i for i in train_by_user.get(u_raw, []) if i in item_vec]
        if len(hist) < 2:
            s2_ranked = mf_ranked
        else:
            V = np.stack([item_vec[i] for i in hist])
            centroids = fit_user_clusters(V, k=K_CLUSTERS, seed=SEED)
            ctx = item_vec[hist[-1]]  # most recent positively-rated item as context
            p = posterior(ctx, centroids)
            active = centroids[int(np.argmax(p))]
            s2_ranked = [i for i, _ in condition(mf_s, item_vec, active, alpha=0.5)]

        rows.append({
            "mf_ndcg":   ndcg_at_k(relevant, mf_ranked, K_TOP),
            "s2_ndcg":   ndcg_at_k(relevant, s2_ranked, K_TOP),
            "mf_recall": recall_at_k(relevant, mf_ranked, K_TOP),
            "s2_recall": recall_at_k(relevant, s2_ranked, K_TOP),
            "mf_div":    diversity(mf_ranked, item_vec, K_TOP),
            "s2_div":    diversity(s2_ranked, item_vec, K_TOP),
            "mf_nov":    novelty(mf_ranked, pop, n_users, K_TOP),
            "s2_nov":    novelty(s2_ranked, pop, n_users, K_TOP),
        })
        all_mf.append(mf_ranked[:K_TOP])
        all_s2.append(s2_ranked[:K_TOP])

    per_user = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "NDCG@10":   [per_user["mf_ndcg"].mean(),   per_user["s2_ndcg"].mean()],
        "Recall@10": [per_user["mf_recall"].mean(), per_user["s2_recall"].mean()],
        "Diversity": [per_user["mf_div"].mean(),    per_user["s2_div"].mean()],
        "Novelty":   [per_user["mf_nov"].mean(),    per_user["s2_nov"].mean()],
        "Coverage":  [coverage(all_mf, catalog),    coverage(all_s2, catalog)],
    }, index=["MF", "MF + S2"])

    summary.round(4).to_csv(OUT / "compare_s2.csv")
    print(summary.round(4).to_string())
    return summary


if __name__ == "__main__":
    run()
