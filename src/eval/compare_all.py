"""Full 4-way comparison: MF, MF+S1, MF+S2, MF+S1+S2 on MovieLens 100K.

Composition for "both" = fuse(MF, semantic) then condition on active cluster.
Run:  python -m eval.compare_all
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
from semantic.fusion import fuse
from semantic.profile import user_profile
from semantic.retrieval import SemanticRetriever

OUT = Path(__file__).resolve().parents[2] / "results"
OUT.mkdir(exist_ok=True)
K_CLUSTERS = 3
K_TOP = 10
SEED = 42


def _split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(SEED)
    mask = rng.rand(len(df)) < test_frac
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def _rank_s1(mf_s, ret, q):
    sem_s = dict(ret.search(q, k=len(mf_s)))
    return [i for i, _ in fuse(mf_s, sem_s, alpha=0.5)]


def _rank_s2(mf_s, item_vec, centroid):
    return [i for i, _ in condition(mf_s, item_vec, centroid, alpha=0.5)]


def _rank_both(mf_s, ret, q, item_vec, centroid):
    sem_s = dict(ret.search(q, k=len(mf_s)))
    fused = dict(fuse(mf_s, sem_s, alpha=0.5))
    return [i for i, _ in condition(fused, item_vec, centroid, alpha=0.5)]


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

    ret = SemanticRetriever()
    ret.fit(items_in["item"].tolist(), items_in["text"].tolist())

    pop = ratings.groupby("item").size().to_dict()
    n_users = int(ratings["user"].nunique())

    train_sorted = train_df.sort_values("ts")
    train_pos = train_sorted[train_sorted["rating"] >= 4.0].groupby("user")["item"].apply(list).to_dict()
    rel = test_df[test_df["rating"] >= 4.0].groupby("user")["item"].apply(set).to_dict()

    configs = ["MF", "MF + S1", "MF + S2", "MF + S1 + S2"]
    metrics = {c: {"ndcg": [], "recall": [], "div": [], "nov": [], "lists": []} for c in configs}

    for u_raw, relevant in rel.items():
        if u_raw not in uid_map:
            continue
        u = uid_map[u_raw]
        mf_s = {inv_iid[i]: float(mf.score(u, i)) for i in range(len(iid_map))}
        mf_ranked = [i for i, _ in sorted(mf_s.items(), key=lambda x: -x[1])]

        hist = [i for i in train_pos.get(u_raw, []) if i in item_vec]

        # user profile for S1
        q = user_profile(np.stack([item_vec[i] for i in hist])) if hist else (
            item_vec[mf_ranked[0]] if mf_ranked else np.zeros(vectors.shape[1], dtype="float32")
        )
        s1 = _rank_s1(mf_s, ret, q)

        # cluster centroid for S2
        if len(hist) >= 2:
            centroids = fit_user_clusters(np.stack([item_vec[i] for i in hist]), k=K_CLUSTERS, seed=SEED)
            active = centroids[int(np.argmax(posterior(item_vec[hist[-1]], centroids)))]
            s2 = _rank_s2(mf_s, item_vec, active)
            both = _rank_both(mf_s, ret, q, item_vec, active)
        else:
            s2, both = mf_ranked, s1

        for name, ranked in zip(configs, [mf_ranked, s1, s2, both]):
            metrics[name]["ndcg"].append(ndcg_at_k(relevant, ranked, K_TOP))
            metrics[name]["recall"].append(recall_at_k(relevant, ranked, K_TOP))
            metrics[name]["div"].append(diversity(ranked, item_vec, K_TOP))
            metrics[name]["nov"].append(novelty(ranked, pop, n_users, K_TOP))
            metrics[name]["lists"].append(ranked[:K_TOP])

    summary = pd.DataFrame({
        "NDCG@10":   [np.mean(metrics[c]["ndcg"])   for c in configs],
        "Recall@10": [np.mean(metrics[c]["recall"]) for c in configs],
        "Diversity": [np.mean(metrics[c]["div"])    for c in configs],
        "Novelty":   [np.mean(metrics[c]["nov"])    for c in configs],
        "Coverage":  [coverage(metrics[c]["lists"], catalog) for c in configs],
    }, index=configs)

    summary.round(4).to_csv(OUT / "compare_all.csv")
    print(summary.round(4).to_string())
    return summary


if __name__ == "__main__":
    run()
