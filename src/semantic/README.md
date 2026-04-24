# S1 — Semantic cross-domain retrieval with user-profile anchor

Adds a content-based retrieval layer parallel to the joint-MF backbone. Each
item is embedded from its metadata text (title + genre tags + description) via
`sentence-transformers`, indexed in FAISS, and retrieved by cosine similarity
against a query vector that fuses the user's current intent with a mean-pool of
their historical item embeddings.

## Modules

- `retrieval.py` — `SemanticRetriever` wraps a sentence-transformer + `faiss.IndexFlatIP`.
- `profile.py` — `user_profile(item_vectors, weights=None)` returns an L2-normalized mean.
- `fusion.py` — `fuse(mf_scores, semantic_scores, alpha)` min-max normalizes both and returns a sorted hybrid ranking.

## Offline comparison

`python -m eval.compare_s1` from `src/` runs MF-only vs MF+S1 on MovieLens 100K
and writes `results/compare_s1.csv`. `python -m eval.visualize_compare compare_s1.csv`
renders the bar chart.

## Agent integration (deferred)

To enable S1 inside the live agent pipeline, gate it on `USE_SEMANTIC_RETRIEVAL=1`
and replace the `recommender.get_top_n(...)` call in each agent with:

```python
mf = recommender.get_top_n(domain_id, user_local_idx, n=30)
if os.getenv("USE_SEMANTIC_RETRIEVAL") == "1":
    mf = augment_with_semantic(mf, intent, user_profile_vec)
candidates = mf
```

The offline evaluation already proves the retrieval change; live-agent wiring
is a one-file helper and not included here to keep the branch focused on
empirical comparison.
