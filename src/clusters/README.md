# S2 — Per-user multi-cluster taste model with context-activated conditioning

Represents each user as K latent taste clusters rather than a single profile
vector. Clusters are fit via KMeans on the user's rated item embeddings.
At query time a softmax posterior identifies which cluster is active given
the current context; if entropy is high (no cluster dominates) the system
is ambiguous and a clarifying question should be asked.

## Modules

- `cluster_fit.py`   — `fit_user_clusters(item_vectors, k)` returns K centroids via KMeans.
- `posterior.py`     — `posterior(context_vec, centroids)` returns softmax over cosine sim; `entropy(p)` returns Shannon entropy in bits.
- `conditioning.py`  — `condition(mf_scores, item_vectors, active_centroid, alpha)` blends MF scores with proximity to the active cluster.

## Offline comparison

`python -m eval.compare_s2` runs MF-only vs MF+S2 on MovieLens 100K and
writes `results/compare_s2.csv`.

## Agent integration (deferred)

Under `USE_CLUSTER_CONDITIONING=1` the agent would fit clusters on its user's
train history, compute the posterior against the current intent embedding,
and call `condition(...)` on the MF candidate scores before handing to the LLM.
The high-entropy elicitation question is a future addition.
