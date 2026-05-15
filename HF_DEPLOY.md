# Hugging Face Spaces Deployment

This repository targets two parallel Spaces:

| Space                          | Behavior                       | Branch / commit                      |
|--------------------------------|--------------------------------|--------------------------------------|
| Baseline                       | MF + LangGraph + LLM           | `master` (Adam's space, already live) |
| Augmented (S1 + S2)            | Same + semantic retrieval + cluster conditioning | `angel/compare-all` |

The augmented Space is the same codebase as the baseline plus two toggleable
modules. Both layers default to off — flipping environment variables turns
each one on without code changes.

---

## 1. Create the augmented Space

1. Go to https://huggingface.co/new-space — sign in if needed.
2. Set:
   - **Owner**: your HF username (e.g., `stephentaylor1234`).
   - **Space name**: e.g., `ai-recommender-augmented`.
   - **License**: MIT.
   - **SDK**: Gradio.
   - **Hardware**: CPU basic is sufficient (sentence-transformer inference
     is fast on CPU for short batches).
3. After creation HF gives you a git URL like
   `https://huggingface.co/spaces/<your-handle>/ai-recommender-augmented`.

## 2. Push the augmented branch to that Space

```bash
cd path/to/AI-Life-Recommender-
git checkout angel/compare-all
git remote add hf https://huggingface.co/spaces/<your-handle>/ai-recommender-augmented
git push hf angel/compare-all:main
```

The Space autodetects `app.py` and rebuilds.

## 3. Set Space secrets (Settings → Variables and secrets)

| Key                          | Value                                  | Required for         |
|------------------------------|----------------------------------------|----------------------|
| `INFERENCE_BACKEND`          | `groq`                                 | LLM path             |
| `GROQ_API_KEY`               | (your Groq key)                        | LLM path             |
| `UPSTASH_REDIS_REST_URL`     | (your Upstash URL)                     | profile persistence  |
| `UPSTASH_REDIS_REST_TOKEN`   | (your Upstash token)                   | profile persistence  |
| `HF_MODEL_REPO`              | `asabry1233/ai-recommender-mf`         | MF checkpoint        |
| **`USE_SEMANTIC_RETRIEVAL`** | `1`                                    | **enables S1**       |
| **`USE_CLUSTER_CONDITIONING`** | `1`                                  | **enables S2**       |
| `S1_ALPHA` *(optional)*      | `0.5` — MF vs semantic weight          | tunes S1             |
| `S2_ALPHA` *(optional)*      | `0.5` — MF vs cluster proximity weight | tunes S2             |

Reuse the same `GROQ_API_KEY` and Upstash credentials as Adam's Space if you
want both Spaces to share user profiles. Use separate Upstash credentials if
you want isolated user data.

## 4. Verify

After the Space builds and starts, the log header should print:

```
[features] semantic retrieval (S1): ON
[features] cluster conditioning (S2): ON
```

If either line says `off`, the env var wasn't set or didn't propagate — fix
the secret and restart the Space.

## 5. Demoing baseline vs augmented side by side

Both Spaces expose the same Gradio chat interface and accept the same intents.
For the live presentation, open both URLs in adjacent browser windows, paste
the same intent into both ("I'm tired but want to feel accomplished tonight"
is a good test prompt), and show how the augmented version returns a
different recommendation package because S1 has re-ranked the MF candidates
by semantic similarity to the intent + onboarding profile, and S2 has
re-ranked again by proximity to the active onboarding facet.

## How the live hooks fit into the agents

Each MF-backed agent (`entertainment`, `food`, `learning`) does the following
when both flags are on:

```
candidates = MF top-30
if S1: candidates = fuse(MF score, sentence-transformer similarity to query)
if S2: candidates = re-rank by proximity to active onboarding-facet centroid
top-3 kept, tail shuffled, 10 sent to the LLM for final selection
```

The habit agent is unaffected because it has no MF backing.

`src/semantic/live.py` and `src/clusters/live.py` are the runtime hooks.
`src/semantic/` and `src/clusters/` contain the offline-evaluation versions
used by the comparison harness in `src/eval/compare_*.py`.

## Tuning weights at runtime

Both layers expose an alpha that mixes MF vs the augmented signal. Defaults
are 0.5 each. Lower α favors the new signal; higher α favors MF. Set via
`S1_ALPHA` and `S2_ALPHA` Space secrets.
