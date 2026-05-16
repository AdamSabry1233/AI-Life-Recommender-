---
title: AI Life Recommender
emoji: 🎬🍜📚✨
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
short_description: Multi-domain AI life recommender — movies, food & books
---

# AI Life Recommender

A multi-agent AI system that recommends a **movie**, **meal**, **book**, and **daily habit** tailored to how you're feeling. Built end-to-end across 8 phases — from raw data processing and model training through a full LangGraph multi-agent pipeline, persistent memory, and cloud deployment.

**Team:** Adam Sabry · Rammy Baroudi · Stephen Taylor
**Course:** CMPE 256 — Recommender Systems — San José State University

---

## Live Demo

Deployed on Hugging Face Spaces: [asabry1233/ai-recommender](https://huggingface.co/spaces/asabry1233/ai-recommender)

---

## How It Works

1. **Onboarding** — New users answer 4 quick questions (genres, dietary restrictions, book preferences, lifestyle). Profile is saved to Upstash Redis so returning users skip onboarding entirely.
2. **Intent** — User describes how they're feeling in natural language.
3. **Multi-Agent Pipeline** — Four specialized agents run sequentially via LangGraph, each picking one recommendation from MF-ranked candidates using the Groq LLM.
4. **Multi-Turn Chat** — Follow-up messages are classified by the LLM as a question, refinement, concern, or new intent — each handled differently without losing context.
5. **Memory** — Sessions are stored in FAISS so the system can recall past recommendations across the conversation.

---

## Architecture

```
User message
     │
     ▼
retrieve_memory (FAISS)
     │
     ▼
entertainment_agent ──► food_agent ──► learning_agent ──► habit_agent
     │                      │                │                  │
  Movie rec             Food rec          Book rec          Habit rec
     └──────────────────────┴────────────────┴──────────────────┘
                                   │
                             orchestrator
                                   │
                            Final response
```

Each agent:
1. Pulls 30 candidates from the Matrix Factorization model
2. Keeps the top 3 (quality floor), shuffles the rest for variety
3. Passes 10 candidates + user profile + intent to the LLM
4. Returns a single named recommendation with a reason

---

## Tech Stack

| Component | Technology |
|---|---|
| Recommendation model | Multi-Domain Matrix Factorization (PyTorch) |
| Agent orchestration | LangGraph (StateGraph) |
| LLM (cloud) | Groq API — Llama 3 8B |
| LLM (local) | Ollama — Llama 3.1 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector memory | FAISS (IndexFlatIP) |
| Persistent profiles | Upstash Redis |
| Backend | FastAPI (local Docker) |
| Frontend | Gradio |
| Deployment | Hugging Face Spaces |
| Local dev | Docker Compose |

---

## Datasets

All datasets are too large for GitHub. Download them from the links below and place them under `data/raw/` before running preprocessing.

| Domain | Source | Size | Download |
|---|---|---|---|
| Movies | MovieLens 25M | ~25M ratings, 62K movies | [grouplens.org/datasets/movielens/25m](https://grouplens.org/datasets/movielens/25m/) |
| Food | Food.com Recipes & Interactions | ~1M interactions | [Kaggle — Food.com Recipes](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) |
| Books | Amazon Books Reviews | ~3M ratings | [Kaggle — Amazon Books](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) |

Expected directory structure after download:
```
data/
└── raw/
    ├── ml-25m/
    │   └── ratings.csv
    ├── food_recipes/
    │   └── RAW_interactions.csv
    └── amazon_books/
        └── Books_rating.csv
```

The MF model checkpoint (`mf_best.pt`, 1.5GB) is hosted separately on HF Hub and downloaded automatically at startup:
[asabry1233/ai-recommender-mf](https://huggingface.co/asabry1233/ai-recommender-mf)

---

## How to Run

### Option A — Docker (Recommended)

**Requirements:** Docker, Docker Compose, Ollama

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/AdamSabry1233/AI-Life-Recommender-.git
cd AI-Life-Recommender-
```

**Step 2 — Pull Llama 3.1 via Ollama:**
```bash
ollama pull llama3.1
```

**Step 3 — Set environment variables:**

Create a `.env` file in the project root:
```
OLLAMA_MODEL=llama3.1
INFERENCE_BACKEND=
```

**Step 4 — Download datasets** (see Datasets section above) and place under `data/raw/`

**Step 5 — Run preprocessing to generate lookup CSVs:**
```bash
docker-compose run dev python preprocess.py --raw data/raw --out data/processed --no-plots
```

**Step 6 — Start the full stack:**
```bash
docker-compose up api ui
```

**Step 7 — Open the UI:**
```
http://localhost:7860
```

**Terminal chat interface (optional):**
```bash
docker-compose exec dev python src/chat.py
```

**Run the graph directly (optional):**
```bash
docker-compose exec dev python src/graph.py --intent "I want to relax but be productive"
```

---

### Option B — Local Python (No Docker)

**Requirements:** Python 3.11+, Ollama

**Step 1 — Clone the repo:**
```bash
git clone https://github.com/AdamSabry1233/AI-Life-Recommender-.git
cd AI-Life-Recommender-
```

**Step 2 — Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Pull Llama 3.1 via Ollama:**
```bash
ollama pull llama3.1
ollama serve
```

**Step 5 — Download datasets** (see Datasets section above) and place under `data/raw/`

**Step 6 — Run preprocessing:**
```bash
python preprocess.py --raw data/raw --out data/processed --no-plots
```

**Step 7 — Run the terminal chat:**
```bash
python src/chat.py
```

**Or run the graph directly:**
```bash
python src/graph.py --intent "I want to relax but be productive"
```

---

### Train the MF Model (Optional)

If you want to retrain instead of using the hosted checkpoint:

```bash
python train_mf.py \
  --data data/processed/movies.csv data/processed/food.csv data/processed/books.csv \
  --domain 0 1 2 \
  --epochs 20 \
  --batch 4096
```

The best checkpoint saves automatically to `checkpoints/mf_best.pt`.

---

## Matrix Factorization Model

A custom multi-domain MF model trained jointly across three domains — movies, food, and books — using a shared user embedding space with domain-aware context vectors.

**Architecture:**
```
score(user, item, domain) = (user_emb + domain_emb) · item_emb
                           + user_bias + item_bias + global_bias
```

**Training:**
- Optimizer: Adam with learning rate scheduling
- Loss: MSE on explicit ratings
- Best checkpoint saved as `mf_best.pt` (hosted on HF Hub)

---

## Multi-Agent Pipeline (LangGraph)

Four domain agents run sequentially, each operating on a shared `AgentState` TypedDict:

| Agent | Domain | Data source |
|---|---|---|
| `entertainment_agent` | Movies | MovieLens via MF |
| `food_agent` | Food & recipes | Food.com via MF |
| `learning_agent` | Books | Amazon Books via MF |
| `habit_agent` | Daily habits | LLM only (no MF) |

After all four agents run, an `orchestrator` node synthesizes a single cohesive summary.

---

## Multi-Turn Chat

After the initial recommendation set, follow-up messages are classified by the LLM into one of four types:

| Classification | What happens |
|---|---|
| `question` | Answers using current recommendations as context |
| `refinement` | Replaces one recommendation, filtered from 40 shuffled candidates |
| `concern` | Acknowledges dietary/religious/ethical concern, replaces with restriction-aware pick |
| `new_intent` | Saves current session to FAISS, runs full pipeline fresh |

---

## Memory (FAISS)

Each completed session is embedded and stored in a FAISS `IndexFlatIP` (cosine similarity via normalized inner product). When a user asks a cross-session question ("what did you recommend last time?"), the system searches past sessions and answers from the stored context.

- Embedder: `all-MiniLM-L6-v2` (384-dim) on HF Spaces, `OllamaEmbeddings` locally
- Store: `data/memory/faiss.index` + `metadata.json`

---

## Persistent User Profiles (Upstash Redis)

User profiles are stored in Upstash Redis keyed by a browser-generated UUID (`gr.BrowserState`). This means:

- Returning users on the same browser skip onboarding entirely
- Their preferences (dietary, genres, book type, lifestyle) are injected into every agent prompt
- TTL: 90 days

---

## Project Structure

```
AI_Recomennder/
├── app.py                      # HF Spaces entry point (Gradio + direct engine)
├── preprocess.py               # Multi-domain preprocessing script
├── train_mf.py                 # PyTorch MF training script
├── requirements.txt
├── data/
│   ├── raw/                    # Raw datasets (not committed — see Datasets section)
│   └── processed/
│       ├── movie_lookup.csv    # item_idx → title, genre
│       ├── food_lookup.csv     # item_idx → name, tags
│       └── book_lookup.csv     # item_idx → title
└── src/
    ├── engine.py               # ChatEngine + ChatSession (shared by all frontends)
    ├── graph.py                # LangGraph pipeline builder
    ├── recommender.py          # MF inference wrapper
    ├── llm_chain.py            # Prompt utilities
    ├── user_store.py           # Upstash Redis profile persistence
    ├── api.py                  # FastAPI backend (local Docker)
    ├── ui.py                   # Gradio frontend (local Docker)
    ├── chat.py                 # Terminal chat interface
    ├── agents/
    │   ├── state.py            # AgentState TypedDict
    │   ├── entertainment_agent.py
    │   ├── food_agent.py
    │   ├── learning_agent.py
    │   ├── habit_agent.py
    │   └── orchestrator.py
    └── memory/
        ├── embedder.py         # sentence-transformers / OllamaEmbeddings
        └── history_store.py    # FAISS index + metadata
```

---

## HF Spaces Deployment

**Environment variables (set in HF Space settings):**
```
INFERENCE_BACKEND=groq
GROQ_API_KEY=<your key>
UPSTASH_REDIS_REST_URL=<your url>
UPSTASH_REDIS_REST_TOKEN=<your token>
```

`mf_best.pt` is hosted on [asabry1233/ai-recommender-mf](https://huggingface.co/asabry1233/ai-recommender-mf) and downloaded automatically at startup via `hf_hub_download`.
