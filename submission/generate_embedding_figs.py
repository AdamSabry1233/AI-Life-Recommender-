"""Three figures making the vector DB and embeddings visible:

  1. embeddings_scatter.png — PCA scatter of MovieLens item embeddings colored by primary genre
  2. retrieval_example.png  — FAISS top-5 for three natural-language queries
  3. vector_db.png          — vector DB architecture diagram

Run from repo root:  python submission/generate_embedding_figs.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import faiss  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from eval.ml100k import GENRES, load_items  # noqa: E402

OUT = REPO / "submission" / "figures"
OUT.mkdir(exist_ok=True)

NAVY = "#1B2A4E"
ORANGE = "#E58A2D"
TEAL = "#4DA8B3"
CREAM = "#F7F4ED"
GREY = "#6B7280"


# ─────────────────────────────────────────── shared embedding pass

def _build():
    """Load MovieLens 100K items, embed text, return (titles, genres, vectors)."""
    df = load_items()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(df["text"].tolist(), normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")
    return df, vectors, model


_PRIMARY_GENRE_ORDER = [
    "Action", "Adventure", "Comedy", "Drama", "Horror",
    "Romance", "SciFi", "Thriller", "Childrens", "Documentary",
]


def _primary_genre(text: str) -> str | None:
    """Pick the first listed canonical genre from an item text field."""
    parts = text.split()
    for g in _PRIMARY_GENRE_ORDER:
        if g in parts:
            return g
    return None


# ─────────────────────────────────────────── 1. PCA scatter

def embedding_scatter(df, vectors):
    pca = PCA(n_components=2, random_state=0)
    pts = pca.fit_transform(vectors)
    primary = df["text"].apply(_primary_genre)

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab10")
    color_for = {g: cmap(i) for i, g in enumerate(_PRIMARY_GENRE_ORDER)}

    # background of "other"
    mask_other = primary.isna()
    ax.scatter(pts[mask_other, 0], pts[mask_other, 1],
               s=10, color="#CCCCCC", alpha=0.35, label="other")

    handles = []
    for g in _PRIMARY_GENRE_ORDER:
        m = primary == g
        if m.sum() == 0:
            continue
        ax.scatter(pts[m, 0], pts[m, 1], s=24, color=color_for[g],
                   alpha=0.75, edgecolor="white", linewidth=0.3, label=g)
        handles.append(mpatches.Patch(color=color_for[g], label=f"{g}  ({int(m.sum())})"))

    ax.set_title("MovieLens 100K — item embeddings (PCA → 2D, colored by primary genre)",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.set_xlabel("PC 1", fontsize=11, color=NAVY)
    ax.set_ylabel("PC 2", fontsize=11, color=NAVY)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95,
              title="primary genre (count)", title_fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "embeddings_scatter.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 2. retrieval example

def retrieval_example(df, vectors, model):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    queries = [
        "high-energy action and adventure for a workout night",
        "thoughtful slow-paced drama for a quiet evening",
        "feel-good comedy that will make me laugh",
    ]
    qvecs = model.encode(queries, normalize_embeddings=True).astype("float32")
    D, I = index.search(qvecs, k=5)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    for ax, q, idx_row, score_row in zip(axes, queries, I, D):
        ax.axis("off")
        # title query block
        ax.add_patch(FancyBboxPatch((0.04, 0.85), 0.92, 0.12,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=NAVY, edgecolor=NAVY,
                                    transform=ax.transAxes))
        ax.text(0.5, 0.91, f"“{q}”",
                ha="center", va="center", fontsize=11, color="white",
                style="italic", transform=ax.transAxes, wrap=True)

        # top-5 results
        for rank, (item_idx, score) in enumerate(zip(idx_row, score_row), start=1):
            title = df.iloc[int(item_idx)]["title"]
            genres = df.iloc[int(item_idx)]["text"].replace(title, "").strip()
            y = 0.78 - (rank - 1) * 0.14
            ax.add_patch(FancyBboxPatch((0.04, y - 0.06), 0.92, 0.12,
                                        boxstyle="round,pad=0.02,rounding_size=0.04",
                                        facecolor=CREAM, edgecolor=TEAL, linewidth=1.0,
                                        transform=ax.transAxes))
            # rank circle
            ax.add_patch(plt.Circle((0.10, y), 0.035, color=ORANGE,
                                    transform=ax.transAxes, zorder=4))
            ax.text(0.10, y, str(rank), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold",
                    transform=ax.transAxes, zorder=5)
            # title
            ax.text(0.17, y + 0.018, title[:48],
                    fontsize=10.5, color=NAVY, fontweight="bold",
                    transform=ax.transAxes)
            # genres
            ax.text(0.17, y - 0.028, genres[:60],
                    fontsize=8.5, color=GREY,
                    transform=ax.transAxes)
            # score
            ax.text(0.92, y, f"{score:.2f}",
                    ha="right", va="center", fontsize=10, color=TEAL,
                    fontweight="bold", transform=ax.transAxes)

    fig.suptitle("FAISS retrieval — natural-language query → top-5 items by cosine similarity",
                 fontsize=13, color=NAVY, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "retrieval_example.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 3. vector-DB architecture

def vector_db_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.set_title("Vector Storage — FAISS as the project's vector database",
                 fontsize=14, color=NAVY, fontweight="bold", pad=14)

    def box(x, y, w, h, text, fc=CREAM, ec=NAVY, fontsize=11, fontcolor=NAVY, bold=False):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.10",
                           linewidth=1.4, edgecolor=ec, facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=fontcolor,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, color=NAVY, lw=1.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=lw))

    # left column — ingestion
    ax.text(2.0, 5.9, "Ingest path (once at startup)",
            ha="center", fontsize=11, color=GREY, fontweight="bold")
    box(0.4, 4.6, 3.2, 0.95,
        "Item metadata text\n(title + genre tags)", fc="white")
    box(0.4, 3.2, 3.2, 0.95,
        "Sentence transformer\nMiniLM-L6-v2 → 384-dim", fc=CREAM)
    box(0.4, 1.8, 3.2, 0.95,
        "L2-normalize\nso IP = cosine", fc=CREAM)
    arrow(2.0, 4.6, 2.0, 4.15)
    arrow(2.0, 3.2, 2.0, 2.75)

    # center — FAISS index
    box(4.5, 2.0, 4.0, 3.3,
        "FAISS\nIndexFlatIP\n\nN items × 384-dim\nfloat32 matrix\n+ ID map\n\ncosine retrieval",
        fc=NAVY, ec=NAVY, fontcolor="white", bold=True, fontsize=13)
    arrow(3.6, 2.3, 4.5, 3.0)

    # right column — query
    ax.text(11.0, 5.9, "Query path (per request)",
            ha="center", fontsize=11, color=GREY, fontweight="bold")
    box(9.4, 4.6, 3.2, 0.95,
        "User intent text\n(mood + onboarding profile)", fc="white")
    box(9.4, 3.2, 3.2, 0.95,
        "Sentence transformer\n(same model, same dim)", fc=CREAM)
    box(9.4, 1.8, 3.2, 0.95,
        "Query vector q ∈ ℝ³⁸⁴", fc=CREAM)
    arrow(11.0, 4.6, 11.0, 4.15)
    arrow(11.0, 3.2, 11.0, 2.75)
    arrow(9.4, 2.3, 8.5, 3.0)

    # output
    box(3.0, 0.3, 7.0, 1.0,
        "Top-K item IDs + similarity scores  ->  agent",
        fc=ORANGE, ec=ORANGE, fontcolor="white", bold=True, fontsize=12)
    arrow(6.5, 2.0, 6.5, 1.3)

    # second FAISS use — session memory
    ax.text(11.0, 1.0, "Same vector DB also stores\npast session embeddings\nfor cross-session recall",
            ha="center", fontsize=9, color=GREY, fontstyle="italic")

    fig.savefig(OUT / "vector_db.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── main

if __name__ == "__main__":
    print("Loading + embedding MovieLens 100K items …")
    df, vectors, model = _build()
    print(f"  ->{len(df)} items, {vectors.shape[1]}-dim vectors")

    print("Rendering embeddings_scatter.png …")
    embedding_scatter(df, vectors)

    print("Rendering retrieval_example.png …")
    retrieval_example(df, vectors, model)

    print("Rendering vector_db.png …")
    vector_db_diagram()

    print(f"Done. Figures in {OUT}")
