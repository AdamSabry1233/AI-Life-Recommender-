"""Generate all diagrams and equation images for the v2 deck and report."""
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

NAVY   = "#1B2A4E"
ORANGE = "#E58A2D"
TEAL   = "#4DA8B3"
CREAM  = "#F7F4ED"
GREY   = "#6B7280"
GREEN  = "#2CA02C"
SLATE  = "#3C4858"


def _box(ax, x, y, w, h, text, fc=CREAM, ec=NAVY, fontsize=10, fontcolor=NAVY, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor,
            fontweight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=NAVY, style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw))


# ─────────────────────────────────────────── 1. Full system architecture

def architecture():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.5); ax.axis("off")

    _box(ax, 0.3, 6.3, 2.6, 0.9, "User intent\n(mood, query, context)", fc="#FFF", bold=True, fontsize=11)
    _box(ax, 3.5, 6.3, 2.2, 0.9, "Upstash Redis\nuser profile",  fc=CREAM, fontsize=10)
    _box(ax, 6.3, 6.3, 2.2, 0.9, "FAISS\nsession memory",        fc=CREAM, fontsize=10)
    _box(ax, 9.1, 6.3, 3.6, 0.9, "Multi-turn classifier\n(question/refinement/concern/new)", fc=CREAM, fontsize=10)

    _arrow(ax, 1.6, 6.3, 1.6, 5.6)
    _arrow(ax, 4.6, 6.3, 4.6, 5.6)
    _arrow(ax, 7.4, 6.3, 7.4, 5.6)

    # LangGraph banner
    _box(ax, 0.3, 4.7, 12.4, 0.7, "LangGraph StateGraph — orchestrates the five-node pipeline",
         fc=NAVY, fontcolor="white", bold=True, fontsize=12)

    # Joint MF backbone bar
    _box(ax, 0.3, 3.6, 12.4, 0.8,
         "Joint multi-domain matrix factorization   ·   score(user, item, d) = (user_emb + domain_emb)·item_emb + biases",
         fc="#E9EEF8", fontsize=11)

    # Four agents
    cols = ["Entertainment", "Food", "Learning", "Habit\n(LLM only)"]
    domains = ["MovieLens 25M", "Food.com 1M", "Amazon Books 3M", "no public data"]
    xs = [0.5, 3.6, 6.7, 9.8]
    for x, name, dom in zip(xs, cols, domains):
        _box(ax, x, 2.0, 2.8, 1.2, f"{name}\n\n{dom}", fc=CREAM, fontsize=10, bold=True)
        _arrow(ax, x + 1.4, 3.6, x + 1.4, 3.2)

    # Orchestrator
    _box(ax, 2.5, 0.6, 8.0, 0.9, "Orchestrator — cross-domain coherence synthesis",
         fc=NAVY, fontcolor="white", bold=True, fontsize=11)
    for x in xs:
        _arrow(ax, x + 1.4, 2.0, 6.5, 1.5)

    # Output
    _box(ax, 2.5, -0.6, 8.0, 0.9, "Movie · Meal · Book · Habit  +  natural-language summary",
         fc=ORANGE, fontcolor="white", bold=True, fontsize=11)
    _arrow(ax, 6.5, 0.6, 6.5, 0.3)

    ax.set_ylim(-0.7, 7.5)
    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 2. S1 mechanism

def s1_mechanism():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.set_title("S1 — Semantic Cross-Domain Retrieval with User-Profile Anchor",
                 fontsize=14, color=NAVY, pad=14, fontweight="bold")

    # left column: item catalog → encoder → index
    _box(ax, 0.3, 4.9, 3.0, 1.0,
         "Item catalog\n(title + tags + description)\nacross all 3 domains", fc=CREAM, fontsize=10)
    _box(ax, 0.3, 3.3, 3.0, 1.0, "Sentence-transformer\nMiniLM-L6-v2 (384-dim)", fc="#FFF", fontsize=10)
    _box(ax, 0.3, 1.6, 3.0, 1.0, "FAISS IndexFlatIP\n(cosine retrieval)", fc=CREAM, fontsize=10)
    _arrow(ax, 1.8, 4.9, 1.8, 4.3); _arrow(ax, 1.8, 3.3, 1.8, 2.6)

    # right column: user history → embed → profile vector
    _box(ax, 9.7, 4.9, 3.0, 1.0, "User's rated items\n(positive interactions)", fc=CREAM, fontsize=10)
    _box(ax, 9.7, 3.3, 3.0, 1.0, "Embed each\n(same encoder)", fc="#FFF", fontsize=10)
    _box(ax, 9.7, 1.6, 3.0, 1.0, "Mean pool, L2-normalize\n= user profile vector q", fc=CREAM, fontsize=10)
    _arrow(ax, 11.2, 4.9, 11.2, 4.3); _arrow(ax, 11.2, 3.3, 11.2, 2.6)

    # center: query → semantic top-K
    _arrow(ax, 9.7, 2.1, 3.3, 2.1, color=TEAL, lw=2.0)
    ax.text(6.5, 2.4, "query q", fontsize=11, color=TEAL, ha="center", fontweight="bold")

    # fusion
    _box(ax, 2.5, 0.1, 8.0, 1.0,
         "Fused ranking  =  α · norm(MF score) + (1 − α) · norm(semantic similarity)",
         fc=NAVY, fontcolor="white", bold=True, fontsize=10)
    _arrow(ax, 1.8, 1.6, 3.0, 0.9)
    ax.text(2.4, 1.2, "MF top-K", fontsize=9, color=NAVY)
    _arrow(ax, 11.2, 1.6, 10.0, 0.9)
    ax.text(10.4, 1.2, "semantic top-K", fontsize=9, color=NAVY)

    fig.savefig(OUT / "s1_mechanism.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 3. S2 mechanism

def s2_mechanism():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.set_title("S2 — Per-User Multi-Cluster Taste Model",
                 fontsize=14, color=NAVY, pad=14, fontweight="bold")

    # Left side — fit clusters
    _box(ax, 0.3, 5.0, 3.0, 1.0, "User's positively\nrated items (history)", fc=CREAM, fontsize=10)
    _box(ax, 0.3, 3.4, 3.0, 1.0, "Embed via\nsentence-transformer", fc="#FFF", fontsize=10)
    _box(ax, 0.3, 1.8, 3.0, 1.0, "KMeans (K = 3)\n→ taste centroids", fc=CREAM, fontsize=10)
    _arrow(ax, 1.8, 5.0, 1.8, 4.4); _arrow(ax, 1.8, 3.4, 1.8, 2.8)

    # Center — synthetic cluster scatter
    rng = np.random.RandomState(2)
    centers = np.array([[5.5, 4.5], [6.8, 3.0], [4.4, 2.7]])
    colors = [TEAL, ORANGE, NAVY]
    for c, col in zip(centers, colors):
        pts = rng.randn(20, 2) * 0.25 + c
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=col, alpha=0.55)
        ax.scatter(*c, s=130, color=col, edgecolor="white", linewidth=1.8, zorder=5)
    ax.text(5.6, 5.4, "Cluster 1", color=TEAL, fontsize=10, ha="center", fontweight="bold")
    ax.text(6.9, 2.3, "Cluster 2", color=ORANGE, fontsize=10, ha="center", fontweight="bold")
    ax.text(4.3, 2.0, "Cluster 3", color=NAVY, fontsize=10, ha="center", fontweight="bold")

    # current context dot
    ax.scatter(5.7, 4.4, s=140, marker="*", color=GREEN, zorder=6, edgecolor="white", linewidth=1.5)
    ax.text(5.9, 4.6, "current\ncontext", color=GREEN, fontsize=9, fontweight="bold")

    # Right side — posterior bars
    _box(ax, 9.7, 4.9, 3.0, 1.0, "Cosine sim\nto each centroid", fc="#FFF", fontsize=10)
    _box(ax, 9.7, 3.3, 3.0, 1.0, "Softmax → posterior\nover active cluster", fc=CREAM, fontsize=10)
    _arrow(ax, 11.2, 4.9, 11.2, 4.3)

    # bars
    barx = [10.0, 10.6, 11.2, 11.8]
    barv = [0.72, 0.20, 0.08, 0]
    for i in range(3):
        ax.add_patch(plt.Rectangle((barx[i], 1.6), 0.35, barv[i] * 1.5, color=colors[i], alpha=0.85))
    ax.text(11.0, 1.3, "p(cluster | context)", fontsize=9, color=NAVY, ha="center")
    ax.add_patch(plt.Rectangle((9.7, 1.4), 3.0, 1.4, fill=False, edgecolor=NAVY, lw=1.0))

    # Bottom — conditioning
    _box(ax, 1.5, 0.0, 10.0, 0.9,
         "Re-rank MF candidates by proximity to the active centroid (winning cluster)",
         fc=NAVY, fontcolor="white", bold=True, fontsize=11)
    _arrow(ax, 1.8, 1.8, 2.5, 0.7)
    _arrow(ax, 11.2, 1.6, 10.5, 0.7)

    fig.savefig(OUT / "s2_mechanism.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 4. Trade-off scatter

def tradeoff():
    cfg = ["MF", "MF + S1", "MF + S2", "MF + S1 + S2"]
    ndcg     = [0.0548, 0.0654, 0.0483, 0.0375]
    coverage = [0.0669, 0.1538, 0.2900, 0.3495]
    colors = [GREY, ORANGE, TEAL, NAVY]

    fig, ax = plt.subplots(figsize=(10, 6))
    for c, n, cov, col in zip(cfg, ndcg, coverage, colors):
        ax.scatter(cov, n, s=420, color=col, edgecolor="white", linewidth=2.0, zorder=4, label=c)
    for c, n, cov in zip(cfg, ndcg, coverage):
        ax.annotate(c, (cov, n), textcoords="offset points",
                    xytext=(12, 10), fontsize=12, color=NAVY, fontweight="bold")

    ax.set_xlabel("Coverage  (fraction of catalog surfaced)", fontsize=12, color=NAVY)
    ax.set_ylabel("NDCG@10  (ranking accuracy)", fontsize=12, color=NAVY)
    ax.set_title("Accuracy ↔ Exploration Trade-off — MovieLens 100K",
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(0.04, 0.40); ax.set_ylim(0.030, 0.075)

    ax.annotate("", xy=(0.36, 0.038), xytext=(0.08, 0.066),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=2, alpha=0.5))
    ax.text(0.20, 0.054, "more exploration\nless accuracy",
            color=GREY, fontsize=11, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 5. Contribution module map

def contributions():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("Module Map — Authorship by Contributor",
                 fontsize=14, color=NAVY, fontweight="bold", pad=12)

    ADAM = "#4DA8B3"
    STEPHEN = "#E58A2D"
    SHARED = "#B7B7B7"

    items = [
        # (x, y, w, h, text, color, author)
        (0.5, 5.6, 3.0, 0.9, "src/recommender.py\nJoint multi-domain MF",       ADAM, "Adam"),
        (0.5, 4.5, 3.0, 0.9, "src/train_mf.py\nMF training loop",                ADAM, "Adam"),
        (0.5, 3.4, 3.0, 0.9, "src/agents/\n4 LangGraph nodes",                   ADAM, "Adam"),
        (0.5, 2.3, 3.0, 0.9, "src/graph.py + engine.py\nStateGraph + classifier", ADAM, "Adam"),
        (0.5, 1.2, 3.0, 0.9, "src/api.py + ui.py + chat.py\nFastAPI / Gradio / CLI", ADAM, "Adam"),
        (0.5, 0.1, 3.0, 0.9, "src/memory/ + user_store.py\nFAISS + Redis",       ADAM, "Adam"),

        (5.0, 5.6, 3.0, 0.9, "src/baselines/cornac_eval.py\nMF/PMF/NMF/BPR sweep",  STEPHEN, "Stephen"),
        (5.0, 4.5, 3.0, 0.9, "src/baselines/visualize.py\nGrouped-metric chart",     STEPHEN, "Stephen"),
        (5.0, 3.4, 3.0, 0.9, "src/semantic/\nS1 — content retrieval",                STEPHEN, "Stephen"),
        (5.0, 2.3, 3.0, 0.9, "src/clusters/\nS2 — taste clusters",                   STEPHEN, "Stephen"),
        (5.0, 1.2, 3.0, 0.9, "src/eval/\nComparison harness + metrics",              STEPHEN, "Stephen"),
        (5.0, 0.1, 3.0, 0.9, "tests/ + tools/test_runner.py\nSuite + delegator",     STEPHEN, "Stephen"),

        (9.5, 5.6, 3.0, 0.9, "Datasets across 3 domains\nMovieLens + Food + Books",  SHARED, "Rammy"),
        (9.5, 4.5, 3.0, 0.9, "Train/test split methodology\nSparsity & EDA",          SHARED, "Rammy"),
        (9.5, 3.4, 3.0, 0.9, "Report — Data & Lessons sections\nWriting + figures",    SHARED, "Rammy"),
        (9.5, 2.3, 3.0, 0.9, "Slides — Data + Evaluation\n(visual support)",          SHARED, "Rammy"),
    ]
    for x, y, w, h, text, col, _ in items:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.2, edgecolor=col, facecolor="white")
        ax.add_patch(box)
        ax.add_patch(plt.Rectangle((x, y), 0.18, h, color=col))
        ax.text(x + 0.32, y + h / 2, text, fontsize=10, color=NAVY, va="center")

    # Headers
    for hx, label, col in [(0.5, "Adam Sabry", ADAM),
                           (5.0, "Stephen Taylor", STEPHEN),
                           (9.5, "Rammy Baroudi", SHARED)]:
        ax.text(hx + 1.5, 6.7, label, ha="center", va="center",
                fontsize=13, color="white", fontweight="bold",
                bbox=dict(facecolor=col, edgecolor="none", boxstyle="round,pad=0.4"))

    fig.savefig(OUT / "contributions.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────── 6. Equations (rendered via mathtext)

def equation(text: str, name: str, w=8, h=1.2, fontsize=22):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, color=NAVY)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def equations():
    equation(
        r"$\mathrm{score}(u, i, d) = (\mathbf{u} + \mathbf{d}) \cdot \mathbf{i} + b_u + b_i + b_0$",
        "eq_mf.png", w=10, h=1.4)
    equation(
        r"$\mathrm{fused}(i) = \alpha \cdot \widetilde{s}_{\mathrm{MF}}(i) + (1 - \alpha) \cdot \widetilde{s}_{\mathrm{sem}}(i)$",
        "eq_s1.png", w=10, h=1.4)
    equation(
        r"$P(k \mid \mathrm{context}) = \mathrm{softmax}\!\left( \mathbf{c}_k \cdot \mathbf{x}_{\mathrm{context}} / \tau \right)$",
        "eq_s2_posterior.png", w=10, h=1.4)
    equation(
        r"$H(P) = -\sum_{k=1}^{K} P(k) \log_2 P(k)$",
        "eq_s2_entropy.png", w=8, h=1.4)
    equation(
        r"$\mathbf{q}_{\mathrm{user}} = \mathrm{normalize}\!\left( \frac{1}{|H_u|} \sum_{i \in H_u} \mathbf{e}_i \right)$",
        "eq_user_profile.png", w=10, h=1.6)


if __name__ == "__main__":
    architecture()
    s1_mechanism()
    s2_mechanism()
    tradeoff()
    contributions()
    equations()
    print("Wrote figures to", OUT)
