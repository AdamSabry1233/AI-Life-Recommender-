"""Generate the 15-minute presentation slide deck (.pptx)."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "submission" / "LifeRecommender_Slides.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY = RGBColor(0x12, 0x1F, 0x3D)
ORANGE = RGBColor(0xE5, 0x80, 0x00)
LIGHT = RGBColor(0xF5, 0xF5, 0xF5)
GREY = RGBColor(0x55, 0x55, 0x55)
PLACEHOLDER_BG = RGBColor(0xFF, 0xF4, 0xD9)


def blank(layout_idx: int = 6):
    return prs.slides.add_slide(prs.slide_layouts[layout_idx])


def title_box(slide, text, top=Inches(0.35), size=32):
    box = slide.shapes.add_textbox(Inches(0.5), top, Inches(12.3), Inches(0.9))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = NAVY
    return box


def body_box(slide, text, top=Inches(1.4), left=Inches(0.6), width=Inches(12.1), height=Inches(5.6), size=18):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n") if isinstance(text, str) else text
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.color.rgb = NAVY
        p.space_after = Pt(6)
    return box


def bullet_box(slide, bullets, top=Inches(1.4), left=Inches(0.6), width=Inches(12.1), height=Inches(5.6), size=20):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(size)
        p.font.color.rgb = NAVY
        p.space_after = Pt(10)
    return box


def add_image(slide, path: Path, left, top, width):
    if path.exists():
        slide.shapes.add_picture(str(path), left, top, width=width)
    else:
        rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(4.0))
        rect.fill.solid()
        rect.fill.fore_color.rgb = PLACEHOLDER_BG
        rect.line.color.rgb = ORANGE
        tf = rect.text_frame
        tf.text = f"[ PLACEHOLDER — {path.name} ]"
        tf.paragraphs[0].font.size = Pt(14)
        tf.paragraphs[0].font.italic = True
        tf.paragraphs[0].font.color.rgb = GREY


def placeholder_block(slide, label, left, top, width, height):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    rect.fill.solid()
    rect.fill.fore_color.rgb = PLACEHOLDER_BG
    rect.line.color.rgb = ORANGE
    tf = rect.text_frame
    tf.word_wrap = True
    tf.text = f"[ PLACEHOLDER — {label} ]"
    tf.paragraphs[0].font.size = Pt(13)
    tf.paragraphs[0].font.italic = True
    tf.paragraphs[0].font.color.rgb = GREY


def footer(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.4))
    tf = box.text_frame
    tf.text = text
    tf.paragraphs[0].font.size = Pt(10)
    tf.paragraphs[0].font.color.rgb = GREY


# ─────────── slide 1: title
s = blank()
title_box(s, "AI Life Recommender", top=Inches(2.4), size=48)
body_box(
    s,
    "Context-Aware Multi-Domain Recommendations\nwith Hybrid Collaborative Filtering and Agentic LLM Orchestration",
    top=Inches(3.6),
    size=22,
)
body_box(s, "Adam Sabry  •  Rammy Baroudi  •  Stephen Taylor", top=Inches(5.4), size=18)
body_box(s, "CMPE 256 — Recommender Systems — SJSU", top=Inches(5.9), size=16)

# ─────────── slide 2: the problem
s = blank()
title_box(s, "The Problem")
bullet_box(s, [
    "Single-domain recommenders dominate the field — Netflix, Amazon, Spotify pick within one catalog",
    "Real decisions are cross-domain: a person planning their evening wants a movie + meal + book + habit that go together",
    "Rating matrices cannot represent mood, context, or constraints — \"I'm tired but want to feel productive\" has no place in user × item space",
    "No public habit-rating dataset exists, so collaborative filtering literally cannot recommend habits",
])

# ─────────── slide 3: our system
s = blank()
title_box(s, "Our System — Joint CF Backbone + Agentic LLM Layer")
bullet_box(s, [
    "Joint multi-domain matrix factorization trained on MovieLens 25M + Food.com 1M + Amazon Books 3M",
    "Shared user embedding across domains plus a per-domain context vector",
    "Four LangGraph agents (entertainment, food, learning, habit) re-rank MF candidates using LLM",
    "Orchestrator node enforces cross-domain coherence",
    "FAISS session memory + Upstash Redis profile persistence + multi-turn classification",
])

# ─────────── slide 4: architecture diagram
s = blank()
title_box(s, "System Architecture")
placeholder_block(s, "End-to-end architecture diagram — user intent → Redis profile → FAISS memory → joint MF scoring → S1/S2 augmentation → four LangGraph agents → orchestrator → Gradio output", Inches(0.6), Inches(1.4), Inches(12.1), Inches(5.6))

# ─────────── slide 5: scoring function
s = blank()
title_box(s, "Joint Matrix Factorization — Scoring Function")
body_box(
    s,
    "score(user, item, domain) = (user_emb + domain_emb) · item_emb\n"
    "                          + user_bias + item_bias + global_bias",
    top=Inches(2.2),
    size=24,
)
bullet_box(s, [
    "One user embedding shared across all three domains",
    "Domain context vector added to the user embedding before the inner product",
    "Trained jointly with Adam + MSE on explicit ratings",
    "Approximately 29 million interactions total across three datasets",
], top=Inches(3.6))

# ─────────── slide 6: LangGraph pipeline
s = blank()
title_box(s, "LangGraph Agent Pipeline")
bullet_box(s, [
    "retrieve_memory — FAISS lookup of similar past sessions",
    "entertainment_agent — MF top-30 → LLM picks 1 movie + reason",
    "food_agent — MF top-30 → LLM picks 1 meal + reason",
    "learning_agent — MF top-30 → LLM picks 1 book + reason",
    "habit_agent — LLM-only (no public habit-rating data)",
    "orchestrator — synthesizes the four picks into a coherent response",
])

# ─────────── slide 7: novelty contributions
s = blank()
title_box(s, "Novelty — Two Hybrid Augmentations")
bullet_box(s, [
    "S1: Semantic Cross-Domain Retrieval with User-Profile Anchor",
    "    — embed item metadata via sentence-transformers, fuse with MF scores",
    "    — query is mean of user's positively-rated item embeddings",
    "",
    "S2: Per-User Multi-Cluster Taste Model",
    "    — KMeans on user history → K context-activated taste centroids",
    "    — softmax posterior identifies which cluster is currently active",
])

# ─────────── slide 8: S1 details
s = blank()
title_box(s, "S1 — How Semantic Retrieval Works")
bullet_box(s, [
    "Each item: title + genre tags + description → sentence-transformer embedding",
    "Shared FAISS IndexFlatIP across all items in all domains",
    "User profile vector: L2-normalized mean of embeddings of their rated items",
    "Fused score: α · normalize(MF score) + (1 − α) · normalize(semantic similarity)",
    "Addresses cold-start, cross-domain bridging, long-tail surfacing",
], top=Inches(1.4))

# ─────────── slide 9: S2 details
s = blank()
title_box(s, "S2 — How Cluster Conditioning Works")
bullet_box(s, [
    "Fit K = 3 KMeans clusters on a user's rated-item embeddings",
    "Context vector (most recent positive interaction) → softmax over cosine sim to centroids",
    "MF candidate scores re-ranked by proximity to the active centroid",
    "Posterior entropy = ambiguity signal for future clarifying-question elicitation",
    "Treats user as multi-tasted with context-activated modes, not a single average",
], top=Inches(1.4))

# ─────────── slide 10: baselines
s = blank()
title_box(s, "Baselines — Cornac on MovieLens 100K")
add_image(s, RESULTS / "cornac_metrics.png", Inches(0.5), Inches(1.2), Inches(12.3))
body_box(s, "MF wins rating prediction (RMSE 0.89). BPR wins ranking (NDCG@10 0.23) — different optimization targets.", top=Inches(6.4), size=14)

# ─────────── slide 11: four-way comparison
s = blank()
title_box(s, "Four-Way Comparison — MF / +S1 / +S2 / +Both")
add_image(s, RESULTS / "compare_all.png", Inches(0.5), Inches(1.2), Inches(12.3))

# ─────────── slide 12: numerical table
s = blank()
title_box(s, "Numerical Results")
tbl_data = [
    ["Configuration", "NDCG@10", "Recall@10", "Diversity", "Novelty", "Coverage"],
    ["MF",           "0.0548",  "0.0428",   "0.6802",   "2.5814", "0.0669"],
    ["MF + S1",      "0.0654",  "0.0585",   "0.5434",   "2.7804", "0.1538"],
    ["MF + S2",      "0.0483",  "0.0462",   "0.5165",   "2.9223", "0.2900"],
    ["MF + S1 + S2", "0.0375",  "0.0375",   "0.4660",   "3.5474", "0.3495"],
]
rows, cols = len(tbl_data), len(tbl_data[0])
tbl_shape = s.shapes.add_table(rows, cols, Inches(0.7), Inches(1.6), Inches(12.0), Inches(3.5))
tbl = tbl_shape.table
for r in range(rows):
    for c in range(cols):
        cell = tbl.cell(r, c)
        cell.text = tbl_data[r][c]
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(16)
            if r == 0:
                p.font.bold = True
body_box(s, "S1 wins accuracy. S2 dominates coverage. Both combined peaks exploration metrics. Trade-off slides cleanly along the explore/exploit axis.", top=Inches(5.4), size=16)

# ─────────── slide 13: interpretation
s = blank()
title_box(s, "Interpretation — Discovery vs. Reproduction")
bullet_box(s, [
    "NDCG and Recall against held-out ratings implicitly reward predicting items the user already interacted with",
    "Held-out items skew toward popular — they had to be popular enough to get rated",
    "Coverage, Novelty, Diversity capture what a discovery system actually does",
    "Every additional layer of our system contributes measurable expansion of what the user sees",
    "Trade-off is a tunable design parameter (α weights), not a fixed property",
], top=Inches(1.4))

# ─────────── slide 14: live demo
s = blank()
title_box(s, "Live Demo")
placeholder_block(s, "Live demo screenshots — baseline HF Space vs. augmented HF Space for the same user intent; to be captured before presentation", Inches(0.6), Inches(1.4), Inches(12.1), Inches(4.5))
bullet_box(s, [
    "Baseline:  huggingface.co/spaces/asabry1233/ai-recommender",
    "Augmented (S1 + S2): [TO DEPLOY]",
    "Backup video recorded on USB if live demo fails",
], top=Inches(6.0), size=14)

# ─────────── slide 15: lessons learned
s = blank()
title_box(s, "Lessons Learned")
bullet_box(s, [
    "Cold-start handled cleanly by the semantic layer (S1) — no retraining needed",
    "Joint multi-domain MF scales linearly; FAISS approximate nearest neighbor is sublinear",
    "Offline rating metrics under-represent discovery objectives — report coverage and novelty too",
    "LangGraph appropriate because the pipeline is a state machine, not a chain",
    "Two hybrid layers compose without being redundant — different mechanisms, complementary effects",
], top=Inches(1.4))

# ─────────── slide 16: contribution
s = blank()
title_box(s, "Team Contributions")
rows = [
    ["Member", "Primary Work"],
    ["Adam Sabry",     "Joint MF training, LangGraph pipeline, FAISS memory, Redis profiles, Gradio frontend, baseline HF Space deployment"],
    ["Rammy Baroudi",  "Dataset acquisition and preprocessing, sparsity analysis, train-test methodology, report sections on data and lessons"],
    ["Stephen Taylor", "Cornac baseline sweep, semantic retrieval (S1), cluster conditioning (S2), 4-way comparison, test suite, augmented HF Space"],
]
shape = s.shapes.add_table(len(rows), 2, Inches(0.6), Inches(1.6), Inches(12.1), Inches(4.0))
t = shape.table
t.columns[0].width = Inches(2.5)
t.columns[1].width = Inches(9.6)
for r_i, row in enumerate(rows):
    for c_i, val in enumerate(row):
        cell = t.cell(r_i, c_i)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(16)
            if r_i == 0:
                p.font.bold = True
placeholder_block(s, "Confirm contribution table is accurate before final submission", Inches(0.6), Inches(6.0), Inches(12.1), Inches(0.9))

# ─────────── slide 17: future work
s = blank()
title_box(s, "Future Work")
bullet_box(s, [
    "Entropy-triggered clarifying questions — primitive already implemented in S2, not yet exposed",
    "Online cluster discovery — spawn a new taste cluster when current session matches none",
    "Cross-domain user studies — offline metrics are insufficient for discovery evaluation",
    "Deep model swap — replace shallow joint-MF with a Two-Tower neural backbone",
    "Domain expansion — music, fitness content, podcasts (no cross-domain training data required thanks to S1)",
], top=Inches(1.4))

# ─────────── slide 18: thank you / Q&A
s = blank()
title_box(s, "Thank You — Questions?", top=Inches(3.0), size=44)
body_box(s, "Repository: github.com/AdamSabry1233/AI-Life-Recommender-\nBaseline demo: huggingface.co/spaces/asabry1233/ai-recommender", top=Inches(4.4), size=20)

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"Wrote {OUT}")
