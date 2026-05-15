"""Visual-first slide deck for the 15-minute presentation."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGS = REPO / "submission" / "figures"
OUT = REPO / "submission" / "LifeRecommender_Slides_v2.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── palette
NAVY    = RGBColor(0x1B, 0x2A, 0x4E)
ORANGE  = RGBColor(0xE5, 0x8A, 0x2D)
TEAL    = RGBColor(0x4D, 0xA8, 0xB3)
CREAM   = RGBColor(0xF7, 0xF4, 0xED)
LIGHT   = RGBColor(0xEC, 0xF1, 0xF9)
GREY    = RGBColor(0x55, 0x55, 0x55)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
PLACE   = RGBColor(0xFF, 0xF4, 0xD9)


def slide():
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    return s


def header_bar(s, title, sub=None):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.9))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    tx = s.shapes.add_textbox(Inches(0.5), Inches(0.18), Inches(12.3), Inches(0.6))
    p = tx.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = WHITE
    if sub:
        tx2 = s.shapes.add_textbox(Inches(0.5), Inches(0.95), Inches(12.3), Inches(0.4))
        p = tx2.text_frame.paragraphs[0]
        p.text = sub
        p.font.size = Pt(14); p.font.color.rgb = GREY; p.font.italic = True


def picture(s, path: Path, left, top, width, height=None):
    if path.exists():
        if height is None:
            s.shapes.add_picture(str(path), left, top, width=width)
        else:
            s.shapes.add_picture(str(path), left, top, width=width, height=height)
    else:
        rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height or Inches(4.0))
        rect.fill.solid(); rect.fill.fore_color.rgb = PLACE; rect.line.color.rgb = ORANGE
        rect.text_frame.text = f"[ PLACEHOLDER — {path.name} ]"
        for p in rect.text_frame.paragraphs:
            p.font.size = Pt(13); p.font.italic = True; p.font.color.rgb = GREY


def text(s, txt, left, top, width, height, size=18, color=NAVY, bold=False, align=None, italic=False):
    box = s.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame; tf.word_wrap = True
    lines = txt.split("\n") if isinstance(txt, str) else txt
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size); p.font.color.rgb = color
        p.font.bold = bold; p.font.italic = italic
        if align: p.alignment = align


def card(s, left, top, w, h, title, body, accent=ORANGE, body_size=14):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    box.fill.solid(); box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = accent; box.line.width = Pt(1.8)
    # accent bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.18), h)
    bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
    # title
    th = s.shapes.add_textbox(left + Inches(0.3), top + Inches(0.15), w - Inches(0.4), Inches(0.45))
    p = th.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = NAVY
    # body
    bx = s.shapes.add_textbox(left + Inches(0.3), top + Inches(0.6), w - Inches(0.4), h - Inches(0.7))
    bx.text_frame.word_wrap = True
    lines = body.split("\n") if isinstance(body, str) else body
    for i, line in enumerate(lines):
        p = bx.text_frame.paragraphs[0] if i == 0 else bx.text_frame.add_paragraph()
        p.text = line
        p.font.size = Pt(body_size); p.font.color.rgb = GREY
        p.space_after = Pt(4)


def stat(s, left, top, w, h, value, label, color=NAVY):
    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = color; box.line.width = Pt(1.2)
    vt = s.shapes.add_textbox(left, top + Inches(0.15), w, Inches(0.9))
    p = vt.text_frame.paragraphs[0]
    p.text = value; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = color
    lt = s.shapes.add_textbox(left, top + Inches(1.05), w, Inches(0.5))
    p = lt.text_frame.paragraphs[0]
    p.text = label; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(13); p.font.color.rgb = GREY


def footer(s, page_num, total):
    f = s.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3))
    p = f.text_frame.paragraphs[0]
    p.text = f"AI Life Recommender   ·   CMPE 256   ·   {page_num} / {total}"
    p.font.size = Pt(10); p.font.color.rgb = GREY


def add_table(s, data, left, top, width, height, header_color=NAVY, header_text=WHITE,
              body_size=14, header_size=14, highlight_row=None):
    rows, cols = len(data), len(data[0])
    shape = s.shapes.add_table(rows, cols, left, top, width, height)
    tbl = shape.table
    for r in range(rows):
        for c in range(cols):
            cell = tbl.cell(r, c)
            cell.text = str(data[r][c])
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.06);  cell.margin_bottom = Inches(0.06)
            for p in cell.text_frame.paragraphs:
                if r == 0:
                    p.font.size = Pt(header_size); p.font.bold = True; p.font.color.rgb = header_text
                else:
                    p.font.size = Pt(body_size); p.font.color.rgb = NAVY
                    if highlight_row is not None and r == highlight_row:
                        p.font.bold = True
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_color
            elif r % 2 == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = CREAM
            else:
                cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
    return tbl


# ─────────────────────────────────────────── slide 1 — title

s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.2))
band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()

text(s, "AI Life Recommender", Inches(0.5), Inches(2.0), Inches(12.3), Inches(1.5),
     size=60, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
text(s, "Context-Aware Multi-Domain Recommendations\nwith Hybrid Collaborative Filtering and Agentic LLM Orchestration",
     Inches(0.5), Inches(3.4), Inches(12.3), Inches(1.0),
     size=22, color=GREY, align=PP_ALIGN.CENTER)

# team line
text(s, "Adam Sabry   ·   Rammy Baroudi   ·   Stephen Taylor",
     Inches(0.5), Inches(5.4), Inches(12.3), Inches(0.5),
     size=20, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
text(s, "CMPE 256   ·   Recommender Systems   ·   San José State University",
     Inches(0.5), Inches(6.0), Inches(12.3), Inches(0.4),
     size=14, color=GREY, align=PP_ALIGN.CENTER, italic=True)

# bottom band
band2 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.2), Inches(13.333), Inches(0.3))
band2.fill.solid(); band2.fill.fore_color.rgb = ORANGE; band2.line.fill.background()


# ─────────────────────────────────────────── slide 2 — problem

s = slide()
header_bar(s, "The Problem")
text(s,
     "Production recommenders pick within a single domain — movies, products, or songs — and optimize "
     "for rating prediction. The real decision a person faces is rarely single-domain and almost never "
     "captured by a rating matrix.",
     Inches(0.6), Inches(1.2), Inches(12.1), Inches(1.4), size=18, color=NAVY)

card(s, Inches(0.6),  Inches(2.9), Inches(2.95), Inches(3.7),
     "Single domain",
     "Netflix picks movies.\nAmazon picks products.\nSpotify picks songs.\n\nNobody picks the whole evening.",
     accent=TEAL)
card(s, Inches(3.75), Inches(2.9), Inches(2.95), Inches(3.7),
     "No mood signal",
     "User-item rating matrices have no place for \"tired but want to feel productive.\"\n\nContext is invisible to CF.",
     accent=ORANGE)
card(s, Inches(6.9), Inches(2.9), Inches(2.95), Inches(3.7),
     "No cross-domain coherence",
     "Independent picks per domain produce mismatched packages — a three-hour movie next to a two-hour recipe.",
     accent=NAVY)
card(s, Inches(10.05), Inches(2.9), Inches(2.95), Inches(3.7),
     "Empty domains",
     "No public rating dataset exists for daily habits, so CF cannot recommend them at all.",
     accent=GREY)

footer(s, 2, 22)


# ─────────────────────────────────────────── slide 3 — system architecture

s = slide()
header_bar(s, "System Architecture", "End-to-end pipeline from user intent to coherent recommendation")
picture(s, FIGS / "architecture.png", Inches(0.4), Inches(1.45), Inches(12.5))
footer(s, 3, 22)


# ─────────────────────────────────────────── slide 4 — datasets

s = slide()
header_bar(s, "Datasets — Three Domains, One User Space")

data = [
    ["Domain", "Source", "Interactions", "Items", "Density"],
    ["Entertainment", "MovieLens 25M", "≈ 25,000,000", "62,000",  "≈ 0.25 %"],
    ["Food",          "Food.com Recipes",  "≈ 1,000,000",  "230,000", "< 0.05 %"],
    ["Learning",      "Amazon Books",      "≈ 3,000,000",  "200,000", "≈ 0.015 %"],
    ["Habit",         "— no public data —", "—", "—", "LLM-only domain"],
]
add_table(s, data, Inches(0.6), Inches(1.5), Inches(12.1), Inches(3.0))

text(s,
     "Combined corpus ≈ 29 million interactions. The joint user space is sparser than any single domain "
     "because most users participate in only one — this is the architectural motivation for a shared user "
     "embedding rather than three independent collaborative filters.",
     Inches(0.6), Inches(5.0), Inches(12.1), Inches(1.6), size=15, color=NAVY)

stat(s, Inches(1.5),  Inches(6.2), Inches(2.4), Inches(1.0), "29M",   "interactions",      color=NAVY)
stat(s, Inches(4.5),  Inches(6.2), Inches(2.4), Inches(1.0), "3",     "rating domains",    color=ORANGE)
stat(s, Inches(7.5),  Inches(6.2), Inches(2.4), Inches(1.0), "1",     "shared user space", color=TEAL)
stat(s, Inches(10.5), Inches(6.2), Inches(2.4), Inches(1.0), "+1",    "LLM-only domain",   color=GREY)

footer(s, 4, 22)


# ─────────────────────────────────────────── slide 5 — joint MF backbone

s = slide()
header_bar(s, "Joint Multi-Domain Matrix Factorization", "Scoring backbone — shared user embedding plus per-domain context")

picture(s, FIGS / "eq_mf.png", Inches(1.6), Inches(2.0), Inches(10.0))

card(s, Inches(0.6), Inches(4.2), Inches(4.0), Inches(2.7),
     "Why joint",
     "Most users participate in only one domain.\n\nA shared user embedding lets collaborative signal in one domain inform predictions in others.",
     accent=TEAL, body_size=14)
card(s, Inches(4.85), Inches(4.2), Inches(4.0), Inches(2.7),
     "Why domain vector",
     "Adding domain_emb to the user embedding lets a single model express that the same user has different latent profiles per domain.",
     accent=ORANGE, body_size=14)
card(s, Inches(9.1), Inches(4.2), Inches(3.6), Inches(2.7),
     "Training",
     "Adam optimizer.\nMSE on explicit ratings.\nTime-split 80/20.\n29M interactions.",
     accent=NAVY, body_size=14)

footer(s, 5, 22)


# ─────────────────────────────────────────── slide 6 — LangGraph pipeline

s = slide()
header_bar(s, "LangGraph Pipeline — Five Sequential Nodes")

# Five-node horizontal flow
nodes = [
    ("retrieve_memory",        "FAISS lookup\nof past sessions"),
    ("entertainment_agent",    "MF top-30 →\nLLM picks 1 movie"),
    ("food_agent",             "MF top-30 →\nLLM picks 1 meal"),
    ("learning_agent",         "MF top-30 →\nLLM picks 1 book"),
    ("habit_agent",            "LLM-only\n(no MF data)"),
]
xs = [0.5, 3.0, 5.5, 8.0, 10.5]
top_n = Inches(2.0)
for i, (title, body) in enumerate(nodes):
    left = Inches(xs[i])
    w = Inches(2.4); h = Inches(2.0)
    rect = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top_n, w, h)
    rect.fill.solid(); rect.fill.fore_color.rgb = WHITE
    rect.line.color.rgb = NAVY; rect.line.width = Pt(1.5)
    text(s, title, left, top_n + Inches(0.15), w, Inches(0.5), size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    text(s, body,  left, top_n + Inches(0.75), w, Inches(1.2), size=12, color=GREY, align=PP_ALIGN.CENTER)

# arrows between nodes
for i in range(4):
    line = s.shapes.add_connector(1, Inches(xs[i] + 2.4), Inches(2.9 + 0.1),
                                  Inches(xs[i + 1]), Inches(2.9 + 0.1))
    line.line.color.rgb = NAVY; line.line.width = Pt(1.6)

# orchestrator
orch = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.0), Inches(5.0), Inches(7.5), Inches(0.9))
orch.fill.solid(); orch.fill.fore_color.rgb = NAVY; orch.line.fill.background()
text(s, "Orchestrator — synthesizes the four picks into a coherent response",
     Inches(3.0), Inches(5.18), Inches(7.5), Inches(0.6), size=16, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

# arrows feeding orchestrator
for i in range(5):
    line = s.shapes.add_connector(1, Inches(xs[i] + 1.2), Inches(4.0),
                                  Inches(6.75), Inches(5.0))
    line.line.color.rgb = NAVY; line.line.width = Pt(0.9)

# final output
out_box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.0), Inches(6.2), Inches(7.5), Inches(0.7))
out_box.fill.solid(); out_box.fill.fore_color.rgb = ORANGE; out_box.line.fill.background()
text(s, "Movie  ·  Meal  ·  Book  ·  Habit   +   natural-language summary",
     Inches(3.0), Inches(6.32), Inches(7.5), Inches(0.5), size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)

footer(s, 6, 22)


# ─────────────────────────────────────────── slide 7 — novelty intro

s = slide()
header_bar(s, "Two Hybrid Contributions on Top of the CF Backbone",
           "Each layer is independently toggleable and addresses a distinct weakness of pure CF")

card(s, Inches(0.7), Inches(1.6), Inches(5.9), Inches(5.4),
     "S1 — Semantic Cross-Domain Retrieval",
     ["Encode item metadata via sentence-transformers.",
      "Build a single FAISS index over all three domains.",
      "Query = mean of the user's positively-rated item embeddings.",
      "Fuse MF score with semantic similarity via tunable weight α.",
      "",
      "Addresses cold-start, cross-domain bridging, long-tail surfacing."],
     accent=ORANGE, body_size=15)

card(s, Inches(6.8), Inches(1.6), Inches(5.9), Inches(5.4),
     "S2 — Per-User Multi-Cluster Taste Model",
     ["KMeans on each user's rated-item embeddings → K taste centroids.",
      "Softmax over cosine(current context, centroid) → active cluster.",
      "Re-rank MF candidates by proximity to the active centroid.",
      "Posterior entropy is the ambiguity signal for elicitation.",
      "",
      "Treats user as multi-tasted with context-activated modes."],
     accent=TEAL, body_size=15)

footer(s, 7, 22)


# ─────────────────────────────────────────── slide 8 — S1 mechanism

s = slide()
header_bar(s, "S1 — Mechanism")
picture(s, FIGS / "s1_mechanism.png", Inches(0.4), Inches(1.3), Inches(12.5))
footer(s, 8, 22)


# ─────────────────────────────────────────── slide 9 — S1 equations

s = slide()
header_bar(s, "S1 — Key Equations")

text(s, "User profile vector (mean-pooled, L2-normalized)",
     Inches(1.0), Inches(1.4), Inches(11.0), Inches(0.4), size=16, color=NAVY, bold=True)
picture(s, FIGS / "eq_user_profile.png", Inches(2.0), Inches(1.8), Inches(9.0))

text(s, "Fused ranking — min-max normalized weighted sum",
     Inches(1.0), Inches(4.0), Inches(11.0), Inches(0.4), size=16, color=NAVY, bold=True)
picture(s, FIGS / "eq_s1.png", Inches(2.0), Inches(4.4), Inches(9.0))

text(s, "α tunes the explore/exploit trade-off — closer to 1 favors collaborative accuracy; closer to 0 favors content-based discovery.",
     Inches(1.0), Inches(6.4), Inches(11.0), Inches(0.6), size=13, color=GREY, align=PP_ALIGN.CENTER, italic=True)

footer(s, 9, 22)


# ─────────────────────────────────────────── slide 10 — S2 mechanism

s = slide()
header_bar(s, "S2 — Mechanism")
picture(s, FIGS / "s2_mechanism.png", Inches(0.4), Inches(1.3), Inches(12.5))
footer(s, 10, 22)


# ─────────────────────────────────────────── slide 11 — S2 equations

s = slide()
header_bar(s, "S2 — Key Equations")

text(s, "Cluster posterior — softmax over cosine similarity to centroids",
     Inches(1.0), Inches(1.4), Inches(11.0), Inches(0.4), size=16, color=NAVY, bold=True)
picture(s, FIGS / "eq_s2_posterior.png", Inches(2.0), Inches(1.8), Inches(9.0))

text(s, "Posterior entropy — ambiguity signal for elicitation",
     Inches(1.0), Inches(4.0), Inches(11.0), Inches(0.4), size=16, color=NAVY, bold=True)
picture(s, FIGS / "eq_s2_entropy.png", Inches(3.0), Inches(4.4), Inches(7.0))

text(s, "Low entropy → one cluster dominates → re-rank MF candidates by that centroid.\nHigh entropy → no cluster dominates → ask one disambiguating question grounded in the centroid contents.",
     Inches(1.0), Inches(6.0), Inches(11.0), Inches(1.0), size=13, color=GREY, align=PP_ALIGN.CENTER, italic=True)

footer(s, 11, 22)


# ─────────────────────────────────────────── slide 12 — Vector Storage

s = slide()
header_bar(s, "Vector Storage — FAISS as our vector database",
           "One index serves item-metadata retrieval (S1) and session memory (Adam's existing path)")
picture(s, FIGS / "vector_db.png", Inches(0.4), Inches(1.45), Inches(12.5))
footer(s, 12, 22)


# ─────────────────────────────────────────── slide 13 — Embedding space

s = slide()
header_bar(s, "Item Embedding Space — what's in the vector DB",
           "1,682 MovieLens items embedded with MiniLM-L6-v2, projected to 2D via PCA")
picture(s, FIGS / "embeddings_scatter.png", Inches(0.6), Inches(1.45), Inches(12.1))
text(s,
     "Action and Adventure share the right cluster; Comedy and Drama spread across the left half; Documentaries sit apart. The structure is what FAISS searches against.",
     Inches(0.6), Inches(6.5), Inches(12.1), Inches(0.7), size=13, color=NAVY, align=PP_ALIGN.CENTER, italic=True)
footer(s, 13, 22)


# ─────────────────────────────────────────── slide 14 — Live retrieval example

s = slide()
header_bar(s, "Live Retrieval — natural-language query into FAISS",
           "Same embedder, same index — three intents, top-5 nearest items")
picture(s, FIGS / "retrieval_example.png", Inches(0.4), Inches(1.45), Inches(12.5))
text(s,
     "No keyword matching, no rules. The query \"feel-good comedy that will make me laugh\" returns Comedy titles by cosine similarity in embedding space alone. This is the retrieval primitive S1 uses to augment MF candidates.",
     Inches(0.6), Inches(6.5), Inches(12.1), Inches(0.7), size=13, color=NAVY, align=PP_ALIGN.CENTER, italic=True)
footer(s, 14, 22)


# ─────────────────────────────────────────── slide 15 — Cornac baselines

s = slide()
header_bar(s, "Baseline Sweep — Cornac on MovieLens 100K")
picture(s, FIGS.parent.parent / "results" / "cornac_metrics.png", Inches(0.4), Inches(1.4), Inches(12.5))
text(s,
     "MF wins rating prediction (RMSE 0.89). BPR wins ranking metrics (NDCG@10 0.23) because it optimizes a pairwise ranking loss rather than reconstruction. The split motivates reporting both metric families for the hybrid comparison.",
     Inches(0.6), Inches(6.4), Inches(12.1), Inches(0.8), size=14, color=NAVY)
footer(s, 15, 22)


# ─────────────────────────────────────────── slide 16 — 4-way chart

s = slide()
header_bar(s, "Four-Way Comparison — MF / +S1 / +S2 / +Both")
picture(s, RESULTS / "compare_all.png", Inches(0.4), Inches(1.4), Inches(12.5))
text(s,
     "Five metrics across four configurations on the same MovieLens 100K split. Accuracy on the left (NDCG, Recall). Exploration on the right (Diversity, Novelty, Coverage). Read across to see each layer's contribution.",
     Inches(0.6), Inches(6.4), Inches(12.1), Inches(0.8), size=14, color=NAVY)
footer(s, 16, 22)


# ─────────────────────────────────────────── slide 17 — trade-off scatter

s = slide()
header_bar(s, "Accuracy ↔ Exploration Trade-off")
picture(s, FIGS / "tradeoff.png", Inches(2.0), Inches(1.4), Inches(9.3))
text(s,
     "Every additional layer slides the system along an explore-exploit axis. S1 alone is the only point that wins accuracy and gains coverage. S2 trades accuracy for coverage. Both combined pushes coverage and novelty to their peaks at the lowest accuracy.",
     Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.8), size=13, color=NAVY, align=PP_ALIGN.CENTER)
footer(s, 17, 22)


# ─────────────────────────────────────────── slide 18 — numerical table + interpretation

s = slide()
header_bar(s, "Numerical Results & Reading")

data = [
    ["Configuration", "NDCG@10", "Recall@10", "Diversity", "Novelty", "Coverage"],
    ["MF",            "0.0548",  "0.0428",   "0.6802",   "2.5814", "0.0669"],
    ["MF + S1",       "0.0654",  "0.0585",   "0.5434",   "2.7804", "0.1538"],
    ["MF + S2",       "0.0483",  "0.0462",   "0.5165",   "2.9223", "0.2900"],
    ["MF + S1 + S2",  "0.0375",  "0.0375",   "0.4660",   "3.5474", "0.3495"],
]
add_table(s, data, Inches(0.6), Inches(1.5), Inches(12.1), Inches(2.6), body_size=15)

card(s, Inches(0.6),  Inches(4.4), Inches(3.95), Inches(2.4),
     "S1 wins accuracy",
     "NDCG up 19 %, Recall up 37 %. Anchoring retrieval to the user's profile beats pure CF popularity on a sparse dataset like ML-100K.",
     accent=ORANGE)
card(s, Inches(4.7),  Inches(4.4), Inches(3.95), Inches(2.4),
     "S2 wins coverage",
     "Coverage up 333 %, Novelty up 13 %. Cluster conditioning routes different users into different parts of the catalog.",
     accent=TEAL)
card(s, Inches(8.8),  Inches(4.4), Inches(3.9), Inches(2.4),
     "Both — peak exploration",
     "Coverage 35 %, Novelty 3.55. Lowest NDCG. The two layers compose multiplicatively along the explore-exploit axis.",
     accent=NAVY)

footer(s, 18, 22)


# ─────────────────────────────────────────── slide 19 — module map

s = slide()
header_bar(s, "Contribution Map", "Who built what")
picture(s, FIGS / "contributions.png", Inches(0.4), Inches(1.3), Inches(12.5))
footer(s, 19, 22)


# ─────────────────────────────────────────── slide 20 — live demo

s = slide()
header_bar(s, "Live Demo", "Baseline vs. augmented on the same user intent")

card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.0),
     "Baseline — Adam's HF Space",
     "huggingface.co/spaces/asabry1233/ai-recommender\n\nMF + LangGraph + LLM agents.\nNo semantic retrieval. No cluster conditioning.",
     accent=GREY)

card(s, Inches(6.7), Inches(1.5), Inches(6.0), Inches(5.0),
     "Augmented — comparison build",
     "Same pipeline with S1 + S2 toggled on.\n\nSide-by-side execution on the same intent shows the trade-off live.",
     accent=ORANGE)

# placeholder strip
rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(6.6), Inches(12.1), Inches(0.5))
rect.fill.solid(); rect.fill.fore_color.rgb = PLACE; rect.line.color.rgb = ORANGE
rect.text_frame.text = "[ PLACEHOLDER — live demo screenshots and 2-minute backup video to be captured before presentation ]"
for p in rect.text_frame.paragraphs:
    p.font.size = Pt(11); p.font.italic = True; p.font.color.rgb = GREY; p.alignment = PP_ALIGN.CENTER

footer(s, 20, 22)


# ─────────────────────────────────────────── slide 21 — lessons + future work

s = slide()
header_bar(s, "Lessons Learned & Next Steps")

card(s, Inches(0.6), Inches(1.4), Inches(5.95), Inches(5.5),
     "Lessons Learned",
     ["Offline rating metrics under-represent discovery objectives — report coverage and novelty alongside NDCG.",
      "",
      "The semantic layer (S1) cleanly addresses cold-start without retraining.",
      "",
      "LangGraph is the right level of the stack when the pipeline is a stateful graph, not a linear chain.",
      "",
      "The two hybrid layers compose without being redundant — different mechanisms, complementary effects."],
     accent=NAVY, body_size=13)

card(s, Inches(6.75), Inches(1.4), Inches(5.95), Inches(5.5),
     "Future Work",
     ["Expose the entropy-triggered clarifying-question elicitation (primitive already implemented in S2).",
      "",
      "Online cluster discovery — spawn a new cluster when the current session matches none well.",
      "",
      "Swap the shallow joint-MF for a deep Two-Tower backbone (NeuMF / LightGCN).",
      "",
      "Domain expansion (music, fitness, podcasts) — S1 makes this free of cross-domain training data."],
     accent=ORANGE, body_size=13)

footer(s, 21, 22)


# ─────────────────────────────────────────── slide 22 — thank you / Q&A

s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.6))
band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
band2 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(6.9), Inches(13.333), Inches(0.6))
band2.fill.solid(); band2.fill.fore_color.rgb = ORANGE; band2.line.fill.background()

text(s, "Thank you — questions?", Inches(0.5), Inches(2.4), Inches(12.3), Inches(1.5),
     size=54, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

text(s,
     "Repository:   github.com/AdamSabry1233/AI-Life-Recommender-\n"
     "Baseline demo:   huggingface.co/spaces/asabry1233/ai-recommender",
     Inches(0.5), Inches(4.3), Inches(12.3), Inches(1.5), size=18, color=NAVY, align=PP_ALIGN.CENTER)

text(s, "Adam Sabry  ·  Rammy Baroudi  ·  Stephen Taylor",
     Inches(0.5), Inches(5.8), Inches(12.3), Inches(0.5),
     size=14, color=GREY, align=PP_ALIGN.CENTER, italic=True)


OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"Wrote {OUT}")
