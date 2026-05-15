"""Generate the final project report (.docx) for CMPE 256 Term Project."""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "submission" / "LifeRecommender_FinalReport.docx"
RESULTS = REPO / "results"

doc = Document()

# default style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)


def h1(text):
    p = doc.add_heading(text, level=1)
    return p


def h2(text):
    p = doc.add_heading(text, level=2)
    return p


def para(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    return p


def placeholder(text):
    p = doc.add_paragraph()
    run = p.add_run(f"[ PLACEHOLDER — {text} ]")
    run.italic = True
    run.font.color.rgb = RGBColor(0x99, 0x66, 0x00)
    return p


def fig(path: Path, caption: str, width_in: float = 6.0):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure: {caption}")
        run.italic = True
        run.font.size = Pt(10)
    else:
        placeholder(f"figure missing: {caption}")


# ─────────────────────────────────────────── Title page

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("AI Life Recommender")
r.bold = True
r.font.size = Pt(28)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Context-Aware Multi-Domain Recommendations\nwith Hybrid Collaborative Filtering and Agentic LLM Orchestration")
r.font.size = Pt(14)

doc.add_paragraph()
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("CMPE 256 — Recommender Systems — Term Project\nSan Jose State University\n").font.size = Pt(12)
meta.add_run("Team: Adam Sabry, Rammy Baroudi, Stephen Taylor\n").font.size = Pt(12)
placeholder("Submission date")

doc.add_page_break()

# ─────────────────────────────────────────── 1. Project Goals

h1("1. Project Goals")
para(
    "Standard recommender systems serve a single domain — movies on Netflix, products on "
    "Amazon, songs on Spotify. They optimize for rating prediction or click-through, and "
    "they assume the user's preferences are a single point in latent space. Neither "
    "assumption holds in the broader case of life recommendation. A person planning their "
    "evening wants a movie, a meal, a book, and a habit that fit together coherently, and "
    "what they want depends on a mood signal that no rating matrix captures."
)
para(
    "This project builds an AI Life Recommender that produces coordinated cross-domain "
    "suggestions conditioned on natural-language intent. The system targets four user-facing "
    "outcomes: cold-start handling without weeks of interaction history, contextual "
    "responsiveness to mood and constraints, cross-domain coherence so the picks form a "
    "package rather than four independent guesses, and explainability so each recommendation "
    "is accompanied by a textual reason. Business value sits in personalized wellness, "
    "lifestyle apps, and conversational commerce — settings where the user wants help "
    "deciding what to do next, not a ranked list of one product."
)

# ─────────────────────────────────────────── 2. Data Description

h1("2. Data Description")

h2("2.1 Sources")
para(
    "The system trains on three publicly available datasets, each contributing a distinct "
    "recommendation domain:"
)
para(
    "• MovieLens 25M — approximately 25 million ratings on 62,000 movies from 162,000 users, "
    "with timestamps, genre tags, and tag relevance scores."
)
para(
    "• Food.com Recipes and User Interactions — approximately 1 million user-recipe "
    "interactions across 230,000 recipes with tags, ingredients, nutritional metadata, and "
    "preparation time."
)
para(
    "• Amazon Books Reviews — approximately 3 million rating-and-review tuples across roughly "
    "200,000 books with titles, authors, and category metadata."
)
para(
    "Combined interaction count is approximately 29 million events. All three sources are "
    "explicit-feedback rating datasets with a 1–5 scale and timestamped events suitable for "
    "chronological train-test splitting."
)

h2("2.2 Sparsity Analysis")
placeholder(
    "Sparsity heatmap and per-domain density table — to be generated from the unified "
    "interaction matrix; will report users × items and density percentage per domain"
)
para(
    "MovieLens 25M sits at approximately 0.25 percent density, Food.com at well under 0.05 "
    "percent, and Amazon Books at approximately 0.015 percent. The joint user space is "
    "considerably sparser because most users participate in only one domain, which is a key "
    "argument for the shared user embedding architecture described in Section 3."
)

h2("2.3 Exploratory Data Analysis")
placeholder(
    "Rating distribution per domain — three histograms showing the per-domain rating "
    "distributions (MovieLens skews to 4 stars, Food.com to 5 stars, Amazon Books bimodal)"
)
placeholder(
    "Long-tail interaction distribution — log-log plot of items by interaction count per "
    "domain, demonstrating the power-law popularity tail that motivates novelty and coverage "
    "metrics rather than purely accuracy-based evaluation"
)
placeholder(
    "Per-user activity histogram — distribution of interactions per user, showing the small "
    "fraction of high-activity users who dominate the rating matrix"
)
para(
    "The three datasets share the standard recommender-system pathologies: heavily "
    "right-skewed item popularity, a small head of highly active users, and a long tail of "
    "single-interaction users who pose the cold-start challenge. These properties motivate "
    "the design choices in Section 3."
)

# ─────────────────────────────────────────── 3. Methodology

h1("3. Methodology and Novelty")

h2("3.1 Joint Multi-Domain Matrix Factorization")
para(
    "The scoring backbone is a single matrix factorization model trained jointly across all "
    "three domains rather than three independent models. Each user has one embedding shared "
    "across movies, food, and books; each item has one embedding within its domain; each "
    "domain has a context vector added to the user embedding before the inner product. The "
    "score function is"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("score(user, item, domain) = (user_emb + domain_emb) · item_emb + user_bias + item_bias + global_bias")
r.italic = True
para(
    "Training uses the Adam optimizer with mean squared error loss on observed ratings. The "
    "joint formulation lets collaborative signal in one domain inform predictions in another "
    "for users who appear in multiple datasets, and gives a coherent latent representation "
    "of taste that downstream layers can query."
)

h2("3.2 LangGraph Agent Orchestration")
para(
    "On top of the scoring backbone sits a LangGraph state machine with five sequential "
    "nodes. A memory-retrieval node queries a FAISS index of past sessions to recover "
    "relevant prior recommendations. Three domain agents — entertainment, food, and "
    "learning — each pull top-30 candidates from the matrix factorization model, retain the "
    "top three for quality, shuffle the remaining 27 for variety, and pass ten candidates "
    "plus the user's intent and persisted preferences to a Groq-hosted Llama 3 model that "
    "selects one item and produces a one-sentence reason. A fourth agent handles daily "
    "habits with no matrix factorization backing — the LLM reasons from intent alone because "
    "no public rating dataset exists for habits. An orchestrator node synthesizes the four "
    "picks into a coherent two-to-three-sentence response."
)

h2("3.3 Multi-Turn Classification and Memory")
para(
    "Follow-up messages do not restart the pipeline. A prompted LLM classifier routes each "
    "incoming message into question (answered from current recommendations), refinement "
    "(swaps one recommendation by re-sampling from 40 shuffled candidates), concern "
    "(re-filters a recommendation under a stated constraint such as a dietary restriction), "
    "or new intent (saves the current session and starts a fresh pipeline). Completed "
    "sessions are embedded with sentence-transformers all-MiniLM-L6-v2 and indexed in FAISS "
    "for cross-session recall. User profiles persist in Upstash Redis with a 90-day TTL."
)

h2("3.4 Semantic Cross-Domain Retrieval (S1)")
para(
    "The first novelty contribution adds a content-based retrieval layer parallel to the "
    "collaborative backbone. Each item's metadata text — title, genre tags, description — is "
    "embedded with sentence-transformers and indexed in a shared FAISS IndexFlatIP. At query "
    "time a user-profile vector is computed as the L2-normalized mean of the embeddings of "
    "items the user has rated highly; this profile vector queries the shared index and "
    "returns nearest neighbors across all three domains in a single retrieval. The semantic "
    "and matrix-factorization scores are fused via min-max normalization and a tunable "
    "weight α. The mechanism addresses three weaknesses of pure collaborative filtering: "
    "cold-start for users with no interaction history, cross-domain bridging without "
    "requiring cross-domain rating data, and surfacing of long-tail items that lack "
    "collaborative signal but share semantic character with the user's profile."
)

h2("3.5 Per-User Cluster Conditioning (S2)")
para(
    "The second novelty contribution replaces the single-profile-vector assumption with a "
    "multi-cluster user model. KMeans is fit on the embeddings of each user's positively "
    "rated items to produce K taste centroids. At query time the user's most recent "
    "positive interaction acts as a context vector; a softmax over its cosine similarity to "
    "each centroid yields a posterior over which cluster is currently active. Matrix "
    "factorization candidate scores are re-ranked by proximity to the active centroid via "
    "min-max blending with a tunable weight. The Shannon entropy of the posterior is "
    "computed as an ambiguity signal — when no cluster dominates, the system should ask a "
    "disambiguating question grounded in the centroid content. The clarifying-question "
    "elicitation path is implemented as a primitive but not yet exposed in the live system; "
    "it is the natural next addition."
)

h2("3.6 System Architecture Diagram")
placeholder(
    "End-to-end system diagram — to be drawn for the slides and replicated here; shows user "
    "intent flowing through Redis profile lookup, FAISS memory retrieval, joint MF scoring, "
    "S1/S2 augmentation, four LangGraph agents, orchestrator synthesis, and Gradio output"
)

# ─────────────────────────────────────────── 4. Benchmark and Evaluation

h1("4. Benchmark and Evaluation")

h2("4.1 Cornac Baseline Sweep")
para(
    "To anchor the joint matrix factorization against standard collaborative-filtering "
    "approaches, four classical models from the Cornac library — MF, PMF, NMF, and BPR — "
    "were evaluated on MovieLens 100K with an 80-20 random split and seed 42. RMSE and MAE "
    "measure rating-prediction quality; NDCG@10, Precision@10, and Recall@10 measure "
    "top-K ranking quality."
)
fig(RESULTS / "cornac_metrics.png", "Cornac baseline comparison: MF, PMF, NMF, BPR on MovieLens 100K.")
para(
    "MF wins the rating-prediction metrics (lowest RMSE at 0.89). BPR wins the ranking "
    "metrics (NDCG@10 of 0.23) but at the cost of much higher RMSE because BPR optimizes a "
    "pairwise ranking loss rather than rating reconstruction. This split between "
    "rating-optimized and ranking-optimized models is the standard reason recommender "
    "systems are evaluated on both metric families rather than only one."
)

h2("4.2 Four-Way Hybrid Comparison")
para(
    "The two novelty contributions (S1 and S2) and their combination were compared against "
    "the matrix-factorization baseline using the same MovieLens 100K split. Metrics include "
    "NDCG@10 and Recall@10 (accuracy), Diversity (intra-list mean cosine distance), Novelty "
    "(mean log-inverse popularity), and Coverage (fraction of catalog ever recommended "
    "across users)."
)
fig(RESULTS / "compare_all.png", "Four-way comparison: MF, MF+S1, MF+S2, MF+S1+S2 on MovieLens 100K.")
para(
    "Numerical results:"
)

# embedded table
table = doc.add_table(rows=5, cols=6)
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
for i, h in enumerate(["Configuration", "NDCG@10", "Recall@10", "Diversity", "Novelty", "Coverage"]):
    hdr[i].text = h
rows = [
    ("MF",            "0.0548", "0.0428", "0.6802", "2.5814", "0.0669"),
    ("MF + S1",       "0.0654", "0.0585", "0.5434", "2.7804", "0.1538"),
    ("MF + S2",       "0.0483", "0.0462", "0.5165", "2.9223", "0.2900"),
    ("MF + S1 + S2",  "0.0375", "0.0375", "0.4660", "3.5474", "0.3495"),
]
for r_idx, row in enumerate(rows, start=1):
    for c_idx, val in enumerate(row):
        table.rows[r_idx].cells[c_idx].text = val

para("")
para(
    "Reading the table: S1 alone is the only configuration that improves accuracy and "
    "coverage simultaneously — NDCG up 19 percent, Recall up 37 percent, Coverage up 130 "
    "percent, Novelty up 8 percent, with diversity giving up 20 percent because anchoring "
    "to the user profile narrows the genre spread of a top-10 list. S2 alone trades "
    "accuracy for exploration — NDCG down 12 percent in exchange for Coverage up 333 "
    "percent. Stacking both layers compounds the exploration effect — Coverage 35 percent "
    "and Novelty 3.55 are the configuration peaks, at the lowest NDCG."
)
para(
    "The interpretation that matters for a recommender whose purpose is discovery rather "
    "than reproduction: NDCG and Recall against held-out ratings implicitly reward "
    "predicting items the user has already interacted with, which over-represents popular "
    "items. Coverage and Novelty are the metrics that capture the project's actual goal. "
    "On those, every additional layer contributes a measurable expansion of what the user "
    "sees. The trade-off is controllable via the fusion weight α and the cluster-conditioning "
    "weight, both of which are exposed parameters and could be set per user, per session, "
    "or per A/B test."
)
para(
    "The two layers are complementary rather than redundant. If they were doing similar "
    "work, the combined configuration would produce metrics close to whichever single layer "
    "dominates; instead it pushes further than either alone. Semantic retrieval bridges "
    "cold regions of the catalog via content, cluster conditioning partitions the user's "
    "taste space into context-activated modes — these are different mechanisms and they "
    "compose."
)

h2("4.3 Live System Evaluation")
placeholder(
    "Side-by-side live demo comparison — baseline HF Space vs. augmented HF Space showing "
    "the same user intent producing different recommendation packages; screenshots to be "
    "captured before the presentation"
)
placeholder(
    "Subjective quality table — qualitative ratings of cross-domain coherence and "
    "constraint handling on a fixed set of 10 user intents; rated by team members on a 1–5 "
    "scale; to be filled in the week before the presentation"
)

# ─────────────────────────────────────────── 5. Lessons Learned

h1("5. Lessons Learned and Takeaways")

h2("5.1 Cold Start")
para(
    "Pure collaborative filtering produces noisy recommendations for users with fewer than "
    "approximately 20 ratings. The joint multi-domain architecture partially mitigates this "
    "for users who participate in multiple domains, since the shared user embedding picks "
    "up signal from any one of them. The semantic retrieval layer (S1) addresses cold-start "
    "more directly: a user with no interaction history at all can still receive sensible "
    "recommendations because the four onboarding questions and the natural-language intent "
    "are embedded into the same space as the items. This is the cleanest demonstration of "
    "why hybrid content-based plus collaborative architectures matter in practice."
)

h2("5.2 Scalability")
para(
    "Joint matrix factorization scales linearly in training time with the number of "
    "interactions, and the trained checkpoint fits in well under 100 MB at the dimensions "
    "used. Inference is O(item count) per recommendation in the naive implementation. The "
    "FAISS index makes the semantic retrieval layer effectively sublinear (approximate "
    "nearest neighbor) in catalog size, so adding S1 actually improves the system's "
    "scalability for cross-domain queries compared to running three separate collaborative "
    "filters and joining their outputs. The per-user cluster fitting in S2 is offline and "
    "embarrassingly parallel across users — it does not affect inference cost."
)

h2("5.3 Evaluation Methodology")
para(
    "A central lesson is that offline metrics against held-out ratings cannot fairly grade "
    "a discovery-oriented recommender. The held-out items are by definition things the user "
    "has already interacted with, and popular items dominate that distribution. Coverage, "
    "Novelty, and Diversity are necessary complements to NDCG and Recall when the system's "
    "purpose is to expand what the user sees rather than predict what they already like. "
    "The four-way comparison in Section 4 only tells the story it does because all five "
    "metrics are reported together."
)

h2("5.4 LLM Orchestration")
para(
    "LangGraph proved appropriate because the pipeline is genuinely a state machine, not a "
    "linear chain — multiple agents share and mutate an AgentState, follow-ups route "
    "conditionally based on the multi-turn classifier, and the orchestrator synthesizes "
    "across the agent outputs. A pure LangChain LCEL implementation would have required "
    "manual dict-threading between calls and bespoke conditional routing. The cost is one "
    "additional dependency and a learning curve; the payoff is that adding a new agent "
    "node — for example a music agent or a fitness agent — is a one-file addition rather "
    "than a refactor."
)

# ─────────────────────────────────────────── 6. Source Code

h1("6. Source Code")
para(
    "Public GitHub repository: https://github.com/AdamSabry1233/AI-Life-Recommender-"
)
para(
    "Baseline live demo on Hugging Face Spaces: "
    "https://huggingface.co/spaces/asabry1233/ai-recommender"
)
placeholder("Augmented live demo URL — Stephen's HF Space with S1 + S2 enabled (to deploy)")

# ─────────────────────────────────────────── 7. Contribution Table

h1("7. Contribution Table")

t = doc.add_table(rows=4, cols=3)
t.style = "Light Grid Accent 1"
hdr = t.rows[0].cells
for i, h in enumerate(["Member", "Primary Responsibilities", "Specific Deliverables"]):
    hdr[i].text = h

rows = [
    ("Adam Sabry",
     "System architecture and primary implementation",
     "Joint multi-domain matrix factorization training; LangGraph agent pipeline; FAISS "
     "session memory; Upstash Redis profile persistence; Gradio frontend; FastAPI backend; "
     "Hugging Face Spaces deployment; initial proposal and documentation."),
    ("Rammy Baroudi",
     "Data pipeline and evaluation methodology",
     "Dataset acquisition and preprocessing across MovieLens, Food.com, and Amazon Books; "
     "train-test split design; sparsity analysis; report writing on data description and "
     "lessons learned; presentation slides for data and evaluation sections."),
    ("Stephen Taylor",
     "Novelty contributions, evaluation infrastructure, baselines",
     "Cornac baseline sweep (MF/PMF/NMF/BPR) with grouped-metric visualization; semantic "
     "cross-domain retrieval module (S1); per-user cluster conditioning module (S2); "
     "four-way comparison harness and chart; pytest test suite with central delegator and "
     "watch/diff-based reruns; augmented Hugging Face Space; report sections on methodology "
     "novelty and evaluation."),
]
for r_idx, row in enumerate(rows, start=1):
    for c_idx, val in enumerate(row):
        t.rows[r_idx].cells[c_idx].text = val

doc.add_paragraph()
placeholder(
    "Adjust the contribution table once final task allocation is confirmed with the team."
)

# save
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"Wrote {OUT}")
