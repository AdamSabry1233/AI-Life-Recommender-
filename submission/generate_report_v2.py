"""Visual-first final report (.docx) — tables, diagrams, equations, prose only where it carries reasoning."""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "submission" / "figures"
RESULTS = REPO / "results"
OUT = REPO / "submission" / "LifeRecommender_FinalReport_v2.docx"

NAVY = RGBColor(0x1B, 0x2A, 0x4E)
ORANGE = RGBColor(0xE5, 0x8A, 0x2D)
GREY = RGBColor(0x55, 0x55, 0x55)

doc = Document()

# base style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

# page margins
section = doc.sections[0]
section.top_margin = Inches(0.9)
section.bottom_margin = Inches(0.9)
section.left_margin = Inches(0.9)
section.right_margin = Inches(0.9)


def h1(text):
    p = doc.add_heading(text, level=1)
    for r in p.runs:
        r.font.color.rgb = NAVY
        r.font.size = Pt(18)
    return p


def h2(text):
    p = doc.add_heading(text, level=2)
    for r in p.runs:
        r.font.color.rgb = NAVY
        r.font.size = Pt(14)
    return p


def para(text, size=11):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)
    for r in p.runs:
        r.font.size = Pt(size)
    return p


def placeholder(text):
    p = doc.add_paragraph()
    run = p.add_run(f"[ PLACEHOLDER — {text} ]")
    run.italic = True
    run.font.color.rgb = RGBColor(0x99, 0x66, 0x00)
    return p


def fig(path: Path, caption: str, width_in: float = 6.4):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(f"Figure. {caption}")
        run.italic = True
        run.font.size = Pt(10)
        run.font.color.rgb = GREY
    else:
        placeholder(f"figure missing: {caption}")


def equation_inline(path: Path, width_in: float = 4.6):
    if path.exists():
        doc.add_picture(str(path), width=Inches(width_in))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER


def make_table(rows, header=True, col_widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Light Grid Accent 1"
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = t.rows[r_i].cells[c_i]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
                    if header and r_i == 0:
                        run.bold = True
                        run.font.color.rgb = NAVY
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return t


# ─────────── title page

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("AI Life Recommender")
r.bold = True
r.font.size = Pt(32)
r.font.color.rgb = NAVY

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Context-Aware Multi-Domain Recommendations\nwith Hybrid Collaborative Filtering and Agentic LLM Orchestration")
r.font.size = Pt(14)
r.font.color.rgb = GREY

doc.add_paragraph()
fig(FIGS / "architecture.png", "End-to-end system architecture.", width_in=6.8)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("\nTeam: Adam Sabry  ·  Rammy Baroudi  ·  Stephen Taylor")
r.font.size = Pt(12)
r.bold = True
r.font.color.rgb = NAVY

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("CMPE 256 — Recommender Systems — San José State University")
r.font.size = Pt(12)
r.font.color.rgb = GREY

placeholder("Submission date")
doc.add_page_break()


# ─────────── 1. Project Goals

h1("1. Project Goals")

para(
    "Production recommenders serve a single domain and optimize a single objective — rating "
    "prediction on Netflix, click-through on Amazon, listen completion on Spotify. The "
    "decisions a person actually makes are seldom single-domain and almost never captured "
    "by a rating matrix. This project builds a recommender that produces coordinated "
    "cross-domain suggestions (movie, meal, book, daily habit) conditioned on natural-"
    "language mood and intent."
)

h2("1.1 Problem Statement Matrix")
make_table([
    ["Production gap",                            "Why standard CF cannot close it",                                 "How this system addresses it"],
    ["Single-domain optimization",                "Rating matrices are within one catalog",                         "Joint MF with one shared user embedding across three datasets"],
    ["No context signal in the model",            "Mood, time, energy are not user × item entries",                 "LLM agents read intent text and constrain candidate selection"],
    ["Independent per-domain picks",              "Three separate CF models cannot align outputs",                  "Orchestrator node synthesizes the four picks into a coherent package"],
    ["Catalog gaps (e.g., habits)",               "No public rating data for daily habits",                         "Habit agent runs LLM-only — no MF backing required"],
    ["Cold start",                                "New users have no collaborative signal",                         "4-question onboarding + S1 semantic profile vector + S2 fallback"],
], col_widths=[1.8, 2.6, 2.6])

para("")
para(
    "Business value sits in personalized wellness, lifestyle assistance, and conversational "
    "commerce — settings where the user wants help deciding what to do, not a ranked list of "
    "one product."
)


# ─────────── 2. Data

h1("2. Data Description")

h2("2.1 Sources")
make_table([
    ["Domain",        "Source",                "Interactions",  "Items",   "Density",         "Notes"],
    ["Entertainment", "MovieLens 25M",         "≈ 25,000,000",  "62,000",  "≈ 0.25 %",        "Genres, tag relevance, timestamps"],
    ["Food",          "Food.com Recipes",      "≈ 1,000,000",   "230,000", "< 0.05 %",        "Tags, ingredients, prep time"],
    ["Learning",      "Amazon Books Reviews",  "≈ 3,000,000",   "200,000", "≈ 0.015 %",       "Titles, authors, category metadata"],
    ["Habit",         "no public dataset",     "—",             "—",       "—",               "LLM-only domain (no rating data exists)"],
], col_widths=[1.2, 1.6, 1.2, 0.9, 0.9, 2.4])

para("")
para(
    "Combined explicit-feedback events ≈ 29 million on a 1–5 scale. All three sources include "
    "timestamps, enabling chronological train/test splits."
)

h2("2.2 Sparsity")
placeholder("Sparsity heatmap of the unified user × item matrix — to be generated from the merged interaction frame")
para(
    "The joint user space is sparser than any individual domain because most users participate "
    "in only one dataset. This is the central motivation for a shared user embedding rather "
    "than three independent collaborative filters — pooled signal lets the model produce "
    "non-degenerate latent representations even for users with thin per-domain history."
)

h2("2.3 Exploratory Data Analysis")
placeholder("Per-domain rating distribution — three histograms (MovieLens, Food.com, Amazon Books)")
placeholder("Long-tail interaction distribution — log-log plot of items by interaction count per domain")
placeholder("Per-user activity distribution — histogram of interactions per user, showing the heavy head and long tail of cold-start users")
para(
    "All three datasets exhibit the standard recommender-system pathologies: heavily right-"
    "skewed item popularity, a small head of highly active users, and a long tail of single-"
    "interaction users who define the cold-start problem. Coverage and novelty metrics are "
    "reported in Section 4 precisely because these pathologies make rating-accuracy metrics "
    "an incomplete signal."
)

doc.add_page_break()

# ─────────── 3. Methodology

h1("3. Methodology and Novelty")

h2("3.1 Joint Multi-Domain Matrix Factorization")
equation_inline(FIGS / "eq_mf.png", width_in=4.6)
para(
    "One user embedding shared across all three domains; one item embedding per item in its "
    "domain; one domain context vector added to the user embedding before the inner product. "
    "Training uses Adam with mean squared error on observed ratings. The joint formulation "
    "lets collaborative signal in one domain inform predictions in others for users active in "
    "multiple datasets, and produces a single latent representation of taste that downstream "
    "layers can query."
)

h2("3.2 LangGraph Agent Orchestration")
fig(FIGS / "architecture.png", "Full LangGraph pipeline. Five sequential nodes operate on a shared AgentState.", width_in=6.6)
para(
    "The retrieve_memory node queries a FAISS index of past sessions for relevant prior "
    "recommendations. Three domain agents — entertainment, food, learning — each pull top-30 "
    "candidates from the joint MF model, retain the top three for a quality floor, shuffle "
    "the remaining 27 for variety, and pass ten candidates plus the user's intent and "
    "persisted preferences to a Groq-hosted Llama 3 model that returns one item with a one-"
    "sentence reason. The fourth agent handles daily habits with no MF backing — the LLM "
    "reasons from intent alone. The orchestrator node synthesizes the four picks into a "
    "coherent two-to-three-sentence response."
)

h2("3.3 Multi-Turn Classification and Memory")
make_table([
    ["Follow-up class", "What it triggers",                                                    "State change"],
    ["question",        "Answer from current recommendation set + recent conversation",        "None"],
    ["refinement",      "Swap one recommendation — resample from 40 shuffled candidates",     "Targeted slot replaced"],
    ["concern",         "Apply dietary/ethical filter and reroll the affected agent",         "Targeted slot replaced under constraint"],
    ["new_intent",      "Save current session to FAISS and run the full pipeline fresh",      "Full state reset"],
], col_widths=[1.4, 4.4, 2.2])

para("")
para(
    "Completed sessions are embedded with sentence-transformers all-MiniLM-L6-v2 and indexed "
    "in FAISS for cross-session recall. User profiles persist in Upstash Redis with a 90-day "
    "TTL."
)

h2("3.4 Novelty 1 — Semantic Cross-Domain Retrieval (S1)")
fig(FIGS / "s1_mechanism.png", "S1 retrieval pipeline. Item embeddings indexed once; user-profile vector built from history; fused with MF scores.", width_in=6.6)

para("User profile vector — mean-pooled and L2-normalized embeddings of the user's rated items:")
equation_inline(FIGS / "eq_user_profile.png", width_in=4.6)

para("Fused ranking — min-max normalized weighted sum of MF and semantic scores:")
equation_inline(FIGS / "eq_s1.png", width_in=4.6)

para(
    "The mechanism addresses three distinct CF weaknesses with one structural addition. Cold "
    "start: a user with zero interaction history can still receive sensible picks because the "
    "intent text and the onboarding answers embed into the same space as the items. Cross-"
    "domain bridging: a unified index lets a single retrieval span all three catalogs without "
    "requiring cross-domain rating data. Long-tail surfacing: items with no collaborative "
    "signal but strong semantic similarity to the user's profile become reachable. The "
    "fusion weight α is exposed as a tunable parameter — see the trade-off analysis in "
    "Section 4."
)

h2("3.5 Novelty 2 — Per-User Multi-Cluster Taste Model (S2)")
fig(FIGS / "s2_mechanism.png", "S2 conditioning pipeline. KMeans on user history produces K centroids; current context picks the active cluster via softmax.", width_in=6.6)

para("Cluster posterior — softmax over cosine similarity between the context vector and each centroid (τ is temperature):")
equation_inline(FIGS / "eq_s2_posterior.png", width_in=4.6)

para("Posterior entropy — ambiguity signal triggering the (future) clarifying-question elicitation:")
equation_inline(FIGS / "eq_s2_entropy.png", width_in=4.0)

para(
    "S2 replaces the single-profile-vector assumption with a multi-cluster user representation. "
    "KMeans on the user's rated-item embeddings produces K = 3 taste centroids; the user's "
    "most recent positive interaction acts as the context vector; the softmax assigns "
    "probability mass to each cluster, and the winning cluster re-ranks MF candidates by "
    "proximity to its centroid. When no cluster dominates (high posterior entropy) the system "
    "is genuinely ambiguous about which user-mode is active — the right response is to ask a "
    "disambiguating question grounded in the centroid contents rather than to guess. The "
    "primitive is implemented; the live elicitation path is the natural next addition."
)

h2("3.6 Vector Storage")
fig(FIGS / "vector_db.png", "FAISS as the vector database. One index serves both the new S1 cross-domain retrieval path and Adam's existing session-memory path.", width_in=6.6)
para(
    "All vector operations in the system — session memory, S1 item retrieval, and the S2 "
    "context-to-centroid similarity — run through one vector database: FAISS with an "
    "IndexFlatIP. L2-normalizing every vector at insertion lets inner-product search "
    "double as cosine similarity, which keeps the math identical across the three use "
    "sites. The embedder is sentence-transformers all-MiniLM-L6-v2, a 22M-parameter "
    "model producing 384-dimensional vectors. The same encoder embeds both items "
    "(metadata text) and queries (intent + onboarding profile), so item vectors and "
    "query vectors live in the same space and a cosine score is meaningful without "
    "additional projection."
)
para(
    "We chose FAISS over a hosted vector database (Pinecone, Weaviate, Qdrant) for three "
    "reasons. First, the working set is small — under 100 K items per domain — so an "
    "in-process flat index outperforms network round-trips to a managed service. Second, "
    "FAISS ships in pip and is already a dependency of the baseline system, so adding "
    "the S1 path required no new infrastructure. Third, IndexFlatIP is exact (no "
    "approximation error), which matters when comparing four configurations on the same "
    "split — we wanted retrieval differences to be attributable to the augmentation "
    "logic, not to ANN-recall variance. At larger catalog sizes the same code can swap "
    "in an IVFFlat or HNSW index with no API changes."
)

doc.add_page_break()

# ─────────── 4. Evaluation

h1("4. Benchmark and Evaluation")

h2("4.1 Cornac Classical Baseline Sweep")
fig(RESULTS / "cornac_metrics.png", "Cornac baseline comparison on MovieLens 100K. Five metrics across MF, PMF, NMF, BPR.", width_in=6.6)
para(
    "MF wins rating prediction (RMSE 0.89). BPR wins ranking metrics (NDCG@10 0.23) at the "
    "cost of much higher RMSE because BPR optimizes a pairwise ranking loss rather than rating "
    "reconstruction. The split between rating-optimized and ranking-optimized models is the "
    "standard reason both metric families are reported."
)

h2("4.2 Four-Way Hybrid Comparison")
fig(RESULTS / "compare_all.png", "Five metrics across MF, MF + S1, MF + S2, MF + S1 + S2 on MovieLens 100K (seed 42, 80/20 split).", width_in=6.6)

make_table([
    ["Configuration", "NDCG@10", "Recall@10", "Diversity", "Novelty", "Coverage"],
    ["MF",            "0.0548",  "0.0428",   "0.6802",   "2.5814", "0.0669"],
    ["MF + S1",       "0.0654",  "0.0585",   "0.5434",   "2.7804", "0.1538"],
    ["MF + S2",       "0.0483",  "0.0462",   "0.5165",   "2.9223", "0.2900"],
    ["MF + S1 + S2",  "0.0375",  "0.0375",   "0.4660",   "3.5474", "0.3495"],
], col_widths=[1.6, 1.1, 1.1, 1.1, 1.1, 1.1])

para("")
fig(FIGS / "tradeoff.png", "Accuracy ↔ exploration trade-off. Each layer slides the system along the same axis.", width_in=5.8)

h2("4.3 Reading the Numbers")
make_table([
    ["Configuration", "Headline shift",                                                    "Cause"],
    ["MF + S1",       "NDCG +19 %, Recall +37 %, Coverage +130 %, Diversity −20 %",       "User-profile-anchored semantic retrieval pulls top-K away from popularity and toward content-similar tail items"],
    ["MF + S2",       "Coverage +333 %, Novelty +13 %, NDCG −12 %",                       "Cluster-conditioned re-rank routes different users into different catalog regions; the active-cluster posterior introduces noise"],
    ["MF + S1 + S2",  "Coverage +423 %, Novelty +37 %, NDCG −32 %",                       "Both layers narrow retrieval into the user's personal region; effects compose multiplicatively along the explore-exploit axis"],
], col_widths=[1.6, 3.2, 3.0])

para("")
para(
    "NDCG and Recall against held-out ratings implicitly reward predicting items the user has "
    "already interacted with, and the held-out distribution over-represents popular items "
    "because rating frequency correlates with popularity. Coverage, Novelty, and Diversity are "
    "the metrics that actually capture a discovery system's objective. On those, every "
    "additional layer of this system contributes a measurable expansion of what the user "
    "sees, and the explore-exploit position is controllable via the α fusion weight in S1 "
    "and the cluster-conditioning weight in S2 — both exposed parameters."
)

para(
    "The two layers are complementary rather than redundant. If they were doing similar work "
    "the combined configuration would produce metrics close to whichever single layer "
    "dominates; instead it pushes further than either alone. S1 bridges cold regions of the "
    "catalog via content similarity. S2 partitions the user's taste space into context-"
    "activated modes. These are different mechanisms and they stack."
)

h2("4.4 Live System Evaluation")
placeholder("Side-by-side screenshots — baseline HF Space vs. augmented HF Space for the same intent")
placeholder("Qualitative cross-domain coherence ratings on a fixed set of 10 intents (1–5 scale, rated by team)")

h2("4.5 Embedding-Space Validation")
fig(FIGS / "embeddings_scatter.png", "PCA projection of the 1,682 MovieLens 100K items embedded with MiniLM-L6-v2, colored by primary genre.", width_in=6.4)
para(
    "Beyond the four-configuration numerical comparison in Section 4.2, the embedding "
    "space itself can be inspected directly to confirm it is doing what S1 and S2 "
    "depend on. Projecting all 1,682 MovieLens 100K item embeddings to two principal "
    "components shows that items of the same primary genre cluster together — Action "
    "and Adventure form a contiguous region, Comedy spreads across one side of the "
    "manifold, and Documentary sits well separated from the dramatic genres. The "
    "structure is implicit in the off-the-shelf sentence transformer; no fine-tuning "
    "was performed on the MovieLens metadata. This is the prior that S1 leverages when "
    "fusing semantic similarity into the MF candidate scores."
)
fig(FIGS / "retrieval_example.png", "FAISS top-5 retrieval for three natural-language queries. No keyword matching — purely cosine similarity in embedding space.", width_in=6.6)
para(
    "Three concrete query examples demonstrate the retrieval primitive end-to-end. The "
    "query \"high-energy action and adventure for a workout night\" returns Rock, "
    "Quest, Conan the Barbarian, Ghost and the Darkness, and Waterworld — all action "
    "or action-adventure titles. The query \"thoughtful slow-paced drama for a quiet "
    "evening\" returns Quiet Room, Very Natural Thing, 8 Seconds, One Fine Day, and "
    "Short Cuts — predominantly drama. The query \"feel-good comedy that will make me "
    "laugh\" returns My Fellow Americans, Senseless, That Thing You Do!, Withnail and "
    "I, and My Favorite Year — comedies. None of these involve keyword matching or "
    "genre tags being parsed; the matches come entirely from cosine similarity in the "
    "384-dimensional space MiniLM-L6-v2 produces. This is the same retrieval primitive "
    "the live S1 hook calls every time a domain agent runs."
)


# ─────────── 5. Lessons Learned

doc.add_page_break()
h1("5. Lessons Learned and Takeaways")

h2("5.1 Cold Start")
para(
    "Pure collaborative filtering produces noisy recommendations for users with fewer than "
    "roughly twenty ratings. The joint multi-domain architecture mitigates this for users "
    "active in multiple domains because the shared user embedding picks up signal from any "
    "one of them. The semantic retrieval layer (S1) addresses cold start more directly: a "
    "user with no interaction history at all still receives sensible picks because the "
    "onboarding answers and the natural-language intent are embedded into the same space as "
    "the items. This is the cleanest empirical case for why hybrid content-plus-collaborative "
    "architectures matter in production."
)

h2("5.2 Scalability")
para(
    "Joint MF scales linearly in training time with interaction count, and the trained "
    "checkpoint fits in under 100 MB at the embedding dimensions used. FAISS approximate "
    "nearest-neighbor search is sublinear in catalog size, so the cross-domain semantic "
    "layer in S1 improves scalability for cross-domain queries relative to running three "
    "independent collaborative filters and merging their outputs. Per-user cluster fitting "
    "in S2 is offline and embarrassingly parallel across users — it adds no inference cost."
)

h2("5.3 Evaluation Methodology")
para(
    "Offline metrics against held-out ratings cannot fairly grade a discovery-oriented "
    "recommender. The held-out items are by definition things the user has already "
    "interacted with, and popularity dominates that distribution. Coverage, novelty, and "
    "diversity are necessary complements to NDCG and recall when the system's purpose is to "
    "expand what the user sees rather than predict what they already like. The four-way "
    "comparison in Section 4 only tells the story it does because all five metrics are "
    "reported together."
)

h2("5.4 LLM Orchestration")
para(
    "LangGraph is the right level of the stack when the pipeline is a stateful graph rather "
    "than a linear chain. Multiple agents share and mutate an AgentState; follow-ups route "
    "conditionally based on the multi-turn classifier; the orchestrator synthesizes across "
    "agent outputs. A pure LangChain LCEL implementation would have required manual dict-"
    "threading between calls and bespoke conditional routing. Adding a new agent — a music "
    "agent or a fitness agent — is now a one-file addition rather than a refactor."
)


# ─────────── 6. Source Code

doc.add_page_break()
h1("6. Source Code")
para("Public GitHub repository: https://github.com/AdamSabry1233/AI-Life-Recommender-")
para("Baseline live demo on Hugging Face Spaces: https://huggingface.co/spaces/asabry1233/ai-recommender")
placeholder("Augmented live demo URL — if deployed before submission")


# ─────────── 7. Contribution Table

h1("7. Contribution Table")
fig(FIGS / "contributions.png", "Module map showing primary authorship per file/module.", width_in=6.6)
make_table([
    ["Member",          "Primary responsibilities",                                                  "Specific deliverables"],
    ["Adam Sabry",      "System architecture and primary implementation",
     "Joint multi-domain MF; LangGraph agent pipeline; FAISS session memory; Upstash Redis "
     "profile persistence; Gradio frontend; FastAPI backend; baseline HF Spaces deployment; "
     "initial proposal and documentation."],
    ["Rammy Baroudi",   "Data pipeline and evaluation methodology",
     "Dataset acquisition and preprocessing across MovieLens, Food.com, and Amazon Books; "
     "train-test split design; sparsity analysis; report writing on data description and "
     "lessons learned; presentation slides for data and evaluation sections."],
    ["Stephen Taylor",  "Novelty contributions, evaluation infrastructure, baselines",
     "Cornac baseline sweep (MF/PMF/NMF/BPR) with grouped-metric visualization; semantic "
     "cross-domain retrieval module (S1); per-user cluster conditioning module (S2); four-way "
     "comparison harness and chart; pytest test suite with central delegator and watch/diff-"
     "based reruns; augmented HF Space; report sections on methodology novelty and evaluation."],
], col_widths=[1.4, 2.0, 4.2])

para("")
placeholder("Confirm allocation with the team before final submission")


# save
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"Wrote {OUT}")
