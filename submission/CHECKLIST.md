# Submission Checklist — CMPE 256 Term Project

Per the rubric (PDF, Section 6).

## A. Canvas Submission

- [ ] `LifeRecommender_FinalReport.pdf` — **draft generated** as `.docx`; export to PDF before submission. [PLACEHOLDER — final EDA visualizations, sparsity heatmap, system architecture diagram, live demo screenshots]
- [ ] `LifeRecommender_Slides.pptx` — **draft generated**; fill placeholder blocks before presentation
- [ ] GitHub URL — `https://github.com/AdamSabry1233/AI-Life-Recommender-` ready
- [ ] GitHub URL pasted into Canvas comment section AND inside the report

## B. GitHub Repository

- [x] Public repo
- [x] Source code in repo (Python scripts under `src/`)
- [x] `README.md` — exists; [PLACEHOLDER — add Stephen's contribution section pointing to `src/baselines/`, `src/semantic/`, `src/clusters/`, `src/eval/`, `tools/test_runner.py`]
- [x] `requirements.txt` — present
- [ ] [PLACEHOLDER — verify "How to Run" steps work end-to-end on a clean clone]

## C. In-Class Deliverable

- [ ] **Oral presentation** — 15 minutes, all three members speak
- [ ] **Live system demo** — baseline HF Space and augmented HF Space side-by-side
- [ ] **Backup demo video** — 2-minute recording on USB/cloud [PLACEHOLDER — record once augmented HF Space is up]

## Rubric Coverage Notes

| Category | Weight | Coverage in this submission |
|---|---|---|
| Novelty & Originality | 20% | Hybrid joint-MF + LLM orchestration + semantic retrieval (S1) + per-user cluster conditioning (S2); cross-domain agentic design. Maps to "creative hybrid architectures, LLMs, advanced techniques" in the rubric. |
| Oral Presentation & Demo | 20% | Slides drafted with 18 slides for the 15-minute slot. [PLACEHOLDER — rehearse with team and time it.] |
| Technical Implementation | 25% | All code in repo, modular under `src/{agents,baselines,semantic,clusters,eval}/`, test suite in place. Edge cases: cold-start (S1 profile fallback), short user history (S2 falls back to MF), missing fixtures (graceful test skip). |
| Evaluation & Benchmarking | 15% | Cornac sweep (4 models) + 4-way comparison (MF / +S1 / +S2 / +both) on five metrics including coverage/diversity/novelty. Explicit reasoning for metric choice in the report. |
| Report Quality & EDA | 10% | Report drafted; [PLACEHOLDER — generate EDA figures (sparsity heatmap, rating distribution, long-tail plot, per-user activity)]. |
| GitHub & Documentation | 10% | [PLACEHOLDER — README update with Stephen's contribution section; verify reproducibility on clean clone]. |

## Open Risks

1. Live demo failing — backup video required; if HF Space is down, fall back to local CLI run via `python src/chat.py`
2. EDA figures not yet generated — schedule a one-hour session before submission to produce them from the unified interaction matrix
3. Augmented HF Space not yet deployed — Stephen owns this; needs Groq key + the augmented branch built into a Space repo
4. Time allocation across three speakers needs rehearsal — slides 1-5 Adam, 6-10 Rammy, 11-18 Stephen is a defensible split given the contribution table
