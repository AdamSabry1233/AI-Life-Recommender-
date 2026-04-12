"""
Phase 3 — Recommendation Pipeline

Entry point that ties the MF recommender and LLM reasoning layer together.

Usage:
    python src/pipeline.py
    python src/pipeline.py --intent "I want to relax but still feel productive"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from recommender import MultiDomainRecommender, DOMAIN_MOVIES, DOMAIN_FOOD, DOMAIN_BOOKS
from llm_chain import build_chain, format_candidates


class RecommendationPipeline:
    """
    Parameters
    ----------
    checkpoint_path : str  Path to mf_best.pt
    data_dir        : str  Project data root
    ollama_model    : str  Ollama model tag (must be pulled locally)
    n_candidates    : int  Candidates per domain passed to the LLM
    """

    def __init__(
        self,
        checkpoint_path: str = "mf_best.pt",
        data_dir: str = "data",
        ollama_model: str = "llama3.1",
        n_candidates: int = 10,
    ):
        print("Loading MF model …")
        self.rec          = MultiDomainRecommender(checkpoint_path, data_dir)
        self.chain        = build_chain(model=ollama_model)
        self.n_candidates = n_candidates
        print("Ready.\n")

    def run(self, intent: str, user_local_idx: int = 0) -> str:
        """
        Parameters
        ----------
        intent         : natural-language description of the user's mood/goal
        user_local_idx : demo user index (0 = first user in each domain's set)

        Returns
        -------
        Structured recommendation string from the LLM.
        """
        movies = self.rec.get_top_n(DOMAIN_MOVIES, user_local_idx, self.n_candidates)
        foods  = self.rec.get_top_n(DOMAIN_FOOD,   user_local_idx, self.n_candidates)
        books  = self.rec.get_top_n(DOMAIN_BOOKS,  user_local_idx, self.n_candidates)

        return self.chain.invoke({
            "intent":           intent,
            "movie_candidates": format_candidates(movies),
            "food_candidates":  format_candidates(foods),
            "book_candidates":  format_candidates(books),
        })


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--intent", "-i",
        default="I want to relax but still feel productive",
        help="Natural-language description of the user's current mood or goal",
    )
    p.add_argument("--checkpoint", default="mf_best.pt")
    p.add_argument("--data-dir",   default="data")
    p.add_argument("--model",      default="llama3.1")
    p.add_argument("--n",          type=int, default=10,
                   help="Number of candidates per domain")
    args = p.parse_args()

    pipeline = RecommendationPipeline(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        ollama_model=args.model,
        n_candidates=args.n,
    )

    print(f"Intent: {args.intent}\n")
    print("─" * 60)
    result = pipeline.run(args.intent)
    print(result)
