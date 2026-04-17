"""Cornac baseline sweep. Run: python -m baselines.cornac_eval"""
from pathlib import Path

import cornac
import pandas as pd
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.metrics import MAE, NDCG, RMSE, Precision, Recall
from cornac.models import BPR, MF, NMF, PMF

OUT = Path(__file__).resolve().parents[2] / "results"
OUT.mkdir(exist_ok=True)


def run(variant: str = "100K", k: int = 10, seed: int = 42) -> pd.DataFrame:
    split = RatioSplit(
        data=movielens.load_feedback(variant=variant),
        test_size=0.2, rating_threshold=4.0, seed=seed, exclude_unknowns=True, verbose=False,
    )
    models = [
        MF(k=k, max_iter=25, learning_rate=0.01, lambda_reg=0.02, seed=seed, verbose=False),
        PMF(k=k, max_iter=50, learning_rate=0.001, seed=seed, verbose=False),
        NMF(k=k, max_iter=50, seed=seed, verbose=False),
        BPR(k=k, max_iter=100, learning_rate=0.01, lambda_reg=0.01, seed=seed, verbose=False),
    ]
    metrics = [RMSE(), MAE(), Precision(k=10), Recall(k=10), NDCG(k=10)]
    exp = cornac.Experiment(eval_method=split, models=models, metrics=metrics, user_based=True, verbose=False)
    exp.run()

    rows = [{"model": r.model_name, **r.metric_avg_results} for r in exp.result]
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(OUT / "cornac_metrics.csv")
    return df


if __name__ == "__main__":
    print(run().round(4).to_string())
