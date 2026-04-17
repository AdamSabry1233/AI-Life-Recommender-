"""Smoke tests for Cornac baseline sweep."""
import pytest

from baselines.cornac_eval import run

pytestmark = [pytest.mark.baselines, pytest.mark.slow]


def test_run_produces_four_models_and_five_metrics():
    df = run(variant="100K")
    assert list(df.index) == ["MF", "PMF", "NMF", "BPR"]
    assert {"RMSE", "MAE", "Precision@10", "Recall@10", "NDCG@10"} <= set(df.columns)
    assert (df[["RMSE", "MAE"]] >= 0).all().all()


def test_csv_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(variant="100K")
    assert (tmp_path / "results" / "cornac_metrics.csv").exists()
