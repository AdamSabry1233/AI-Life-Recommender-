"""Plot Cornac metrics. Run: python -m baselines.visualize"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results"


GROUPS = [
    ("Error (lower is better)",   ["MAE", "RMSE"]),
    ("Ranking (higher is better)", ["NDCG@10", "Precision@10", "Recall@10"]),
    ("Timing (lower is better)",   ["Train (s)", "Test (s)"]),
]


def plot(csv_path: Path = OUT / "cornac_metrics.csv") -> Path:
    df = pd.read_csv(csv_path, index_col=0)
    ncols = max(len(cols) for _, cols in GROUPS)
    fig, axes = plt.subplots(len(GROUPS), ncols, figsize=(3.2 * ncols, 3.6 * len(GROUPS)))
    for r, (row_title, cols) in enumerate(GROUPS):
        lower_better = "lower" in row_title
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(cols):
                ax.axis("off"); continue
            col = cols[c]
            best = df[col].min() if lower_better else df[col].max()
            colors = ["#2ca02c" if v == best else "#4c78a8" for v in df[col]]
            df[col].plot.bar(ax=ax, color=colors)
            ax.set_title(col); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=0)
            if c == 0:
                ax.set_ylabel(row_title, fontsize=10, labelpad=12)
    fig.suptitle("Cornac baselines — MovieLens 100K (green = best)", y=1.00)
    fig.tight_layout()
    out = OUT / "cornac_metrics.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot())
