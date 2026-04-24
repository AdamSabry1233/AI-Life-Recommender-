"""Side-by-side comparison chart. Run: python -m eval.visualize_compare [csv_name]"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results"

LOWER_BETTER: set[str] = set()  # none of NDCG/Recall/Diversity/Novelty/Coverage are lower-better


def plot(csv_name: str = "compare_s1.csv") -> Path:
    df = pd.read_csv(OUT / csv_name, index_col=0)
    ncols = len(df.columns)
    max_label = max(len(s) for s in df.index.astype(str))
    nrows_cfg = len(df.index)
    col_width = max(2.8, 0.18 * max_label * nrows_cfg / 4 + 2.2)
    rotate = 0 if max_label <= 6 else 30
    fig, axes = plt.subplots(1, ncols, figsize=(col_width * ncols, 4.6))
    if ncols == 1:
        axes = [axes]
    for ax, col in zip(axes, df.columns):
        best = df[col].min() if col in LOWER_BETTER else df[col].max()
        colors = ["#2ca02c" if v == best else "#4c78a8" for v in df[col]]
        df[col].plot.bar(ax=ax, color=colors)
        ax.set_title(col); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=rotate)
        if rotate:
            for lbl in ax.get_xticklabels():
                lbl.set_horizontalalignment("right")
    fig.suptitle(f"{csv_name}  (green = best)", y=1.02)
    fig.tight_layout()
    out = OUT / csv_name.replace(".csv", ".png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(plot(sys.argv[1] if len(sys.argv) > 1 else "compare_s1.csv"))
