"""MovieLens 100K loader with item metadata (titles + genre tags)."""
from __future__ import annotations
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cornac" / "ml-100k"
URL_ITEM = "http://files.grouplens.org/datasets/movielens/ml-100k/u.item"
URL_DATA = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery",
    "Romance", "SciFi", "Thriller", "War", "Western",
]


def _ensure(url: str, path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)


def load_ratings() -> pd.DataFrame:
    _ensure(URL_DATA, CACHE / "u.data")
    return pd.read_csv(
        CACHE / "u.data", sep="\t", names=["user", "item", "rating", "ts"],
        dtype={"user": np.int64, "item": np.int64, "rating": np.float32},
    )


def load_items() -> pd.DataFrame:
    """Return DataFrame with columns: item, title, text (title + genre tags)."""
    _ensure(URL_ITEM, CACHE / "u.item")
    cols = ["item", "title", "release", "vrelease", "url"] + GENRES
    df = pd.read_csv(CACHE / "u.item", sep="|", names=cols, encoding="latin-1")
    genre_cols = df[GENRES].astype(bool)
    df["genres"] = genre_cols.apply(lambda r: " ".join(g for g in GENRES if r[g]), axis=1)
    df["text"] = df["title"] + " " + df["genres"]
    return df[["item", "title", "text"]]
