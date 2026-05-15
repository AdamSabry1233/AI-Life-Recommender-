"""Unit tests for the live S1/S2 hooks used inside the agents."""
from __future__ import annotations
import os

import pytest

pytestmark = [pytest.mark.unit]

CANDIDATES = [
    {"name": "Action Hero",    "tags": "action thriller", "score": 0.9},
    {"name": "Slow Burn",      "tags": "drama",           "score": 0.7},
    {"name": "Sci-Fi Voyage",  "tags": "sci-fi space",    "score": 0.5},
    {"name": "Cooking Light",  "tags": "documentary food", "score": 0.3},
]


def test_s1_disabled_by_default(monkeypatch):
    monkeypatch.delenv("USE_SEMANTIC_RETRIEVAL", raising=False)
    from semantic import live
    assert not live.enabled()


def test_s1_enabled_via_env(monkeypatch):
    monkeypatch.setenv("USE_SEMANTIC_RETRIEVAL", "1")
    from semantic import live
    assert live.enabled()


def test_s1_empty_query_passes_through():
    from semantic import live
    out = live.augment(CANDIDATES, "")
    assert out == CANDIDATES


def test_s1_empty_candidates_returns_empty():
    from semantic import live
    out = live.augment([], "energetic action movie")
    assert out == []


def test_s2_disabled_by_default(monkeypatch):
    monkeypatch.delenv("USE_CLUSTER_CONDITIONING", raising=False)
    from clusters import live
    assert not live.enabled()


def test_s2_no_prefs_passes_through():
    from clusters import live
    out, debug = live.condition(CANDIDATES, "", "feeling tired")
    assert out == CANDIDATES
    assert debug["active"] is None


def test_s2_single_facet_passes_through():
    from clusters import live
    out, debug = live.condition(CANDIDATES, "action", "feeling tired")
    assert out == CANDIDATES
    assert debug["active"] is None


def test_s2_split_facets():
    from clusters.live import _split_facets
    assert _split_facets("action, drama; sci-fi") == ["action", "drama", "sci-fi"]
    assert _split_facets("") == []
    assert _split_facets("only one") == ["only one"]
    assert _split_facets("a, b, c, d, e") == ["a", "b", "c", "d"]  # capped at 4
