"""
Stub integration tests for the full LangGraph pipeline (src/graph.py).

Real tests will run retrieve_memory → entertainment → food → learning → habit →
orchestrator end-to-end with a mocked LLM and a fixture MF recommender, asserting
all four rec slots are populated and the orchestrator produces a non-empty summary.
"""

import pytest

pytestmark = [pytest.mark.integration]


@pytest.mark.skip(reason="pending: full graph harness with fixture recommender + fake LLM")
def test_full_pipeline_populates_all_four_recs():
    pass


@pytest.mark.skip(reason="pending")
def test_retrieve_memory_returns_empty_on_fresh_session():
    pass
