"""
Stub unit tests for src/engine.py — ChatEngine multi-turn classification routing.

Real tests will verify that follow-up messages are correctly classified into
{question, refinement, concern, new_intent} and dispatched to the right handler,
using a mocked LLM so no network calls happen.
"""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.engine]


@pytest.mark.skip(reason="pending: ChatEngine test harness with mocked LLM + UserStore")
def test_classify_question_routes_to_answer_question():
    pass


@pytest.mark.skip(reason="pending")
def test_classify_refinement_swaps_single_recommendation():
    pass


@pytest.mark.skip(reason="pending")
def test_classify_new_intent_saves_prior_session_to_faiss():
    pass
