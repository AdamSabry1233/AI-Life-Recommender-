"""
Shared pytest fixtures and path setup.

The project's source modules use bare imports like `from agents.state import AgentState`,
which requires `src/` to be on sys.path. We inject it here so tests can import the
same way the app does, without touching PYTHONPATH in the developer's shell.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def fake_llm():
    """
    Minimal LangChain-Runnable-like stub. Tests that want deterministic LLM output
    should override .invoke on the returned mock.
    """
    llm = MagicMock()
    llm.invoke.return_value = "Habit: Go for a walk — it matches your low-energy intent."
    return llm


@pytest.fixture
def sample_agent_state():
    """A plausible AgentState dict for agents that read intent + preferences."""
    return {
        "intent": "I want to unwind but stay productive",
        "user_preferences": "genres: sci-fi; dietary: vegetarian",
        "user_local_idx": 0,
        "messages": [],
    }
