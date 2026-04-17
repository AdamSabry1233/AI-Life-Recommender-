"""
Unit tests for src/agents/habit_agent.py — focused on the brittle _parse() helper
that splits LLM output into (habit, reason) using an em-dash delimiter.
"""

import pytest

from agents.habit_agent import _parse

pytestmark = [pytest.mark.unit, pytest.mark.agents]


class TestParseWellFormed:
    def test_standard_em_dash_format(self):
        habit, reason = _parse("Habit: Go for a walk — clears your head")
        assert habit == "Go for a walk"
        assert reason == "clears your head"

    def test_preceded_by_whitespace_and_newlines(self):
        habit, reason = _parse("\n\n   Habit: Drink water — hydration helps focus\n")
        assert habit == "Drink water"
        assert reason == "hydration helps focus"

    def test_case_insensitive_prefix(self):
        habit, _ = _parse("HABIT: Stretch — loosens tension")
        assert habit == "Stretch"


class TestParseDegraded:
    def test_missing_em_dash_returns_empty_reason(self):
        habit, reason = _parse("Habit: Meditate for 5 minutes")
        assert habit == "Meditate for 5 minutes"
        assert reason == ""

    def test_hyphen_instead_of_em_dash_is_not_split(self):
        """The parser only splits on ' — ' (em-dash), not '-'. Documents current behavior."""
        habit, reason = _parse("Habit: Meditate - five minutes")
        assert habit == "Meditate - five minutes"
        assert reason == ""

    def test_no_habit_prefix_returns_unknown(self):
        habit, reason = _parse("Sure! I think you should go for a walk today.")
        assert habit == "Unknown"
        assert reason == "Sure! I think you should go for a walk today."

    def test_empty_input(self):
        habit, reason = _parse("")
        assert habit == "Unknown"
        assert reason == ""

    def test_multiline_picks_first_habit_line(self):
        text = "Some preamble.\nHabit: Walk — fresh air\nMore text below."
        habit, reason = _parse(text)
        assert habit == "Walk"
        assert reason == "fresh air"
