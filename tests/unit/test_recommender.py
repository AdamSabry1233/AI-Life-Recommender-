"""
Stub unit tests for src/recommender.py (MF inference wrapper).

These are placeholders until a tiny fake checkpoint + lookup CSVs are added under
tests/fixtures/. The real tests will verify:
  - get_top_n returns n items in descending score order
  - item metadata is joined correctly (name, tags)
  - domain_id routing uses the right user_offset / item_offset
  - results are deterministic across calls on the same seeded model
"""

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.recommender]


@pytest.mark.skip(reason="pending: fixtures/mini_checkpoint.pt + lookup CSVs")
def test_get_top_n_returns_n_items_sorted_desc():
    pass


@pytest.mark.skip(reason="pending: deterministic fixture model")
def test_get_top_n_is_deterministic():
    pass


@pytest.mark.skip(reason="pending: fixture lookup CSVs")
def test_missing_lookup_file_raises():
    pass
