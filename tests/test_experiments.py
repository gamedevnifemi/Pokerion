"""The experiments take minutes at full size. These run them tiny, to prove they still run."""

import pytest

from experiments import safe_exploitation_kuhn as study


@pytest.fixture(scope="module")
def rows():
    return study.run(opponents=2, hands=10, seed=0)


def test_safe_exploitation_reports_every_opponent_class(rows):
    assert [row.opponents for row in rows] == ["random", "sophisticated static", "equilibrium"]
    for row in rows:
        assert set(row.best_response) == set(study.TIE_BREAKS)
        assert row.published == study.PUBLISHED[row.opponents]


def test_nothing_beats_the_game_value_against_the_equilibrium(rows):
    # The first player's value in Kuhn is -1/18, whatever it tries against equilibrium play.
    equilibrium = rows[2]
    assert equilibrium.core == pytest.approx(-1 / 18, abs=1e-3)
    for mean, _ in equilibrium.best_response.values():
        assert mean == pytest.approx(-1 / 18, abs=1e-3)


def test_report_has_one_line_per_class_and_names_the_tie_breaks(rows):
    text = study.report(rows)
    assert len(text.splitlines()) == 1 + len(rows)
    for label in study.TIE_BREAKS:
        assert label in text
