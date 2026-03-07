"""Tests for data cleaning edge cases."""

import pandas as pd

from src.preprocessing.clean import clean_squiggle_games


def test_incomplete_matches_have_null_targets():
    raw = pd.DataFrame(
        [
            {
                "id": 1,
                "year": 2026,
                "round": 1,
                "date": "2026-03-20T19:30:00+10:00",
                "hteam": "Collingwood",
                "ateam": "Carlton",
                "hscore": 0,
                "ascore": 0,
                "complete": 0,
            },
            {
                "id": 2,
                "year": 2026,
                "round": 1,
                "date": "2026-03-21T13:00:00+10:00",
                "hteam": "Richmond",
                "ateam": "Essendon",
                "hscore": 90,
                "ascore": 80,
                "complete": 100,
            },
        ]
    )

    cleaned = clean_squiggle_games(raw)
    incomplete = cleaned.loc[cleaned["match_id"] == 1].iloc[0]
    complete = cleaned.loc[cleaned["match_id"] == 2].iloc[0]

    assert bool(incomplete["is_complete"]) is False
    assert pd.isna(incomplete["margin"])
    assert pd.isna(incomplete["home_win"])
    assert pd.isna(incomplete["total_score"])

    assert bool(complete["is_complete"]) is True
    assert complete["margin"] == 10
    assert complete["home_win"] == 1
    assert complete["total_score"] == 170
