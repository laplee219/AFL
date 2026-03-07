"""Regression tests for pipeline behavior."""

from pathlib import Path

import pandas as pd

import src.pipeline.feedback_loop as feedback_loop
from src.pipeline.feedback_loop import Pipeline
from src.preprocessing.clean import clean_squiggle_games


def test_ingest_results_replaces_existing_fixture_row(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(feedback_loop, "PROCESSED_DATA_DIR", processed_dir)

    fixture_raw = pd.DataFrame(
        [
            {
                "id": 1001,
                "year": 2026,
                "round": 1,
                "date": "2026-03-20T19:30:00+10:00",
                "hteam": "Collingwood",
                "ateam": "Carlton",
                "venue": "MCG",
                "hscore": 0,
                "ascore": 0,
                "complete": 0,
            }
        ]
    )
    fixture = clean_squiggle_games(fixture_raw)
    fixture.to_csv(processed_dir / "matches_all.csv", index=False)

    completed_raw = pd.DataFrame(
        [
            {
                "id": 1001,
                "year": 2026,
                "round": 1,
                "date": "2026-03-20T19:30:00+10:00",
                "hteam": "Collingwood",
                "ateam": "Carlton",
                "venue": "MCG",
                "hscore": 95,
                "ascore": 80,
                "complete": 100,
                "winner": "Collingwood",
            }
        ]
    )

    pipeline = Pipeline()
    pipeline.squiggle.get_completed_games = lambda y, r=None: completed_raw
    pipeline.odds_collector.save_odds_snapshot = lambda *args, **kwargs: 0
    pipeline.tracker.settle_round = lambda *args, **kwargs: None
    pipeline.monitor.check_retrain_needed = (
        lambda *args, **kwargs: {"should_retrain": False, "reason": ""}
    )

    pipeline.ingest_results(year=2026, round_num=1)

    merged = pd.read_csv(processed_dir / "matches_all.csv")
    assert len(merged) == 1
    assert int(merged.iloc[0]["match_id"]) == 1001
    assert bool(merged.iloc[0]["is_complete"]) is True
    assert float(merged.iloc[0]["margin"]) == 15.0
    assert int(merged.iloc[0]["home_score"]) == 95
    assert int(merged.iloc[0]["away_score"]) == 80


def test_get_status_latest_model_uses_sorted_version(monkeypatch):
    class FakeDir:
        def __init__(self, name):
            self.name = name

        def is_dir(self):
            return True

    class FakeModelsDir:
        def exists(self):
            return True

        def iterdir(self):
            # Deliberately unsorted iteration order
            return [
                FakeDir("v_2026-02-10T10-00-00"),
                FakeDir("v_2026-01-01T10-00-00"),
                FakeDir("v_2026-02-22T10-00-00"),
            ]

    monkeypatch.setattr(feedback_loop, "MODELS_DIR", FakeModelsDir())

    pipeline = Pipeline()
    status = pipeline.get_status()

    assert status["n_saved_models"] == 3
    assert status["latest_model"] == "v_2026-02-22T10-00-00"
