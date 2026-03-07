"""CLI module wiring tests."""

import main


def test_main_exposes_pd_and_aflmodel_globals():
    # Commands like backtest/update reference these names directly.
    assert hasattr(main, "pd")
    assert hasattr(main, "AFLModel")
