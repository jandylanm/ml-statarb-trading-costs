"""Tests for study periods and the timing of the long-short returns."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import buffer_weights, long_short_returns, portfolio_weights, weighted_returns
from src.costs import turnover
from src.features import study_periods


def random_predictions(n_days: int = 60, n_stocks: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic predictions: random scores and next-day returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    pred = pd.DataFrame([(d, i) for d in dates for i in range(n_stocks)], columns=["date", "id"])
    pred["score"] = rng.random(len(pred))
    pred["ret_t1"] = rng.normal(0, 0.02, len(pred))
    return pred


def test_trading_windows_do_not_overlap():
    periods = study_periods(n_days=2100, train_days=750, trade_days=250)
    for a, b in zip(periods, periods[1:]):
        assert a["end"] == b["train_end"]
    assert periods[-1]["end"] == 2100   # the last trading window is shorter


def test_portfolio_earns_return_of_next_day_or_day_after():
    pred = pd.DataFrame({
        "date": pd.Timestamp("2020-01-02"),
        "id": range(6),
        "score": [0.9, 0.8, 0.5, 0.4, 0.2, 0.1],
        "ret_t1": [0.02, 0.01, 0.0, 0.01, -0.01, -0.03],
        "ret_t2": [0.01, 0.0, 0.0, 0.0, 0.0, 0.01],
    })
    lag0 = long_short_returns(pred, "score", k=2, lag=0).iloc[0]
    lag1 = long_short_returns(pred, "score", k=2, lag=1).iloc[0]
    assert lag0 == pytest.approx((0.02 + 0.01) / 2 - (-0.01 - 0.03) / 2)
    assert lag1 == pytest.approx((0.01 + 0.0) / 2 - (0.0 + 0.01) / 2)


def test_buffer_of_k_gives_baseline():
    pred = random_predictions()
    weights = buffer_weights(pred, "score", k=5, buffer=5)
    pd.testing.assert_frame_equal(weights, portfolio_weights(pred, "score", k=5))
    returns = pred.pivot(index="date", columns="id", values="ret_t1")
    pd.testing.assert_series_equal(weighted_returns(weights, returns),
                                   long_short_returns(pred, "score", k=5), check_names=False)


def test_buffer_keeps_k_positions_and_lowers_turnover():
    pred = random_predictions()
    weights = buffer_weights(pred, "score", k=5, buffer=10)
    assert ((weights > 0).sum(axis=1) == 5).all()
    assert ((weights < 0).sum(axis=1) == 5).all()
    assert turnover(weights).mean() < turnover(portfolio_weights(pred, "score", k=5)).mean()
