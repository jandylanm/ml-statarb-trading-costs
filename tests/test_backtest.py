"""Tests for study periods and the timing of the long-short returns."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import long_short_returns
from src.features import study_periods


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
