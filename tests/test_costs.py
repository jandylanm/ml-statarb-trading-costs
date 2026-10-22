"""Tests for the Corwin-Schultz spread estimator, turnover and costs."""

import numpy as np
import pandas as pd
import pytest

from src.costs import breakeven_spread, rolling_spread, spread_costs, tick_size, turnover, two_day_spread


def simulate_prices(n_days: int, spread: float, seed: int = 0) -> tuple:
    """Daily high, low and close of a stock with a known bid-ask spread.

    The efficient log price is a random walk (390 steps per day, 1.5 % daily
    volatility, 0.5 % overnight moves). Every trade happens at the ask or the bid
    with equal probability, i.e., at the efficient price times (1 +- spread / 2).
    """
    rng = np.random.default_rng(seed)
    steps = 390
    intraday = np.cumsum(rng.normal(0, 0.015 / np.sqrt(steps), (n_days, steps)), axis=1)
    start = np.cumsum(intraday[:, -1]) - intraday[:, -1] + np.cumsum(rng.normal(0, 0.005, n_days))
    price = 100 * np.exp(intraday + start[:, None])
    trades = price * (1 + rng.choice([-1, 1], size=price.shape) * spread / 2)
    return trades.max(axis=1), trades.min(axis=1), trades[:, -1]


def test_recovers_known_spread():
    true_spread = 0.005
    high, low, close = simulate_prices(2520, true_spread)
    estimates = two_day_spread(high[:-1], low[:-1], high[1:], low[1:], close[:-1])
    # with a finite number of trades per day the estimator is slightly too low
    assert 0.5 * true_spread < estimates.mean() < 1.5 * true_spread


def test_overnight_gap_is_removed():
    # day t opens 4 above the close of day t-1 and never trades below it
    with_gap = two_day_spread(np.array([101.0]), np.array([99.0]), np.array([106.0]),
                              np.array([104.0]), np.array([100.0]))
    # the same day shifted down by the gap
    without_gap = two_day_spread(np.array([101.0]), np.array([99.0]), np.array([102.0]),
                                 np.array([100.0]), np.array([100.0]))
    np.testing.assert_allclose(with_gap, without_gap)


def test_raw_estimate_can_be_negative():
    # small daily ranges but a large two-day range give a negative raw estimate,
    # which must not be set to zero inside two_day_spread
    estimate = two_day_spread(np.array([101.0]), np.array([100.0]), np.array([102.0]),
                              np.array([101.0]), np.array([101.0]))
    assert estimate[0] < 0


def test_turnover_and_spread_costs():
    # k = 1: day 1 long A, short B; day 2 the same; day 3 long C, short D
    weights = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [-1.0, -1.0, 0.0],
                            "C": [0.0, 0.0, 1.0], "D": [0.0, 0.0, -1.0]})
    np.testing.assert_allclose(turnover(weights), [2.0, 0.0, 4.0])  # build, hold, replace everything

    spreads = pd.DataFrame(0.002, index=weights.index, columns=weights.columns)
    costs = spread_costs(weights, spreads)
    assert costs.iloc[2] == pytest.approx(4 * 0.002 / 2)            # four trades at half the spread

    # at the break-even spread, total costs equal the total gross return
    gross = pd.Series([0.01, 0.0, 0.002])
    s = breakeven_spread(gross, turnover(weights))
    assert s / 2 * turnover(weights).sum() == pytest.approx(gross.sum())


def test_tick_size_floor():
    ticks = {"1990-01-01": 0.125, "1997-06-02": 0.0625, "2001-01-29": 0.01}
    dates = pd.Series(pd.to_datetime(["1995-06-01", "1999-06-01", "2010-06-01"]))
    np.testing.assert_allclose(tick_size(dates, ticks), [0.125, 0.0625, 0.01])

    # two days in 1995 whose two-day estimate is negative: the zero floor would
    # give 0, the tick floor gives one eighth relative to the close
    panel = pd.DataFrame({"id": [1, 1], "date": pd.to_datetime(["1995-06-01", "1995-06-02"]),
                          "high": [101.0, 102.0], "low": [100.0, 101.0], "close": [101.0, 101.0],
                          "cumfacpr": [1.0, 1.0]})
    result = rolling_spread(panel, window=21, min_obs=1, ticks=ticks).iloc[-1]
    assert result["unfloored"] < 0
    assert result["spread"] == pytest.approx(0.125 / 101)
