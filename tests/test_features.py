"""Tests for features and labels on small made-up data."""

import numpy as np
import pandas as pd
import pytest

from src.features import lagged_returns, median_labels, period_dataset


def test_lagged_returns_definition():
    ret = np.array([[0.01], [0.02], [-0.01], [0.03]])
    out = lagged_returns(ret, [1, 2])
    assert out[2, 0, 1] == pytest.approx(1.02 * 0.99 - 1)  # R(2, 2) uses the returns of days 1 and 2
    assert out[3, 0, 0] == pytest.approx(0.03)
    assert np.isnan(out[1, 0, 1])                           # not enough history


def test_features_do_not_look_ahead():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, size=(300, 4))
    changed = ret.copy()
    changed[150:] = rng.normal(0, 0.5, size=(150, 4))       # change all returns from day 150 on
    horizons = [1, 5, 20, 240]
    np.testing.assert_array_equal(lagged_returns(ret, horizons)[:150], lagged_returns(changed, horizons)[:150])


def test_labels_strictly_above_median():
    next_ret = np.array([[0.0, 0.0, 0.0, 0.01, -0.01, 0.5]])
    universe = np.array([[True, True, True, True, True, False]])
    labels = median_labels(next_ret, universe)
    np.testing.assert_array_equal(labels[0, :5], [0, 0, 0, 1, 0])  # median 0, ties get label 0
    assert np.isnan(labels[0, 5])                                    # outside the universe


def test_next_day_return_kept_after_index_exit():
    # stock 0 leaves the index after day 899; its last return (e.g., a delisting return) is on day 900
    dates = pd.bdate_range("2000-01-03", periods=1100)
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.02, size=(1100, 3)), index=dates)
    member = pd.DataFrame(True, index=dates, columns=ret.columns)
    member.iloc[900:, 0] = False
    ret.iloc[901:, 0] = np.nan

    period = {"index": 0, "start": 0, "train_end": 750, "end": 1000}
    train, trade = period_dataset(ret, member, period, [1, 2, 240])

    assert train["date"].min() == dates[240] and trade["date"].min() == dates[750]
    last = trade[trade["id"] == 0].iloc[-1]
    assert last["date"] == dates[899]
    assert last["ret_t1"] == pytest.approx(ret.iloc[900, 0])
