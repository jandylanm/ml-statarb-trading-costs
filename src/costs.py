"""Trading costs: Corwin-Schultz spread estimates, turnover and net returns."""

import numpy as np
import pandas as pd


def two_day_spread(high_prev: np.ndarray, low_prev: np.ndarray, high: np.ndarray,
                   low: np.ndarray, close_prev: np.ndarray) -> np.ndarray:
    """Corwin-Schultz (2012) spread estimate from two consecutive days.

    All inputs are arrays of the same length; element i describes one stock on
    days t-1 (high_prev, low_prev, close_prev) and t (high, low).

    Steps:
    1. Overnight adjustment: if the low of day t is above the close of day t-1,
       subtract (low - close_prev) from the high and low of day t; if the high of
       day t is below the close of day t-1, add (close_prev - high) to both.
    2. beta  = ln(H_{t-1}/L_{t-1})^2 + ln(H_t/L_t)^2
       gamma = ln(H_{t-1,t}/L_{t-1,t})^2, with the high and low over both days
    3. alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt(2)) - sqrt(gamma / (3 - 2 sqrt(2)))
    4. S = 2 (exp(alpha) - 1) / (1 + exp(alpha))

    Returns the raw estimate S, which can be negative. Negative values are not
    set to zero here: the estimates are first averaged over a rolling window,
    and only the average is floored (see rolling_spread).
    """
    # 1. remove overnight jumps from the prices of day t
    gap = np.where(low > close_prev, low - close_prev, np.where(high < close_prev, high - close_prev, 0.0))
    high = high - gap
    low = low - gap

    # 2. one-day ranges (beta) and the two-day range (gamma)
    beta = np.log(high_prev / low_prev) ** 2 + np.log(high / low) ** 2
    gamma = np.log(np.maximum(high_prev, high) / np.minimum(low_prev, low)) ** 2

    # 3. and 4. spread component of the range and the implied spread
    c = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / c - np.sqrt(gamma / c)
    return 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))


def tick_size(dates: pd.Series, ticks: dict) -> pd.Series:
    """Minimum price increment in USD on each date.

    `ticks` maps the first date of each tick regime to its size, e.g.
    {"1990-01-01": 0.125, "1997-06-02": 0.0625, "2001-01-29": 0.01}.
    """
    tick = pd.Series(np.nan, index=dates.index)
    for start, size in sorted(ticks.items()):  # later regimes overwrite earlier ones
        tick[dates >= pd.Timestamp(start)] = size
    return tick


def rolling_spread(panel: pd.DataFrame, window: int, min_obs: int, ticks: dict) -> pd.DataFrame:
    """Spread estimate per stock and day: mean of the last `window` two-day estimates, floored at one tick.

    Returns the columns id, date, raw (two-day estimate of days t-1 and t, for
    diagnostics), unfloored (the rolling mean before any flooring, used to sort
    stocks by liquidity) and spread (the estimate used for the costs).

    The estimate for day t uses the days t-1 and t of each pair, so it is known at
    the close of day t. Pairs with a split between the two days (change in the
    cumulative price factor) are skipped, because their prices are not comparable.

    A quoted spread cannot be smaller than one tick, so the rolling mean is
    floored at tick / close. With a missing close the floor is zero.
    """
    df = panel[["id", "date", "high", "low", "close", "cumfacpr"]].sort_values(["id", "date"])
    prev = df.groupby("id")[["high", "low", "close", "cumfacpr"]].shift(1)

    raw = two_day_spread(prev["high"].to_numpy(), prev["low"].to_numpy(), df["high"].to_numpy(),
                         df["low"].to_numpy(), prev["close"].to_numpy())
    raw[(df["cumfacpr"] != prev["cumfacpr"]).to_numpy()] = np.nan
    df["raw"] = raw

    mean = df.groupby("id")["raw"].rolling(window, min_periods=min_obs).mean()
    df["unfloored"] = mean.reset_index(level=0, drop=True)

    floor = (tick_size(df["date"], ticks) / df["close"]).fillna(0)
    df["spread"] = df["unfloored"].clip(lower=floor)   # a missing rolling mean stays missing
    return df[["id", "date", "raw", "unfloored", "spread"]]


def trades(weights: pd.DataFrame) -> pd.DataFrame:
    """Traded amount per stock and day, |w(t) - w(t-1)| (on the first day the whole portfolio is bought)."""
    return weights.diff().abs().fillna(weights.abs())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily turnover: sum over all stocks of |w(t) - w(t-1)| (weights: dates x stocks)."""
    return trades(weights).sum(axis=1)


def breakeven_spread(gross: pd.Series, turnover: pd.Series) -> float:
    """Spread at which the costs would eat the whole gross return.

    Every unit traded costs half the spread, so the costs equal the gross
    return when spread / 2 * sum(turnover) = sum(gross). Both series must
    cover the same days.
    """
    return 2 * gross.sum() / turnover.sum()


def spread_costs(weights: pd.DataFrame, spreads: pd.DataFrame) -> pd.Series:
    """Daily costs: half the estimated spread for every unit traded, sum over all stocks.

    `spreads` has the same shape as `weights` and contains the spread estimate
    of the trading day. A missing estimate (e.g., in the first days after a
    listing) is replaced by the median spread of all stocks on that day.
    """
    daily_median = spreads.median(axis=1)
    spreads = spreads.T.fillna(daily_median).T   # fill each day's gaps with that day's median
    return (spreads / 2 * trades(weights)).sum(axis=1)
