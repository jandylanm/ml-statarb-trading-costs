"""Study periods, features (lagged returns) and labels, following Krauss et al. (2017).

    R(t, m) = P(t) / P(t - m) - 1   return over the last m trading days (feature)
    y(t)    = 1 if r(t + 1) > median of r(t + 1) over the stocks of day t (label)

The features of day t only use returns up to and including day t.
"""

import warnings

import numpy as np
import pandas as pd


def study_periods(n_days: int, train_days: int, trade_days: int) -> list[dict]:
    """Rolling periods of train_days + trade_days, shifted by trade_days.

    The trading windows follow each other without overlap. The last period may
    have a shorter trading window. Positions refer to the sorted trading days;
    as in Python slicing, `train_end` and `end` are exclusive.
    """
    periods = []
    start = 0
    while start + train_days < n_days:
        end = min(start + train_days + trade_days, n_days)
        periods.append({"index": len(periods), "start": start, "train_end": start + train_days, "end": end})
        start += trade_days
    return periods


def to_wide(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns and index membership as matrices (rows = trading days, columns = stocks)."""
    ret = panel.pivot(index="date", columns="id", values="ret").sort_index()
    member = panel.pivot(index="date", columns="id", values="in_sp500").reindex_like(ret).eq(True)
    return ret, member


def lagged_returns(ret: np.ndarray, horizons: list[int]) -> np.ndarray:
    """R(t, m) for all days, stocks and horizons, shape (days, stocks, horizons).

    Uses cumulated log returns: R(t, m) = exp(L(t) - L(t - m)) - 1. The value is
    NaN if one of the m daily returns is missing or if t < m.
    """
    # missing returns count as 0 here (those windows are set to NaN below);
    # the floor avoids log(0) for a return of exactly -100 %
    log_price = np.cumsum(np.log1p(np.maximum(np.nan_to_num(ret), -0.9999)), axis=0)
    n_valid = np.cumsum(~np.isnan(ret), axis=0)

    out = np.full(ret.shape + (len(horizons),), np.nan, dtype=np.float32)
    for h, m in enumerate(horizons):
        complete = (n_valid[m:] - n_valid[:-m]) == m
        out[m:, :, h] = np.where(complete, np.expm1(log_price[m:] - log_price[:-m]), np.nan)
    return out


def median_labels(next_ret: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """1 if next_ret is strictly above the median of the day's universe, 0 if not, NaN outside."""
    usable = universe & ~np.isnan(next_ret)
    with warnings.catch_warnings():  # days without any stock in the universe give NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(usable, next_ret, np.nan), axis=1, keepdims=True)
    labels = (next_ret > median).astype(float)
    labels[~usable] = np.nan
    return labels


def period_dataset(ret_wide: pd.DataFrame, member_wide: pd.DataFrame, period: dict,
                   horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and trading data of one study period (one row per stock and day).

    As in the paper, the features are computed from the returns inside the
    period, so the first max(horizons) = 240 days are only used as look-back.
    A stock is in the universe of day t if it is an index member on day t and
    all features exist. ret_t1 and ret_t2 are the returns of days t+1 and t+2.
    They are taken from all trading days, also after a stock left the index,
    so that delisting returns are not lost.
    """
    start, train_end, end = period["start"], period["train_end"], period["end"]
    members = member_wide.iloc[start:end]
    stocks = members.columns[members.any().to_numpy()]  # stocks with at least one member day
    ret = ret_wide[stocks].to_numpy()

    features = lagged_returns(ret[start:end], horizons)
    ret_t1 = np.full((end - start, len(stocks)), np.nan)
    ret_t2 = np.full((end - start, len(stocks)), np.nan)
    n = min(end + 1, len(ret)) - (start + 1)         # days for which t+1 exists in the data
    ret_t1[:n] = ret[start + 1:start + 1 + n]
    n = min(end + 2, len(ret)) - (start + 2)
    ret_t2[:n] = ret[start + 2:start + 2 + n]

    universe = members[stocks].to_numpy() & ~np.isnan(features).any(axis=2)
    labels = median_labels(ret_t1, universe)

    day, stock = np.nonzero(universe)
    data = pd.DataFrame(features[day, stock], columns=[f"R{m}" for m in horizons])
    data.insert(0, "date", ret_wide.index[start + day])
    data.insert(1, "id", stocks[stock])
    data["y"] = labels[day, stock]
    data["ret_t1"] = ret_t1[day, stock]
    data["ret_t2"] = ret_t2[day, stock]

    in_training = (day >= max(horizons)) & (day < train_end - start)
    train = data[in_training & data["y"].notna()].reset_index(drop=True)
    trade = data[day >= train_end - start].reset_index(drop=True)
    return train, trade
