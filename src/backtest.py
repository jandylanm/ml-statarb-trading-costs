"""Long-short portfolio and performance measures.

Each day, the stocks are sorted by a score (e.g., the predicted probability).
The strategy buys the k stocks with the highest score and sells the k stocks
with the lowest score short, both legs equally weighted.

Timing: the signal uses information up to the close of day t. With lag 0 the
portfolio is formed at that close and earns the return of day t+1 (as in the
paper); with lag 1 it is formed one day later and earns the return of day t+2.
"""

import numpy as np
import pandas as pd


def rank_by_score(pred: pd.DataFrame, score: str, seed: int = 0) -> pd.DataFrame:
    """Add the rank of each stock per day (0 = highest score) and the number of stocks per day.

    Ties are broken randomly. Breaking them by stock id would put the same
    stocks into the long leg on every day with ties (many zero returns in the
    1990s give many ties for the reversal score).
    """
    df = pred.sort_values(["date", "id"]).reset_index(drop=True)
    df["tiebreak"] = np.random.default_rng(seed).random(len(df))
    df = df.sort_values(["date", score, "tiebreak"], ascending=[True, False, True])
    df["rank"] = df.groupby("date").cumcount()
    df["n"] = df.groupby("date")["id"].transform("size")
    return df


def long_short_returns(pred: pd.DataFrame, score: str, k: int, lag: int = 0) -> pd.Series:
    """Daily return: mean return of the top k minus mean return of the bottom k."""
    df = rank_by_score(pred, score)
    ret = df[f"ret_t{lag + 1}"].fillna(0)  # a held stock without return (rare delisting) counts as 0
    long = ret[df["rank"] < k].groupby(df["date"]).mean()
    short = ret[df["rank"] >= df["n"] - k].groupby(df["date"]).mean()
    return long - short


def hit_rate(pred: pd.DataFrame, score: str, k: int) -> float:
    """Share of held positions on the right side of the median (long: y = 1, short: y = 0)."""
    df = rank_by_score(pred, score)
    long = df[df["rank"] < k]
    short = df[df["rank"] >= df["n"] - k]
    correct = (long["y"] == 1).sum() + (short["y"] == 0).sum()
    labeled = long["y"].notna().sum() + short["y"].notna().sum()
    return float(correct / labeled)


def performance(returns: pd.DataFrame, days_per_year: int = 252) -> pd.DataFrame:
    """Mean daily return, t-statistic, annualized return and volatility, Sharpe ratio, max drawdown.

    The long-short portfolio is self-financing, so the Sharpe ratio uses the
    raw returns (no risk-free rate).
    """
    rows = {}
    for name, r in returns.items():
        r = r.dropna()
        wealth = (1 + r).cumprod()
        rows[name] = {
            "mean_daily": r.mean(),
            "t_stat": r.mean() / r.std() * np.sqrt(len(r)),
            "ann_return": wealth.iloc[-1] ** (days_per_year / len(r)) - 1,
            "ann_vol": r.std() * np.sqrt(days_per_year),
            "sharpe": r.mean() / r.std() * np.sqrt(days_per_year),
            "max_drawdown": (1 - wealth / wealth.cummax()).max(),
        }
    return pd.DataFrame(rows).T
