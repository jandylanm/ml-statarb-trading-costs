"""Download CRSP data from WRDS and build the stock-day panel.

Only aggregate numbers are printed (WRDS terms: no stock-level data in logs).
"""

import os

import numpy as np
import pandas as pd

DAILY_COLUMNS = ["permno", "dlycaldt", "ticker", "dlyret", "dlyopen", "dlyhigh",
                 "dlylow", "dlyclose", "dlyvol", "dlycumfacpr"]
NUMERIC_COLUMNS = ["dlyret", "dlyopen", "dlyhigh", "dlylow", "dlyclose", "dlyvol", "dlycumfacpr"]
RENAME = {"permno": "id", "dlycaldt": "date", "dlyret": "ret", "dlyopen": "open",
          "dlyhigh": "high", "dlylow": "low", "dlyclose": "close", "dlyvol": "volume",
          "dlycumfacpr": "cumfacpr"}


def download_crsp(cfg: dict, force: bool = False) -> None:
    """Download S&P 500 membership and daily stock data (one file per year).

    Only stocks that were in the index at some point during the sample are
    downloaded, but with all their trading days: the features need up to 240
    days of history before a stock enters the index, and delisting returns are
    often booked on the day after the last membership day.
    """
    import wrds

    raw = cfg["paths"]["raw"] / "crsp"
    raw.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"] or pd.Timestamp.today())
    members_table = cfg["data"]["membership_table"]

    db = wrds.Connection(wrds_username=os.environ[cfg["data"]["wrds_username_env"]])

    path = raw / "membership.parquet"
    if force or not path.exists():
        members = db.raw_sql(f"select * from {members_table}", date_cols=["mbrstartdt", "mbrenddt"])
        members.to_parquet(path)

    ever_member = f"select distinct permno from {members_table} where mbrenddt >= '{start.date()}'"
    for year in range(start.year, end.year + 1):
        path = raw / f"daily_{year}.parquet"
        if path.exists() and not force:
            continue
        sql = (f"select {', '.join(DAILY_COLUMNS)} from {cfg['data']['daily_table']} "
               f"where permno in ({ever_member}) "
               f"and dlycaldt between '{year}-01-01' and '{year}-12-31'")
        daily = db.raw_sql(sql, date_cols=["dlycaldt"])
        if len(daily) > 0:
            daily.to_parquet(path)
        print(f"{year}: {len(daily):,} rows")

    db.close()


def is_member(panel: pd.DataFrame, members: pd.DataFrame) -> np.ndarray:
    """True where the stock is an S&P 500 member on that day (interval bounds included)."""
    intervals = members.rename(columns={"permno": "id"})[["id", "mbrstartdt", "mbrenddt"]]
    intervals = intervals.fillna({"mbrenddt": pd.Timestamp.max})  # open interval = still a member

    # one row per (stock-day, membership interval of that stock)
    rows = panel[["id", "date"]].reset_index().merge(intervals, on="id")
    inside = (rows["date"] >= rows["mbrstartdt"]) & (rows["date"] <= rows["mbrenddt"])

    flag = np.zeros(len(panel), dtype=bool)
    flag[rows.loc[inside, "index"]] = True
    return flag


def build_panel(cfg: dict) -> pd.DataFrame:
    """Combine the yearly files into one panel with one row per stock and trading day."""
    raw = cfg["paths"]["raw"] / "crsp"
    files = sorted(raw.glob("daily_*.parquet"))
    # cast per file: in early years some columns (e.g., open) are completely empty
    daily = pd.concat([pd.read_parquet(f, columns=DAILY_COLUMNS).astype({c: float for c in NUMERIC_COLUMNS})
                       for f in files], ignore_index=True)

    panel = daily.rename(columns=RENAME)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(["id", "date"]).sort_values(["id", "date"]).reset_index(drop=True)

    members = pd.read_parquet(raw / "membership.parquet")
    panel["in_sp500"] = is_member(panel, members)
    return panel


def check_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Print the data checks of Step 1 and return the coverage by year (aggregates only)."""
    m = panel[panel["in_sp500"]]
    per_day = m.groupby("date").size()
    ohlc = m.dropna(subset=["open", "high", "low", "close"])

    print(f"Rows: {len(panel):,}, stocks: {panel['id'].nunique():,}, trading days: {panel['date'].nunique():,} "
          f"({panel['date'].min().date()} to {panel['date'].max().date()})")
    print(f"Index members per day: min {per_day.min()}, median {per_day.median():.0f}, max {per_day.max()}")
    print(f"Member-day returns beyond +-100%: {(m['ret'].abs() > 1).sum()}")
    print(f"High below max(open, close): {(ohlc['high'] < ohlc[['open', 'close']].max(axis=1)).sum()}, "
          f"low above min(open, close): {(ohlc['low'] > ohlc[['open', 'close']].min(axis=1)).sum()} "
          f"(of {len(ohlc):,} rows)")
    print("Share of missing values on member days:")
    print(m[["ret", "open", "high", "low", "close", "volume"]].isna().mean().round(4).to_string())

    coverage = m.groupby(m["date"].dt.year).agg(
        trading_days=("date", "nunique"),
        member_days=("id", "size"),
        stocks=("id", "nunique"),
        missing_ret=("ret", lambda x: x.isna().mean()),
        missing_high_low=("high", lambda x: x.isna().mean()),
    )
    coverage["members_per_day"] = coverage["member_days"] / coverage["trading_days"]
    return coverage.rename_axis("year")
