"""Step 2: baseline before trading costs.

For each study period, fit gradient-boosted trees on the training days and
predict the trading days. The reversal benchmark needs no fitting. Then form
the long-short portfolios and evaluate them.

Usage:
    python scripts/02_run_baseline.py           # predictions are cached per period
    python scripts/02_run_baseline.py --noise   # leakage test: features replaced by noise
    python scripts/02_run_baseline.py --force   # recompute the predictions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import hit_rate, long_short_returns, performance
from src.config import load_config
from src.features import period_dataset, study_periods, to_wide
from src.models import gbt_probability, reversal_score

parser = argparse.ArgumentParser()
parser.add_argument("--noise", action="store_true")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

cfg = load_config()
horizons = cfg["features"]["horizons"]
features = [f"R{m}" for m in horizons]
name = "noise" if args.noise else "baseline"
pred_dir = cfg["paths"]["processed"] / "predictions" / name
out_dir = cfg["paths"]["results"] / name
pred_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Predictions per study period
# ----------------------------------------------------------------------------
panel = pd.read_parquet(cfg["paths"]["processed"] / "panel.parquet")
ret_wide, member_wide = to_wide(panel)
dates = ret_wide.index
periods = study_periods(len(dates), cfg["study_periods"]["train_days"], cfg["study_periods"]["trade_days"])
print(f"{len(periods)} study periods, first trading day {dates[periods[0]['train_end']].date()}")

preds = []
for p in periods:
    path = pred_dir / f"period_{p['index']:02d}.parquet"
    if path.exists() and not args.force:
        preds.append(pd.read_parquet(path))
        continue

    train, trade = period_dataset(ret_wide, member_wide, p, horizons)
    if args.noise:  # same labels and timing, but the features carry no information
        rng = np.random.default_rng(cfg["seed"] + p["index"])
        train[features] = rng.standard_normal((len(train), len(features)))
        trade[features] = rng.standard_normal((len(trade), len(features)))

    trade["gbt"] = gbt_probability(train, trade, features, cfg["model"], cfg["seed"])
    trade["reversal"] = reversal_score(trade)
    result = trade[["date", "id", "y", "ret_t1", "ret_t2", "gbt", "reversal"]]
    result.to_parquet(path)
    preds.append(result)
    print(f"Period {p['index']:2d}: trading {dates[p['train_end']].date()} to {dates[p['end'] - 1].date()}, "
          f"{len(train):,} training rows, share of y = 1: {train['y'].mean():.3f}")

pred = pd.concat(preds, ignore_index=True)

# ----------------------------------------------------------------------------
# Portfolio returns and evaluation
# ----------------------------------------------------------------------------
returns = pd.DataFrame({
    f"{signal}_k{k}_lag{lag}": long_short_returns(pred, signal, k, lag)
    for signal in ["gbt", "reversal"]
    for k in cfg["backtest"]["k_values"]
    for lag in cfg["backtest"]["lags"]
})
returns.to_csv(out_dir / "daily_returns.csv", float_format="%.6f")

days = cfg["backtest"]["days_per_year"]
start, end = cfg["backtest"]["paper_period"]
summary = performance(returns, days)
summary_paper = performance(returns.loc[start:end], days)
summary.to_csv(out_dir / "summary.csv", float_format="%.6f")
summary_paper.to_csv(out_dir / "summary_paper_period.csv", float_format="%.6f")

labeled = pred[pred["y"].notna()]
accuracy = {"gbt overall": ((labeled["gbt"] > 0.5) == (labeled["y"] == 1)).mean()}
for k in cfg["backtest"]["k_values"]:
    accuracy[f"gbt hit rate k={k}"] = hit_rate(pred, "gbt", k)
    accuracy[f"reversal hit rate k={k}"] = hit_rate(pred, "reversal", k)

by_year = returns.groupby(returns.index.year).mean()
by_year.to_csv(out_dir / "by_year.csv", float_format="%.6f")

print("\nPerformance before costs, full sample:")
print(summary.round(4).to_string())
print(f"\nPerformance before costs, paper period {start} to {end}:")
print(summary_paper[["mean_daily", "sharpe"]].round(4).to_string())
print("\nAccuracy:")
for key, value in accuracy.items():
    print(f"  {key:<24} {value:.4f}")
print("\nMean daily return by year (lag 0):")
print(by_year.filter(like="lag0").round(4).to_string())

# ----------------------------------------------------------------------------
# Figure: cumulative returns, k = 10, lag 0
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for signal, label in [("gbt", "Gradient-boosted trees"), ("reversal", "Simple reversal")]:
    (1 + returns[f"{signal}_k10_lag0"].fillna(0)).cumprod().plot(ax=ax, label=label)
ax.set_yscale("log")
ax.set_ylabel("Value of 1 invested (log scale)")
ax.set_title("Long-short portfolio, k = 10, before trading costs")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "cumulative_returns.png", dpi=150)
