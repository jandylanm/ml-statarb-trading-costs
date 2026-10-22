"""Step 3: net returns after trading costs.

Estimates stock-level bid-ask spreads with Corwin-Schultz, computes the daily
turnover of each long-short portfolio and subtracts the costs in three scenarios:
flat costs as in the paper (5 bps per half-turn), half the estimated spread for
every unit traded (spread floored at one tick, main scenario), and the same with
the spread floored at zero only (robustness). Uses the predictions of
scripts/02_run_baseline.py.

Usage:
    python scripts/03_run_costs.py           # spread estimates are cached
    python scripts/03_run_costs.py --force   # recompute the spread estimates
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import long_short_returns, performance, portfolio_weights
from src.config import load_config
from src.costs import breakeven_spread, rolling_spread, spread_costs, turnover

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

cfg = load_config()
out_dir = cfg["paths"]["results"] / "costs"
out_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Spread estimates (stock level, cached in data/processed)
# ----------------------------------------------------------------------------
panel = pd.read_parquet(cfg["paths"]["processed"] / "panel.parquet")
spread_path = cfg["paths"]["processed"] / "spreads.parquet"
spreads = pd.read_parquet(spread_path) if spread_path.exists() and not args.force else None
if spreads is None or "unfloored" not in spreads.columns:  # no cache, --force, or a cache from an older version
    spreads = rolling_spread(panel, cfg["costs"]["spread_window"], cfg["costs"]["spread_min_obs"],
                             cfg["costs"]["tick_sizes"])
    spreads.to_parquet(spread_path)

# aggregate checks on index members only
spreads["spread_zero"] = spreads["unfloored"].clip(lower=0)  # robustness: floored at zero instead of one tick
members = spreads.merge(panel.loc[panel["in_sp500"], ["id", "date"]], on=["id", "date"])
by_month = members.groupby(members["date"].dt.to_period("M"))["spread"].median()
by_month.to_csv(out_dir / "median_spread_by_month.csv", float_format="%.6f")
print("Spread estimates of index members (in percent):")
print(f"  median {members['spread'].median() * 100:.3f}, mean {members['spread'].mean() * 100:.3f}, "
      f"10% quantile {members['spread'].quantile(0.1) * 100:.3f}, 90% quantile {members['spread'].quantile(0.9) * 100:.3f}")
print(f"  share of member days without estimate: {members['spread'].isna().mean():.4f}")
for month in ["2008-10", "2020-03"]:  # crisis months: spreads should be visibly higher
    if pd.Period(month) in by_month.index:
        print(f"  median in {month}: {by_month[pd.Period(month)] * 100:.3f}")

# by year: rolling estimate (tick floor and zero floor), how often the tick floor binds,
# share of zeros and the raw two-day estimates before flooring
members["tick_floor"] = members["spread"] > members["spread_zero"]
spread_by_year = members.groupby(members["date"].dt.year).agg(
    median_spread=("spread", "median"),
    median_spread_zero=("spread_zero", "median"),
    share_tick_floor=("tick_floor", "mean"),
    share_zero=("spread", lambda x: (x == 0).mean()),
    mean_raw=("raw", "mean"),
    median_raw=("raw", "median"),
    share_raw_negative=("raw", lambda x: (x < 0).mean()),
)
spread_by_year[["median_spread", "median_spread_zero", "mean_raw", "median_raw"]] *= 100
spread_by_year.to_csv(out_dir / "spread_by_year.csv", float_format="%.4f")
print("\nSpread estimates by year (spreads in percent):")
print(spread_by_year.round(3).to_string())

# ----------------------------------------------------------------------------
# Gross and net returns
# ----------------------------------------------------------------------------
pred = pd.concat([pd.read_parquet(p) for p in sorted((cfg["paths"]["processed"] / "predictions" / "baseline").glob("*.parquet"))],
                 ignore_index=True)
spread_matrix = spreads.pivot(index="date", columns="id", values="spread")
spread_matrix_zero = spreads.pivot(index="date", columns="id", values="spread_zero")
flat = cfg["costs"]["flat_cost_bps"] / 10_000

daily, stats, breakeven_by_year = {}, {}, {}
for signal in ["gbt", "reversal"]:
    for k in cfg["backtest"]["k_values"]:
        weights = portfolio_weights(pred, signal, k)
        to = turnover(weights)
        for lag in cfg["backtest"]["lags"]:
            name = f"{signal}_k{k}_lag{lag}"
            gross = long_short_returns(pred, signal, k, lag)
            # trades are executed lag days after the signal, at that day's spread
            exec_spreads = spread_matrix.shift(-lag).reindex(index=weights.index, columns=weights.columns)
            exec_spreads_zero = spread_matrix_zero.shift(-lag).reindex(index=weights.index, columns=weights.columns)
            cost_cs = spread_costs(weights, exec_spreads)
            cost_cs_zero = spread_costs(weights, exec_spreads_zero)
            daily[f"{name}_gross"] = gross
            daily[f"{name}_flat"] = gross - flat * to # as in the original paper
            daily[f"{name}_cs"] = gross - cost_cs # Corwin-Schultz costs, spread floored at one tick
            daily[f"{name}_cs_zero"] = gross - cost_cs_zero # robustness: spread floored at zero
            stats[name] = {
                "turnover": to.mean(),
                "cost_flat_bps": flat * to.mean() * 10_000,
                "cost_cs_bps": cost_cs.mean() * 10_000,
                "cost_cs_zero_bps": cost_cs_zero.mean() * 10_000,
                "spread_paid_bps": (2 * cost_cs.sum() / to.sum()) * 10_000,  # average spread per unit traded
                "breakeven_spread_bps": breakeven_spread(gross, to) * 10_000,
            }
            year = gross.index.year
            breakeven_by_year[name] = pd.Series(
                {y: breakeven_spread(gross[year == y], to[year == y]) * 10_000 for y in sorted(set(year))})
            if name == "gbt_k10_lag0":  # spread paid by the main strategy, by year
                paid_by_year = 2 * cost_cs.groupby(year).sum() / to.groupby(year).sum() * 10_000

daily = pd.DataFrame(daily)
daily.to_csv(out_dir / "daily_returns.csv", float_format="%.6f")
days = cfg["backtest"]["days_per_year"]
summary = performance(daily, days)
summary.to_csv(out_dir / "summary.csv", float_format="%.6f")
stats = pd.DataFrame(stats).T
stats.to_csv(out_dir / "turnover_and_costs.csv", float_format="%.4f")
by_year_returns = daily.groupby(daily.index.year).mean()
by_year_returns.to_csv(out_dir / "by_year.csv", float_format="%.6f")

print("\nTurnover and costs per day:")
print(stats.round(2).to_string())
print("\nMean daily return (%) and Sharpe ratio: gross, flat costs, Corwin-Schultz costs (tick floor, zero floor)")
table = pd.DataFrame(index=stats.index)
scenarios = ["gross", "flat", "cs", "cs_zero"]
for scen in scenarios:
    table[f"mean_{scen}"] = [summary.loc[f"{n}_{scen}", "mean_daily"] * 100 for n in stats.index]
for scen in scenarios:
    table[f"sharpe_{scen}"] = [summary.loc[f"{n}_{scen}", "sharpe"] for n in stats.index]
table.to_csv(out_dir / "net_summary.csv", float_format="%.4f")
print(table.round(3).to_string())
print("\nMean daily return by year (%) and average spread paid (bps), gbt k=10 lag 0:")
year_table = by_year_returns[["gbt_k10_lag0_gross", "gbt_k10_lag0_flat", "gbt_k10_lag0_cs"]] * 100
year_table["spread_paid_bps"] = paid_by_year
year_table["median_member_spread_bps"] = spread_by_year["median_spread"] * 100
print(year_table.round(3).to_string())

breakeven_by_year = pd.DataFrame(breakeven_by_year)
breakeven_by_year.to_csv(out_dir / "breakeven_spread_by_year.csv", float_format="%.2f")
print("\nBreak-even spread by year (bps): spread at which costs eat the whole gross return")
print(breakeven_by_year[["gbt_k10_lag0", "gbt_k10_lag1", "gbt_k50_lag0", "reversal_k10_lag0"]].round(1).to_string())

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for scen, label in [("gross", "Before costs"), ("flat", "Flat costs (5 bps per half-turn)"),
                    ("cs", "Estimated spreads (Corwin-Schultz)")]:
    (1 + daily[f"gbt_k10_lag0_{scen}"].fillna(0)).cumprod().plot(ax=ax, label=label)
ax.set_yscale("log")
ax.set_ylabel("Value of 1 invested (log scale)")
ax.set_title("Gradient-boosted trees, k = 10")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "cumulative_returns_costs.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 4))
breakeven_by_year["gbt_k10_lag0"].plot(ax=ax, marker="o", label="Break-even spread, trading at the signal close")
breakeven_by_year["gbt_k10_lag1"].plot(ax=ax, marker="o", label="Break-even spread, trading one day later")
ax.axhline(2 * cfg["costs"]["flat_cost_bps"], color="gray", linestyle="--", label="Flat costs of the paper")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Spread (bps)")
ax.set_title("Gradient-boosted trees, k = 10: spread at which costs eat the gross return")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "breakeven_spread.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 3.5))
(by_month * 100).plot(ax=ax)
ax.set_ylabel("Median spread estimate (%)")
ax.set_title("Corwin-Schultz spread estimates of S&P 500 members")
fig.tight_layout()
fig.savefig(out_dir / "median_spread.png", dpi=150)
