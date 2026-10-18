# ML Statistical Arbitrage after Trading Costs

Work in progress.

Krauss, Do and Huck (2017) show that machine-learning models trained on past returns earn large daily returns with a long-short strategy on the S&P 500, before transaction costs. This project rebuilds the strategy in simplified form (gradient-boosted trees, plus a simple reversal rule as a benchmark) and asks how much of the return survives stock-specific trading costs estimated from daily high and low prices.

## Data

CRSP daily stock data and point-in-time S&P 500 membership via WRDS, 1990 to 2025. The data cannot be shared.

Wharton Research Data Services (WRDS) was used in preparing this project: "ML Statistical Arbitrage after Trading Costs". This service and the data available thereon constitute valuable intellectual property and trade secrets of WRDS and/or its third-party suppliers.

## Structure

```
config.yaml              parameters
src/data.py              CRSP download and stock-day panel
src/features.py          study periods, lagged-return features, labels
src/models.py            gradient-boosted trees and reversal benchmark
src/backtest.py          long-short portfolio and performance measures
scripts/01_build_data.py
scripts/02_run_baseline.py
tests/                   pytest
docs/documentation.tex   write-up
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/01_build_data.py      # needs a WRDS account (username in WRDS_USERNAME)
python scripts/02_run_baseline.py
pytest
```
