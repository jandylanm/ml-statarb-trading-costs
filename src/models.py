"""Trading signals: gradient-boosted trees and a simple reversal benchmark."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def gbt_probability(train: pd.DataFrame, trade: pd.DataFrame, features: list[str],
                    params: dict, seed: int) -> np.ndarray:
    """Fit gradient-boosted trees on the training set and return P(y = 1) for the trading set."""
    model = LGBMClassifier(**params, random_state=seed, verbose=-1)
    model.fit(train[features].to_numpy(), train["y"].astype(int))
    return model.predict_proba(trade[features].to_numpy())[:, 1]


def reversal_score(trade: pd.DataFrame) -> np.ndarray:
    """Benchmark without machine learning: buy the day's losers, sell the day's winners.

    The score is minus the return of day t (feature R1), so the stocks that fell
    most on day t get the highest score.
    """
    return -trade["R1"].to_numpy()
