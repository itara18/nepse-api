"""Replay the forecast model day by day, using only data known at the time.

For each recent day, the model is retrained on older days only and asked to
guess that day's close. The guess can then be compared with the real close.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import InsufficientDataError, build_features

MIN_TRAIN_ROWS = 60
WARMUP_ROWS = 25  # rows lost while indicators warm up


def track_record(df: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """Guess vs real close for the last `days` days.

    Columns: date, actual, predicted, previous (yesterday's close).
    """
    n = len(df)
    first = max(n - days, MIN_TRAIN_ROWS + WARMUP_ROWS)
    if first >= n:
        raise InsufficientDataError("Not enough history to replay forecasts")

    close = df["close"].reset_index(drop=True)
    dates = df["date"].reset_index(drop=True)
    X = build_features(df).reset_index(drop=True).to_numpy()
    ret = ((close.shift(-1) / close - 1) * 100).to_numpy()  # next-day return, %
    usable = ~np.isnan(X).any(axis=1) & ~np.isnan(ret)

    rows = []
    for j in range(first, n):
        # Day j is guessed from day j-1. Train only on rows 0..j-2, because
        # their next-day close is already known at the end of day j-1.
        train = usable.copy()
        train[j - 1 :] = False
        if train.sum() < MIN_TRAIN_ROWS or np.isnan(X[j - 1]).any():
            continue
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X[train], ret[train])
        pred_ret = float(model.predict(X[j - 1 : j])[0])
        rows.append(
            {
                "date": dates[j],
                "actual": float(close[j]),
                "predicted": float(close[j - 1] * (1 + pred_ret / 100)),
                "previous": float(close[j - 1]),
            }
        )
    if not rows:
        raise InsufficientDataError("Not enough history to replay forecasts")
    return pd.DataFrame(rows)