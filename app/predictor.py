"""Load a trained model and forecast the next close."""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.features import InsufficientDataError, latest_features

# Referencing these loads scikit-learn at startup, so the first forecast is fast.
_PRELOADED = (LinearRegression, Pipeline, StandardScaler)


class ModelNotFoundError(Exception):
    """No trained model on disk for this symbol."""


@dataclass(frozen=True)
class Forecast:
    as_of: dt.date
    current_close: float
    predicted_close: float
    rsi: float
    ema_20: float


def load_bundle(symbol: str, model_dir: str) -> dict:
    path = Path(model_dir) / f"{symbol}.joblib"
    if not path.is_file():
        raise ModelNotFoundError(symbol)
    return joblib.load(path)


def forecast(df: pd.DataFrame, symbol: str, model_dir: str) -> Forecast:
    bundle = load_bundle(symbol, model_dir)
    feats = latest_features(df)  # may raise InsufficientDataError
    raw = float(bundle["model"].predict(feats[bundle["features"]])[0])
    row = feats.iloc[0]
    current = float(row["close"])
    # Bundles from the notebook predict next-day return in percent.
    # Bundles from app.training predict the price directly.
    if bundle.get("target") == "return_pct":
        predicted = current * (1 + raw / 100)
    else:
        predicted = raw
    return Forecast(
        as_of=df["date"].iloc[-1],
        current_close=current,
        predicted_close=predicted,
        rsi=float(row["rsi"]),
        ema_20=float(row["ema_20"]),
    )


__all__ = ["Forecast", "InsufficientDataError", "ModelNotFoundError", "forecast"]