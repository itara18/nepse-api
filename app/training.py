"""Train one model per symbol and compare against a naive baseline.

Usage: python -m app.training
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.features import FEATURE_COLUMNS, build_training_set
from app.repository import get_price_df, list_symbols

MIN_TRAIN_ROWS = 60
TRAIN_RATIO = 0.8


def _rmse(actual, predicted) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(diff**2)))


def _mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def train_symbol(df: pd.DataFrame, symbol: str, model_dir: str) -> dict:
    """Fit, evaluate on a time-ordered holdout, save, return metrics.

    Baseline is "tomorrow's close equals today's close". A model that
    cannot beat it has no real predictive value.
    """
    X, y = build_training_set(df)
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(f"Need at least {MIN_TRAIN_ROWS} usable rows, got {len(X)}")

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    naive = X_test["close"]

    metrics = {
        "symbol": symbol,
        "rows": len(X),
        "test_rows": len(X_test),
        "model_rmse": round(_rmse(y_test, predicted), 4),
        "model_mae": round(_mae(y_test, predicted), 4),
        "naive_rmse": round(_rmse(y_test, naive), 4),
        "naive_mae": round(_mae(y_test, naive), 4),
    }
    metrics["beats_naive"] = metrics["model_rmse"] < metrics["naive_rmse"]

    model.fit(X, y)  # refit on everything for deployment
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "metrics": metrics},
        out_dir / f"{symbol}.joblib",
    )
    return metrics


def main() -> None:
    init_db()
    settings = get_settings()
    with SessionLocal() as db:
        symbols = list_symbols(db)
        if not symbols:
            print("No price data. Run: python -m app.loader data/")
            return
        for symbol in symbols:
            try:
                m = train_symbol(get_price_df(db, symbol), symbol, settings.model_dir)
            except ValueError as exc:
                print(f"{symbol}: skipped ({exc})")
                continue
            verdict = "beats" if m["beats_naive"] else "does NOT beat"
            print(
                f"{symbol}: RMSE {m['model_rmse']} vs naive {m['naive_rmse']} "
                f"-> model {verdict} baseline"
            )


if __name__ == "__main__":
    main()
