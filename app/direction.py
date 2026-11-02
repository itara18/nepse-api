"""Next-day direction model with honest walk-forward evaluation.

Why direction and not price: next-day price level is dominated by today's
price, so a model that predicts it cannot beat "tomorrow = today". The
useful question is "is P(up tomorrow) better than the base rate", which
we can score against a majority-class baseline.

All features at row t use data up to and including day t.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import InsufficientDataError, compute_rsi

CIRCUIT_PCT = 9.5  # NEPSE daily price band is 10 percent
DIRECTION_FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10",
    "gap", "range_pct", "vol_20", "vol_ratio",
    "dist_ema20", "rsi", "circuit_hit",
]
MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5"]
MIN_ROWS = 120


def direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """NEPSE-aware features. Early rows are NaN."""
    close = df["close"]
    ret1 = close.pct_change() * 100
    ema20 = close.ewm(span=20, adjust=False).mean()
    feats = pd.DataFrame(
        {
            "ret_1": ret1,
            "ret_2": close.pct_change(2) * 100,
            "ret_5": close.pct_change(5) * 100,
            "ret_10": close.pct_change(10) * 100,
            "gap": (df["open"] / close.shift(1) - 1) * 100,
            "range_pct": (df["high"] - df["low"]) / close * 100,
            "vol_20": ret1.rolling(20).std(),
            "vol_ratio": df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan),
            "dist_ema20": (close / ema20 - 1) * 100,
            "rsi": compute_rsi(close) / 100,
            "circuit_hit": (ret1.abs() >= CIRCUIT_PCT).astype(float),
        },
        index=df.index,
    )
    return feats[DIRECTION_FEATURES]


def make_model(C: float = 0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


@dataclass(frozen=True)
class Evidence:
    n: int
    accuracy: float
    base_rate: float  # accuracy of always guessing the training majority class
    edge: float
    p_value: float
    brier: float
    brier_base: float
    folds_beating_base: int
    folds: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.edge > 0


def _z_test_p(acc: float, p0: float, n: int) -> float:
    """Two-sided normal-approximation p-value for acc vs p0."""
    if n == 0 or p0 <= 0 or p0 >= 1:
        return 1.0
    z = (acc - p0) / math.sqrt(p0 * (1 - p0) / n)
    return math.erfc(abs(z) / math.sqrt(2))


def _labelled(df: pd.DataFrame, extra: pd.DataFrame | None = None):
    X = direction_features(df)
    if extra is not None:
        X = pd.concat([X, extra.reindex(X.index)], axis=1)
    nxt = df["close"].shift(-1) / df["close"] - 1
    y = (nxt > 0).astype(float).where(nxt.notna() & (nxt != 0))  # drop flat days
    return X, y


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    first_frac: float = 0.5,
    extra: pd.DataFrame | None = None,
) -> tuple[Evidence, pd.Series]:
    """Expanding-window evaluation. Returns evidence and out-of-sample P(up).

    The returned Series is indexed like `df` and is NaN where no
    out-of-sample forecast exists (the first `first_frac` of the data).
    """
    if len(df) < MIN_ROWS:
        raise InsufficientDataError(f"Need at least {MIN_ROWS} rows, got {len(df)}")
    X, y = _labelled(df, extra)
    ok = X.notna().all(axis=1) & y.notna()
    idx = np.flatnonzero(ok.to_numpy())
    start = int(len(idx) * first_frac)
    edges = np.linspace(start, len(idx), n_folds + 1).astype(int)

    p_up = pd.Series(np.nan, index=df.index)
    hits = n_total = 0
    brier_sum = brier_base_sum = 0.0
    folds_beating = 0
    base_hits = 0
    for k in range(n_folds):
        tr, te = idx[: edges[k]], idx[edges[k] : edges[k + 1]]
        if len(te) == 0 or len(tr) < 30:
            continue
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        actual = y.iloc[te].to_numpy()
        majority = 1.0 if y.iloc[tr].mean() >= 0.5 else 0.0
        fold_hits = int(((proba > 0.5) == (actual == 1)).sum())
        fold_base = int((actual == majority).sum())
        hits += fold_hits
        base_hits += fold_base
        n_total += len(te)
        brier_sum += float(((proba - actual) ** 2).sum())
        brier_base_sum += float(((y.iloc[tr].mean() - actual) ** 2).sum())
        folds_beating += int(fold_hits > fold_base)
        p_up.iloc[te] = proba
    if n_total == 0:
        raise InsufficientDataError("Not enough usable rows for walk-forward")
    acc = hits / n_total
    base = base_hits / n_total
    ev = Evidence(
        n=n_total,
        accuracy=acc,
        base_rate=base,
        edge=acc - base,
        p_value=_z_test_p(acc, base, n_total),
        brier=brier_sum / n_total,
        brier_base=brier_base_sum / n_total,
        folds_beating_base=folds_beating,
        folds=n_folds,
    )
    return ev, p_up


def market_features(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight market proxy built from the available stocks, indexed by date.

    Replace with the real NEPSE index if you have it. Uses same-day data only.
    """
    closes = pd.DataFrame({s: d.set_index("date")["close"] for s, d in frames.items()})
    daily = closes.pct_change().mean(axis=1) * 100
    idx = (1 + daily.fillna(0) / 100).cumprod()
    return pd.DataFrame({"mkt_ret_1": daily, "mkt_ret_5": idx.pct_change(5) * 100})


def latest_probability(df: pd.DataFrame, extra: pd.DataFrame | None = None) -> float:
    """Fit on all labelled history and return P(up) for the next day."""
    X, y = _labelled(df, extra)
    ok = X.notna().all(axis=1) & y.notna()
    if ok.sum() < 60:
        raise InsufficientDataError("Not enough history")
    model = make_model().fit(X[ok], y[ok])
    last = X.iloc[[-1]]
    if last.isna().any().any():
        raise InsufficientDataError("Not enough history to compute features")
    return float(model.predict_proba(last)[0, 1])
