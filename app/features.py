"""Technical-indicator features.

Row t uses data up to and including day t. The target is the close of day
t+1, so no future information leaks into the features.
"""
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "open_close",
    "high_low",
    "pct_change",
    "ma_5",
    "ma_10",
    "ema_10",
    "ema_20",
    "volatility",
    "rsi",
    "close_lag1",
    "close_lag2",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


class InsufficientDataError(ValueError):
    """Not enough history to compute features."""


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, range 0 to 100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators for every row. Early rows contain NaN."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    close = df["close"]
    feats = pd.DataFrame(
        {
            "open_close": df["open"] - close,
            "high_low": df["high"] - df["low"],
            "pct_change": close.pct_change() * 100,
            "ma_5": close.rolling(5).mean(),
            "ma_10": close.rolling(10).mean(),
            "ema_10": close.ewm(span=10, adjust=False).mean(),
            "ema_20": close.ewm(span=20, adjust=False).mean(),
            "volatility": df["high"].rolling(10).std(),
            "rsi": compute_rsi(close),
            "close_lag1": close.shift(1),
            "close_lag2": close.shift(2),
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": close,
            "volume": df["volume"],
        },
        index=df.index,
    )
    return feats[FEATURE_COLUMNS]


def build_training_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features at day t, target is close at day t+1."""
    feats = build_features(df)
    target = df["close"].shift(-1).rename("target")
    data = pd.concat([feats, target], axis=1).dropna()
    return data[FEATURE_COLUMNS], data["target"]


def latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single-row frame of features for the most recent day."""
    if df.empty:
        raise InsufficientDataError("No price data")
    last = build_features(df).iloc[[-1]]
    if last.isna().any().any():
        raise InsufficientDataError("Not enough history to compute indicators")
    return last
