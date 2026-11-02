import pandas as pd
import pytest

from app.features import (
    FEATURE_COLUMNS,
    InsufficientDataError,
    build_features,
    build_training_set,
    compute_rsi,
    latest_features,
)


def test_rsi_is_100_when_price_only_rises():
    close = pd.Series(range(1, 60), dtype=float)
    assert compute_rsi(close).iloc[-1] == 100


def test_rsi_stays_between_0_and_100(price_df):
    rsi = compute_rsi(price_df["close"]).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_build_features_has_expected_columns(price_df):
    assert list(build_features(price_df).columns) == FEATURE_COLUMNS


def test_build_features_rejects_missing_columns(price_df):
    with pytest.raises(ValueError, match="Missing columns"):
        build_features(price_df.drop(columns=["volume"]))


def test_target_is_next_day_close(price_df):
    X, y = build_training_set(price_df)
    expected = price_df["close"].shift(-1).loc[y.index]
    assert (y == expected).all()
    assert X.index.max() < price_df.index.max()  # last day has no target


def test_changing_last_day_does_not_alter_earlier_features(price_df):
    changed = price_df.copy()
    changed.loc[changed.index[-1], "close"] *= 2
    pd.testing.assert_frame_equal(
        build_features(price_df).iloc[:-1], build_features(changed).iloc[:-1]
    )


def test_latest_features_returns_one_row(price_df):
    assert len(latest_features(price_df)) == 1


def test_latest_features_needs_enough_history(price_df):
    with pytest.raises(InsufficientDataError):
        latest_features(price_df.head(5))


def test_latest_features_rejects_empty_frame(price_df):
    with pytest.raises(InsufficientDataError):
        latest_features(price_df.head(0))
