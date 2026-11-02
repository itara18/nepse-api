import pandas as pd
import pytest

from app.backtest import track_record
from app.features import InsufficientDataError


def test_track_record_returns_requested_days(price_df):
    rec = track_record(price_df, days=30)
    assert len(rec) == 30
    assert list(rec.columns) == ["date", "actual", "predicted", "previous"]


def test_actual_and_previous_match_the_prices(price_df):
    rec = track_record(price_df, days=20)
    assert rec["actual"].iloc[-1] == pytest.approx(price_df["close"].iloc[-1])
    assert rec["previous"].iloc[-1] == pytest.approx(price_df["close"].iloc[-2])


def test_guesses_never_use_the_day_being_guessed(price_df):
    changed = price_df.copy()
    changed.loc[changed.index[-1], "close"] *= 2
    original = track_record(price_df, days=30)
    modified = track_record(changed, days=30)
    pd.testing.assert_series_equal(original["predicted"], modified["predicted"])


def test_too_little_history_raises(price_df):
    with pytest.raises(InsufficientDataError):
        track_record(price_df.head(40), days=30)


def test_endpoint_returns_rows_and_summary(client, seeded):
    resp = client.get("/stocks/NABIL/track-record?days=30")
    body = resp.json()
    assert resp.status_code == 200
    assert body["days"] == 30
    assert len(body["rows"]) == 30
    assert isinstance(body["beats_naive"], bool)
    assert 0 <= body["model_closer_days"] <= 30


def test_endpoint_unknown_symbol_404(client):
    assert client.get("/stocks/NOPE/track-record").status_code == 404


def test_endpoint_rejects_bad_days(client, seeded):
    assert client.get("/stocks/NABIL/track-record?days=5").status_code == 422