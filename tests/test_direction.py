import numpy as np
import pandas as pd
import pytest

from app.direction import (
    DIRECTION_FEATURES,
    direction_features,
    latest_probability,
    market_features,
    walk_forward,
)
from app.features import InsufficientDataError


@pytest.fixture()
def long_df(price_df):
    rng = np.random.default_rng(7)
    n = 400
    close = 500 + np.cumsum(rng.normal(0, 3, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2023-01-02", periods=n).date,
            "open": close - 1, "high": close + 2, "low": close - 2,
            "close": close, "volume": rng.integers(500, 2000, n),
        }
    )


def test_features_have_expected_columns(long_df):
    assert list(direction_features(long_df).columns) == DIRECTION_FEATURES


def test_features_never_use_the_future(long_df):
    changed = long_df.copy()
    changed.loc[changed.index[-1], "close"] *= 3
    a = direction_features(long_df).iloc[:-1]
    b = direction_features(changed).iloc[:-1]
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_is_out_of_sample_only(long_df):
    ev, p_up = walk_forward(long_df)
    assert p_up.iloc[: len(long_df) // 2].isna().all()
    assert p_up.dropna().between(0, 1).all()
    assert ev.n > 0 and 0 <= ev.accuracy <= 1 and 0 <= ev.p_value <= 1


def test_walk_forward_forecasts_ignore_later_data(long_df):
    changed = long_df.copy()
    changed.loc[changed.index[-1], "close"] *= 2
    _, p1 = walk_forward(long_df, n_folds=5)
    _, p2 = walk_forward(changed, n_folds=5)
    # the final fold's training set is the same either way up to its first row
    first_fold = p1.dropna().index[0]
    assert p1.loc[first_fold] == pytest.approx(p2.loc[first_fold])


def test_random_walk_shows_no_significant_edge(long_df):
    ev, _ = walk_forward(long_df)
    assert not ev.significant


def test_too_little_data_raises(price_df):
    with pytest.raises(InsufficientDataError):
        walk_forward(price_df.head(50))


def test_latest_probability_in_range(long_df):
    assert 0 <= latest_probability(long_df) <= 1


def test_market_features_shape(long_df):
    m = market_features({"A": long_df, "B": long_df})
    assert list(m.columns) == ["mkt_ret_1", "mkt_ret_5"]


def test_endpoints(client, db_session, long_df):
    from app import models

    for r in long_df.itertuples(index=False):
        db_session.add(models.Price(symbol="TEST", date=r.date, open=float(r.open),
                                    high=float(r.high), low=float(r.low),
                                    close=float(r.close), volume=int(r.volume)))
    db_session.commit()
    d = client.get("/stocks/TEST/direction")
    assert d.status_code == 200
    body = d.json()
    assert 0 <= body["prob_up"] <= 1 and "verdict" in body
    s = client.get("/stocks/TEST/strategy")
    assert s.status_code == 200 and isinstance(s.json()["beats_buy_hold"], bool)
    assert client.get("/stocks/NOPE/direction").status_code == 404
    assert client.get("/stocks/NOPE/strategy").status_code == 404
