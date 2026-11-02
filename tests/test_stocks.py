from app.training import train_symbol

NEW_PRICE = {
    "date": "2026-01-05",
    "open": 100,
    "high": 105,
    "low": 99,
    "close": 102,
    "volume": 1000,
}


def test_list_stocks_empty(client):
    resp = client.get("/stocks")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_stocks_shows_summary(client, seeded):
    data = client.get("/stocks").json()
    assert data[0]["symbol"] == "NABIL"
    assert data[0]["rows"] == 150


def test_history_respects_limit_and_order(client, seeded):
    body = client.get("/stocks/NABIL/history?limit=5").json()
    dates = [row["date"] for row in body]
    assert len(body) == 5
    assert dates == sorted(dates)


def test_history_is_case_insensitive(client, seeded):
    assert client.get("/stocks/nabil/history").status_code == 200


def test_history_unknown_symbol_404(client):
    assert client.get("/stocks/NOPE/history").status_code == 404


def test_history_rejects_malformed_symbol(client):
    assert client.get("/stocks/NA..BIL/history").status_code == 422


def test_history_rejects_bad_limit(client, seeded):
    assert client.get("/stocks/NABIL/history?limit=0").status_code == 422


def test_add_prices_requires_auth(client):
    resp = client.post("/stocks/NICA/prices", json={"prices": [NEW_PRICE]})
    assert resp.status_code == 401


def test_add_prices_skips_duplicates(client, auth_headers):
    body = {"prices": [NEW_PRICE]}
    first = client.post("/stocks/NICA/prices", json=body, headers=auth_headers)
    second = client.post("/stocks/NICA/prices", json=body, headers=auth_headers)
    assert first.json() == {"inserted": 1, "skipped": 0}
    assert second.json() == {"inserted": 0, "skipped": 1}


def test_add_prices_rejects_high_below_low(client, auth_headers):
    bad = {**NEW_PRICE, "high": 90}
    resp = client.post("/stocks/NICA/prices", json={"prices": [bad]}, headers=auth_headers)
    assert resp.status_code == 422


def test_predict_unknown_symbol_404(client, model_dir):
    assert client.get("/stocks/NOPE/predict").status_code == 404


def test_predict_without_model_503(client, seeded, model_dir):
    assert client.get("/stocks/NABIL/predict").status_code == 503


def test_predict_returns_signal(client, seeded, price_df, model_dir):
    train_symbol(price_df, "NABIL", str(model_dir))
    resp = client.get("/stocks/NABIL/predict")
    body = resp.json()
    assert resp.status_code == 200
    assert body["signal"] in {"BUY", "SELL", "HOLD"}
    assert body["predicted_close"] > 0
    assert "Not financial advice" in body["disclaimer"]


def test_metrics_404_before_training(client, model_dir):
    assert client.get("/stocks/NABIL/metrics").status_code == 404


def test_metrics_after_training(client, price_df, model_dir):
    train_symbol(price_df, "NABIL", str(model_dir))
    body = client.get("/stocks/NABIL/metrics").json()
    assert body["symbol"] == "NABIL"
    assert isinstance(body["beats_naive"], bool)
