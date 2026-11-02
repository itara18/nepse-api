import pytest

from app.predictor import ModelNotFoundError, forecast
from app.training import train_symbol


def test_train_saves_model_and_reports_baseline(price_df, tmp_path):
    metrics = train_symbol(price_df, "NABIL", str(tmp_path))
    assert (tmp_path / "NABIL.joblib").exists()
    assert {"model_rmse", "naive_rmse", "beats_naive"} <= set(metrics)


def test_train_rejects_short_history(price_df, tmp_path):
    with pytest.raises(ValueError):
        train_symbol(price_df.head(40), "NABIL", str(tmp_path))


def test_forecast_after_training(price_df, tmp_path):
    train_symbol(price_df, "NABIL", str(tmp_path))
    result = forecast(price_df, "NABIL", str(tmp_path))
    assert result.predicted_close > 0
    assert result.as_of == price_df["date"].iloc[-1]


def test_forecast_without_model_raises(price_df, tmp_path):
    with pytest.raises(ModelNotFoundError):
        forecast(price_df, "NABIL", str(tmp_path))


def test_forecast_supports_return_target_bundles(price_df, tmp_path):
    import joblib
    from sklearn.dummy import DummyRegressor

    from app.features import FEATURE_COLUMNS, build_training_set

    X, _ = build_training_set(price_df)
    model = DummyRegressor(strategy="constant", constant=1.0).fit(X, [0.0] * len(X))
    joblib.dump(
        {"model": model, "features": FEATURE_COLUMNS, "target": "return_pct", "metrics": {}},
        tmp_path / "NABIL.joblib",
    )
    result = forecast(price_df, "NABIL", str(tmp_path))
    assert result.predicted_close == pytest.approx(result.current_close * 1.01)
