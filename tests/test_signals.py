import pytest

from app.signals import generate_signal


def test_buy_when_all_factors_bullish():
    s = generate_signal(current=100, predicted=103, ema_20=95, rsi=25)
    assert s.label == "BUY"
    assert s.score == 3


def test_sell_when_all_factors_bearish():
    s = generate_signal(current=100, predicted=97, ema_20=105, rsi=75)
    assert s.label == "SELL"
    assert s.score == -3


def test_hold_when_signals_mixed():
    s = generate_signal(current=100, predicted=100.2, ema_20=95, rsi=50)
    assert s.label == "HOLD"


def test_reasons_explain_each_factor():
    assert len(generate_signal(100, 103, 95, 25).reasons) == 3


def test_rejects_non_positive_price():
    with pytest.raises(ValueError):
        generate_signal(current=0, predicted=1, ema_20=1, rsi=50)
