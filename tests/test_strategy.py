import numpy as np
import pytest

from app.strategy import CostModel, Rule, buy_and_hold, positions_from_scores, simulate

FREE = CostModel(brokerage=0.0, sebon=0.0, cgt=0.0)


def test_always_in_with_no_costs_matches_price_move():
    close = np.array([100.0, 110.0, 99.0, 120.0])
    out = simulate(close, np.ones(4, dtype=int), FREE)
    assert out["net_return"] == pytest.approx(120 / 100 - 1)
    assert out["trades"] == 1


def test_never_in_market_returns_zero():
    out = simulate(np.array([100.0, 90.0, 80.0]), np.zeros(3, dtype=int))
    assert out["net_return"] == 0 and out["trades"] == 0


def test_costs_and_tax_reduce_return():
    close = np.linspace(100, 150, 30)
    pos = np.ones(30, dtype=int)
    free = simulate(close, pos, FREE)["net_return"]
    paid = simulate(close, pos, CostModel())["net_return"]
    assert paid < free


def test_no_tax_on_losses():
    close = np.array([100.0, 90.0, 80.0])
    taxed = simulate(close, np.ones(3, dtype=int), CostModel(0.0, 0.0, 0.5))
    untaxed = simulate(close, np.ones(3, dtype=int), FREE)
    assert taxed["net_return"] == pytest.approx(untaxed["net_return"])


def test_each_round_trip_counts_one_trade():
    close = np.linspace(100, 110, 8)
    pos = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    assert simulate(close, pos, FREE)["trades"] == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        simulate(np.array([1.0, 2.0]), np.array([1]))


def test_min_hold_blocks_early_exit():
    scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    pos = positions_from_scores(scores, Rule(enter=0.55, exit=0.5, min_hold=3))
    assert list(pos[:4]) == [1, 1, 1, 0]


def test_no_trade_zone():
    pos = positions_from_scores(np.array([0.52, 0.54, 0.50]), Rule(enter=0.55))
    assert pos.sum() == 0


def test_nan_scores_stay_flat():
    assert positions_from_scores(np.array([np.nan, np.nan])).sum() == 0


def test_bad_rule_rejected():
    with pytest.raises(ValueError):
        positions_from_scores(np.array([0.6]), Rule(enter=0.5, exit=0.6))


def test_buy_and_hold_one_trade():
    assert buy_and_hold(np.array([100.0, 101.0, 102.0]))["trades"] == 1
