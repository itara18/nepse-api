"""Cost-aware long-only backtester.

Position rule: a position decided at the close of day t earns the move from
the close of day t to the close of day t+1. Every entry and exit pays
brokerage plus the SEBON fee, and profitable trades pay capital gains tax.

Rates change and depend on broker and trade size. Check current NEPSE rates
and edit `CostModel` before trusting any net number.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    brokerage: float = 0.004  # per side, small-trade slab
    sebon: float = 0.00015  # per side
    cgt: float = 0.075  # on profit, holdings under 365 days

    @property
    def per_side(self) -> float:
        return self.brokerage + self.sebon


@dataclass(frozen=True)
class Rule:
    enter: float = 0.55  # enter when P(up) is above this
    exit: float = 0.50  # leave when P(up) falls to this or below
    min_hold: int = 5  # days to stay in once entered


def positions_from_scores(scores: np.ndarray, rule: Rule = Rule()) -> np.ndarray:
    """0/1 position per day with a no-trade zone and a minimum holding time."""
    if rule.exit > rule.enter:
        raise ValueError("exit must not be above enter")
    pos = np.zeros(len(scores), dtype=int)
    held = 0
    for t, s in enumerate(scores):
        if np.isnan(s):
            held = 0
            continue
        if held > 0:
            held += 1
            if held > rule.min_hold and s <= rule.exit:
                held = 0
        elif s > rule.enter:
            held = 1
        pos[t] = 1 if held > 0 else 0
    return pos


def simulate(close: np.ndarray, position: np.ndarray, costs: CostModel = CostModel()) -> dict:
    """Run the positions over `close`. position[t] earns close[t+1]/close[t]-1.

    The last position value has no next-day price and is ignored.
    """
    close = np.asarray(close, dtype=float)
    position = np.asarray(position, dtype=int)
    if len(close) != len(position):
        raise ValueError("close and position must have the same length")
    n = len(close) - 1
    if n < 1:
        raise ValueError("Need at least two prices")
    ret = close[1:] / close[:-1] - 1
    pos = position[:n]

    equity_net = 1.0
    equity_gross = 1.0
    curve = [1.0]
    trades = 0
    trade_start_equity = None
    for t in range(n):
        entering = pos[t] == 1 and (t == 0 or pos[t - 1] == 0)
        leaving = pos[t] == 0 and t > 0 and pos[t - 1] == 1
        if leaving:
            equity_net = _close_trade(equity_net, trade_start_equity, costs)
            trade_start_equity = None
        if entering:
            equity_net *= 1 - costs.per_side
            trade_start_equity = equity_net
            trades += 1
        if pos[t] == 1:
            equity_net *= 1 + ret[t]
            equity_gross *= 1 + ret[t]
        curve.append(equity_net)
    if pos[-1] == 1:  # still holding at the end: liquidate at the last close
        equity_net = _close_trade(equity_net, trade_start_equity, costs)
        curve[-1] = equity_net

    curve_arr = np.asarray(curve)
    peak = np.maximum.accumulate(curve_arr)
    return {
        "net_return": equity_net - 1,
        "gross_return": equity_gross - 1,
        "max_drawdown": float(np.min(curve_arr / peak - 1)),
        "trades": trades,
        "time_in_market": float(pos.mean()),
        "equity_curve": curve_arr,
    }


def _close_trade(equity: float, start_equity: float | None, costs: CostModel) -> float:
    equity *= 1 - costs.per_side
    if start_equity is not None and equity > start_equity:
        equity -= (equity - start_equity) * costs.cgt
    return equity


def buy_and_hold(close: np.ndarray, costs: CostModel = CostModel()) -> dict:
    """Buy at the first close, sell at the last, same costs."""
    pos = np.ones(len(close), dtype=int)
    out = simulate(close, pos, costs)
    out["trades"] = 1
    return out
