"""Direction forecast and honest strategy evaluation endpoints."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import SymbolPath
from app.direction import latest_probability, walk_forward
from app.features import InsufficientDataError
from app.repository import get_price_df
from app.schemas import DirectionEvidence, DirectionOut, StrategyOut
from app.signals import DISCLAIMER
from app.strategy import CostModel, Rule, buy_and_hold, positions_from_scores, simulate

router = APIRouter(prefix="/stocks", tags=["research"])


def _frame(db: Session, symbol: str):
    df = get_price_df(db, symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return df


@router.get("/{symbol}/direction", response_model=DirectionOut)
def direction(symbol: SymbolPath, db: Session = Depends(get_db)):
    """P(up tomorrow) plus the walk-forward evidence on whether it means anything."""
    symbol = symbol.upper()
    df = _frame(db, symbol)
    try:
        ev, _ = walk_forward(df)
        p = latest_probability(df)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    if ev.significant:
        verdict = "The model shows a statistically significant edge out of sample."
    else:
        verdict = (
            "No statistically significant edge over the baseline out of sample. "
            "Treat prob_up as roughly the base rate."
        )
    return DirectionOut(
        symbol=symbol,
        as_of=df["date"].iloc[-1],
        prob_up=round(p, 3),
        lean="UP" if p > 0.5 else "DOWN",
        evidence=DirectionEvidence(
            n=ev.n,
            accuracy=round(ev.accuracy, 4),
            majority_baseline=round(ev.base_rate, 4),
            edge_points=round(ev.edge * 100, 2),
            p_value=round(ev.p_value, 4),
            folds_beating_baseline=ev.folds_beating_base,
            folds=ev.folds,
            significant=ev.significant,
        ),
        verdict=verdict,
        disclaimer=DISCLAIMER,
    )


@router.get("/{symbol}/strategy", response_model=StrategyOut)
def strategy(symbol: SymbolPath, db: Session = Depends(get_db)):
    """Out-of-sample backtest of the fixed rule, net of NEPSE-style costs."""
    symbol = symbol.upper()
    df = _frame(db, symbol)
    rule, costs = Rule(), CostModel()
    try:
        _, p_up = walk_forward(df)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    first = int(np.argmax(p_up.notna().to_numpy()))
    close = df["close"].to_numpy()[first:]
    s = simulate(close, positions_from_scores(p_up.to_numpy()[first:], rule), costs)
    b = buy_and_hold(close, costs)
    return StrategyOut(
        symbol=symbol,
        days=len(close) - 1,
        strategy_net_pct=round(s["net_return"] * 100, 2),
        strategy_gross_pct=round(s["gross_return"] * 100, 2),
        buy_hold_net_pct=round(b["net_return"] * 100, 2),
        strategy_max_drawdown_pct=round(s["max_drawdown"] * 100, 2),
        trades=s["trades"],
        time_in_market_pct=round(s["time_in_market"] * 100, 1),
        beats_buy_hold=bool(s["net_return"] > b["net_return"]),
        rule={"enter": rule.enter, "exit": rule.exit, "min_hold_days": rule.min_hold},
        note="Fixed rule, out-of-sample only, fills at the close, no slippage. "
        "Verify current NEPSE fee rates in app/strategy.py.",
    )
