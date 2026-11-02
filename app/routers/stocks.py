import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.backtest import track_record
from app.config import Settings, get_settings
from app.database import get_db
from app.deps import SymbolPath, get_current_user
from app.features import InsufficientDataError
from app.models import Price
from app.predictor import ModelNotFoundError, forecast, load_bundle
from app.repository import get_price_df
from app.schemas import (
    ModelMetrics,
    PredictionOut,
    PriceBatch,
    PriceBatchResult,
    PriceOut,
    StockSummary,
    TrackRecordOut,
    TrackRecordRow,
)
from app.signals import DISCLAIMER, generate_signal

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("", response_model=list[StockSummary])
def list_stocks(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Price.symbol, func.count(Price.id), func.max(Price.date))
        .group_by(Price.symbol)
        .order_by(Price.symbol)
    ).all()
    return [
        StockSummary(symbol=s, rows=n, latest_date=d) for s, n, d in rows
    ]


@router.get("/{symbol}/history", response_model=list[PriceOut])
def history(
    symbol: SymbolPath,
    limit: int = Query(60, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    symbol = symbol.upper()
    rows = db.scalars(
        select(Price)
        .where(Price.symbol == symbol)
        .order_by(Price.date.desc())
        .limit(limit)
    ).all()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    return list(reversed(rows))


@router.post(
    "/{symbol}/prices",
    response_model=PriceBatchResult,
    status_code=201,
    dependencies=[Depends(get_current_user)],
)
def add_prices(
    symbol: SymbolPath, batch: PriceBatch, db: Session = Depends(get_db)
):
    symbol = symbol.upper()
    seen = set(db.scalars(select(Price.date).where(Price.symbol == symbol)))
    new_rows = []
    for item in batch.prices:
        if item.date in seen:
            continue
        seen.add(item.date)
        new_rows.append(Price(symbol=symbol, **item.model_dump()))
    db.add_all(new_rows)
    db.commit()
    return PriceBatchResult(
        inserted=len(new_rows), skipped=len(batch.prices) - len(new_rows)
    )


@router.get("/{symbol}/predict", response_model=PredictionOut)
def predict(
    symbol: SymbolPath,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    symbol = symbol.upper()
    df = get_price_df(db, symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    try:
        f = forecast(df, symbol, settings.model_dir)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=503, detail=f"No trained model for {symbol}"
        ) from None
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    sig = generate_signal(f.current_close, f.predicted_close, f.ema_20, f.rsi)
    return PredictionOut(
        symbol=symbol,
        as_of=f.as_of,
        current_close=round(f.current_close, 2),
        predicted_close=round(f.predicted_close, 2),
        change_pct=round((f.predicted_close - f.current_close) / f.current_close * 100, 2),
        signal=sig.label,
        reasons=sig.reasons,
        rsi=round(f.rsi, 1),
        ema_20=round(f.ema_20, 2),
        disclaimer=DISCLAIMER,
    )


@router.get("/{symbol}/metrics", response_model=ModelMetrics)
def metrics(symbol: SymbolPath, settings: Settings = Depends(get_settings)):
    symbol = symbol.upper()
    try:
        return load_bundle(symbol, settings.model_dir)["metrics"]
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"No trained model for {symbol}"
        ) from None


@router.get("/{symbol}/track-record", response_model=TrackRecordOut)
def get_track_record(
    symbol: SymbolPath,
    days: int = Query(60, ge=10, le=250),
    db: Session = Depends(get_db),
):
    """Real closes next to what the model would have guessed the day before."""
    symbol = symbol.upper()
    df = get_price_df(db, symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    try:
        rec = track_record(df, days)
    except InsufficientDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    model_diff = rec["actual"] - rec["predicted"]
    naive_diff = rec["actual"] - rec["previous"]
    model_rmse = float(np.sqrt(np.mean(model_diff**2)))
    naive_rmse = float(np.sqrt(np.mean(naive_diff**2)))
    return TrackRecordOut(
        symbol=symbol,
        days=len(rec),
        model_mae=round(float(model_diff.abs().mean()), 2),
        naive_mae=round(float(naive_diff.abs().mean()), 2),
        model_rmse=round(model_rmse, 2),
        naive_rmse=round(naive_rmse, 2),
        model_closer_days=int((model_diff.abs() < naive_diff.abs()).sum()),
        beats_naive=bool(model_rmse < naive_rmse),
        rows=[
            TrackRecordRow(
                date=r.date,
                actual_close=round(r.actual, 2),
                predicted_close=round(r.predicted, 2),
                previous_close=round(r.previous, 2),
            )
            for r in rec.itertuples(index=False)
        ],
        note="Replayed day by day with a linear model retrained each day on "
        "past data only. Not the saved model.",
    )