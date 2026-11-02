"""Read helpers that turn database rows into DataFrames."""
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Price

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def list_symbols(db: Session) -> list[str]:
    return list(db.scalars(select(Price.symbol).distinct().order_by(Price.symbol)))


def get_price_df(db: Session, symbol: str) -> pd.DataFrame:
    """Prices for one symbol, oldest first. Empty frame if none."""
    rows = db.scalars(
        select(Price).where(Price.symbol == symbol).order_by(Price.date)
    ).all()
    return pd.DataFrame(
        [{c: getattr(r, c) for c in COLUMNS} for r in rows], columns=COLUMNS
    )
