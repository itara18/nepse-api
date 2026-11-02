"""Load OHLCV CSV files into the database.

Each file is named <SYMBOL>.csv with columns:
Date, Open, High, Low, Close, Volume (case-insensitive).

Usage: python -m app.loader data/
"""
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import Price

REQUIRED = ["date", "open", "high", "low", "close", "volume"]


def read_price_csv(path: Path) -> pd.DataFrame:
    """Read and clean one OHLCV CSV. Returns rows sorted by date."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ""), errors="coerce"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=REQUIRED).drop_duplicates("date")
    return df.sort_values("date").reset_index(drop=True)


def load_csv(db: Session, symbol: str, path: Path) -> int:
    """Insert new rows only. Returns number inserted."""
    df = read_price_csv(path)
    existing = set(db.scalars(select(Price.date).where(Price.symbol == symbol)))
    new_rows = [
        Price(
            symbol=symbol,
            date=row.date,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        )
        for row in df.itertuples(index=False)
        if row.date not in existing
    ]
    db.add_all(new_rows)
    db.commit()
    return len(new_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    init_db()
    with SessionLocal() as db:
        for path in sorted(args.folder.glob("*.csv")):
            symbol = path.stem.upper()
            try:
                count = load_csv(db, symbol, path)
            except ValueError as exc:
                print(exc)
                continue
            print(f"{symbol}: {count} rows inserted")


if __name__ == "__main__":
    main()
