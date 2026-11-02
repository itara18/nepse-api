"""Download NEPSE price history from the ShareSansarScraper public API.

Usage: python -m app.fetcher                 (default 5 stocks)
       python -m app.fetcher NABIL NICA      (chosen stocks)

Writes data/<SYMBOL>.csv. Data credit: ShareSansarScraper by OmitNomis (MIT).
"""
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://omitnomis.github.io/ShareSansarScraper/api/history/{symbol}.json"
DEFAULT_SYMBOLS = ["NABIL", "NICA", "GBIME", "SCB", "SHIVM"]
COLUMN_MAP = {
    "d": "Date",
    "o": "Open",
    "h": "High",
    "l": "Low",
    "c": "Close",
    "vol": "Volume",
}

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Download or parsing failed."""


def parse_history(payload: dict) -> pd.DataFrame:
    """Turn the API JSON ({cols, rows}) into a DataFrame."""
    try:
        df = pd.DataFrame(payload["rows"], columns=payload["cols"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FetchError(f"Unexpected response format: {exc}") from exc
    df = df.rename(columns=COLUMN_MAP)
    missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise FetchError(f"Response missing columns: {missing}")
    return df[list(COLUMN_MAP.values())]


def fetch_history(
    symbol: str,
    retries: int = 3,
    timeout: float = 15,
    backoff: float = 1.0,
    getter=requests.get,
) -> pd.DataFrame:
    """Fetch one symbol. Retries on network or format errors."""
    if not re.fullmatch(r"[A-Za-z0-9]{1,12}", symbol):
        raise ValueError(f"Invalid symbol: {symbol!r}")
    url = BASE_URL.format(symbol=symbol.upper())
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = getter(url, timeout=timeout)
            resp.raise_for_status()
            return parse_history(resp.json())
        except (requests.RequestException, ValueError, FetchError) as exc:
            last_error = exc
            logger.warning("%s attempt %d/%d failed: %s", symbol, attempt, retries, exc)
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise FetchError(f"{symbol}: failed after {retries} attempts: {last_error}")


def download_symbol(symbol: str, out_dir: Path | str = "data", **kwargs) -> Path:
    """Fetch one symbol and save data/<SYMBOL>.csv."""
    df = fetch_history(symbol, **kwargs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol.upper()}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=DEFAULT_SYMBOLS)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    failed = 0
    for symbol in args.symbols:
        try:
            path = download_symbol(symbol, args.out)
        except (FetchError, ValueError) as exc:
            print(f"{symbol}: FAILED ({exc})")
            failed += 1
            continue
        print(f"{symbol}: saved {path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
