import pytest
import requests

from app.fetcher import FetchError, download_symbol, fetch_history, parse_history
from app.loader import read_price_csv

PAYLOAD = {
    "cols": ["d", "o", "h", "l", "c", "vol"],
    "rows": [
        ["2025-01-01", 100, 105, 99, 102, 1000],
        ["2025-01-02", 102, 106, 101, 104, 1200],
    ],
}


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")

    def json(self):
        return self.payload


def test_parse_history_renames_columns():
    df = parse_history(PAYLOAD)
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2


def test_parse_history_rejects_bad_payload():
    with pytest.raises(FetchError):
        parse_history({})


def test_fetch_retries_then_succeeds():
    calls = []

    def getter(url, timeout):
        calls.append(url)
        return FakeResponse(status=500) if len(calls) < 3 else FakeResponse(PAYLOAD)

    df = fetch_history("NABIL", backoff=0, getter=getter)
    assert len(calls) == 3
    assert len(df) == 2


def test_fetch_gives_up_after_retries():
    calls = []

    def getter(url, timeout):
        calls.append(url)
        return FakeResponse(status=503)

    with pytest.raises(FetchError, match="failed after 3 attempts"):
        fetch_history("NABIL", backoff=0, getter=getter)
    assert len(calls) == 3


def test_fetch_rejects_bad_symbol():
    with pytest.raises(ValueError):
        fetch_history("../etc", getter=lambda *a, **k: FakeResponse(PAYLOAD))


def test_download_writes_csv_the_loader_can_read(tmp_path):
    path = download_symbol(
        "nabil", tmp_path, getter=lambda url, timeout: FakeResponse(PAYLOAD)
    )
    assert path.name == "NABIL.csv"
    assert len(read_price_csv(path)) == 2
