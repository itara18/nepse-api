import logging

import pytest

from app.config import Settings
from app.main import check_settings


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_requests_are_logged(client, caplog):
    with caplog.at_level(logging.INFO, logger="nepse"):
        client.get("/health")
    assert "GET /health -> 200" in caplog.text


def test_production_rejects_default_secret():
    with pytest.raises(RuntimeError):
        check_settings(Settings(environment="production", secret_key="change-me"))


def test_production_rejects_short_secret():
    with pytest.raises(RuntimeError):
        check_settings(Settings(environment="production", secret_key="short"))


def test_production_accepts_long_secret():
    check_settings(Settings(environment="production", secret_key="x" * 32))


def test_development_allows_default_secret():
    check_settings(Settings(environment="development", secret_key="change-me"))


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/docs"