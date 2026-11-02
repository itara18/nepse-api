import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import models, ratelimit
from app.config import Settings, get_settings
from app.database import Base, get_db
from app.main import app


@pytest.fixture(autouse=True)
def _reset_ratelimit():
    ratelimit.reset()


@pytest.fixture()
def price_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 500 + np.cumsum(rng.normal(0, 3, 150))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=150).date,
            "open": close - 1,
            "high": close + 2,
            "low": close - 2,
            "close": close,
            "volume": rng.integers(500, 2000, 150),
        }
    )


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def client(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def seeded(db_session, price_df):
    """150 days of NABIL prices in the test database."""
    for row in price_df.itertuples(index=False):
        db_session.add(
            models.Price(
                symbol="NABIL",
                date=row.date,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            )
        )
    db_session.commit()


@pytest.fixture()
def model_dir(tmp_path):
    """Point the app at an empty temporary model folder."""
    app.dependency_overrides[get_settings] = lambda: Settings(model_dir=str(tmp_path))
    return tmp_path


@pytest.fixture()
def make_headers(client):
    def _make(email="a@b.com", password="password123"):
        client.post("/auth/register", json={"email": email, "password": password})
        resp = client.post("/auth/login", data={"username": email, "password": password})
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _make


@pytest.fixture()
def auth_headers(make_headers):
    return make_headers()
