"""Pydantic request and response models."""
import datetime as dt

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

SYMBOL_PATTERN = r"^[A-Za-z0-9]{1,12}$"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PriceIn(BaseModel):
    date: dt.date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)

    @model_validator(mode="after")
    def check_ohlc(self) -> "PriceIn":
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must lie between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must lie between low and high")
        return self


class PriceBatch(BaseModel):
    prices: list[PriceIn] = Field(min_length=1, max_length=1000)


class PriceBatchResult(BaseModel):
    inserted: int
    skipped: int


class PriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: dt.date
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockSummary(BaseModel):
    symbol: str
    rows: int
    latest_date: dt.date


class PredictionOut(BaseModel):
    symbol: str
    as_of: dt.date
    current_close: float
    predicted_close: float
    change_pct: float
    signal: str
    reasons: list[str]
    rsi: float
    ema_20: float
    disclaimer: str


class ModelMetrics(BaseModel):
    symbol: str
    rows: int
    test_rows: int
    model_rmse: float
    model_mae: float
    naive_rmse: float
    naive_mae: float
    beats_naive: bool


class WatchlistIn(BaseModel):
    symbol: str = Field(pattern=SYMBOL_PATTERN)


class WatchlistOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str

class TrackRecordRow(BaseModel):
    date: dt.date
    actual_close: float
    predicted_close: float
    previous_close: float


class TrackRecordOut(BaseModel):
    symbol: str
    days: int
    model_mae: float
    naive_mae: float
    model_rmse: float
    naive_rmse: float
    model_closer_days: int
    beats_naive: bool
    rows: list[TrackRecordRow]
    note: str


class DirectionEvidence(BaseModel):
    n: int
    accuracy: float
    majority_baseline: float
    edge_points: float
    p_value: float
    folds_beating_baseline: int
    folds: int
    significant: bool


class DirectionOut(BaseModel):
    symbol: str
    as_of: dt.date
    prob_up: float
    lean: str
    evidence: DirectionEvidence
    verdict: str
    disclaimer: str


class StrategyOut(BaseModel):
    symbol: str
    days: int
    strategy_net_pct: float
    strategy_gross_pct: float
    buy_hold_net_pct: float
    strategy_max_drawdown_pct: float
    trades: int
    time_in_market_pct: float
    beats_buy_hold: bool
    rule: dict
    note: str
