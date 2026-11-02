"""Rule-based Buy / Sell / Hold signal. Educational use only."""
from dataclasses import dataclass, field

DISCLAIMER = (
    "Educational project. Not financial advice. "
    "Predictions are unreliable; do not trade on them."
)


@dataclass(frozen=True)
class Signal:
    label: str
    score: int
    reasons: list[str] = field(default_factory=list)


def generate_signal(
    current: float, predicted: float, ema_20: float, rsi: float
) -> Signal:
    """Combine three factors into BUY, SELL or HOLD.

    Each factor adds +1 (bullish) or -1 (bearish). Score >= 2 is BUY,
    score <= -2 is SELL, anything else is HOLD.
    """
    if current <= 0:
        raise ValueError("current price must be positive")

    score = 0
    reasons: list[str] = []

    change_pct = (predicted - current) / current * 100
    if change_pct > 1:
        score += 1
        reasons.append(f"Model expects +{change_pct:.1f}% next day")
    elif change_pct < -1:
        score -= 1
        reasons.append(f"Model expects {change_pct:.1f}% next day")
    else:
        reasons.append("Model expects a flat next day")

    if current > ema_20:
        score += 1
        reasons.append("Price above 20-day EMA (uptrend)")
    else:
        score -= 1
        reasons.append("Price below 20-day EMA (downtrend)")

    if rsi < 30:
        score += 1
        reasons.append(f"RSI {rsi:.0f}: oversold")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI {rsi:.0f}: overbought")
    else:
        reasons.append(f"RSI {rsi:.0f}: neutral")

    label = "BUY" if score >= 2 else "SELL" if score <= -2 else "HOLD"
    return Signal(label=label, score=score, reasons=reasons)
