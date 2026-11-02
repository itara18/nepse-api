# Results (auto-generated)

Walk-forward, 5 expanding folds, out-of-sample only. Flat days are dropped.

## Direction model vs majority-class baseline

| Symbol | Features | n | Accuracy | Baseline | Edge | p-value | Folds won |
|---|---|---|---|---|---|---|---|
| GBIME | stock only | 284 | 52.5% | 52.8% | -0.4 pts | 0.91 | 3/5 |
| GBIME | stock + market | 284 | 53.9% | 52.8% | +1.1 pts | 0.72 | 3/5 |
| NABIL | stock only | 281 | 48.4% | 51.2% | -2.8 pts | 0.34 | 1/5 |
| NABIL | stock + market | 281 | 53.0% | 51.2% | +1.8 pts | 0.55 | 2/5 |
| NICA | stock only | 289 | 55.7% | 57.8% | -2.1 pts | 0.47 | 1/5 |
| NICA | stock + market | 289 | 56.7% | 57.8% | -1.0 pts | 0.72 | 1/5 |
| SCB | stock only | 285 | 51.6% | 49.5% | +2.1 pts | 0.48 | 2/5 |
| SCB | stock + market | 285 | 52.3% | 49.5% | +2.8 pts | 0.34 | 3/5 |
| SHIVM | stock only | 287 | 51.9% | 51.9% | +0.0 pts | 1.00 | 2/5 |
| SHIVM | stock + market | 287 | 53.0% | 51.9% | +1.0 pts | 0.72 | 3/5 |

Significant positive edge (p<0.05): **0 of 10** rows. With this many tests, one or two at p<0.05 is expected by chance alone.

## Strategy vs buy-and-hold (net of costs)

Rule fixed in advance: enter when P(up) > 0.55, exit at <= 0.5, hold at least 5 days. Costs per side 0.415%, capital gains tax 7.5% on profit.

| Symbol | Strategy net | Strategy gross | Buy & hold net | Strategy max DD | Trades | In market |
|---|---|---|---|---|---|---|
| GBIME | -19.2% | -7.6% | 7.2% | -24.1% | 15 | 13.4% |
| NABIL | -9.0% | 4.1% | 9.9% | -17.2% | 15 | 12.2% |
| NICA | -9.0% | -2.1% | -7.6% | -15.5% | 8 | 5.0% |
| SCB | -14.3% | 13.2% | 7.8% | -19.3% | 32 | 30.5% |
| SHIVM | -6.3% | 0.4% | 34.1% | -10.4% | 8 | 6.2% |

Strategy beat buy-and-hold after costs on **0 of 5** stocks.

Limits: no slippage, no liquidity limits, trades fill at the close, small sample, one market regime. Verify current NEPSE fee rates. Not financial advice.
