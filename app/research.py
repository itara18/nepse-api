"""Run the full honest evaluation over data/*.csv and write results/.

Usage: python -m app.research [data_dir] [out_dir]

Writes direction_walk_forward.csv, strategy_backtest.csv and REPORT.md.
The strategy rule is fixed in advance (see strategy.Rule); it is not tuned
on the test period.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.direction import market_features, walk_forward
from app.strategy import CostModel, Rule, buy_and_hold, positions_from_scores, simulate

COLS = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def load_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path).rename(columns=COLS)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        frames[path.stem.upper()] = df.sort_values("date").reset_index(drop=True)
    return frames


def run(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    frames = load_frames(data_dir)
    if not frames:
        raise SystemExit(f"No CSV files in {data_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mkt = market_features(frames)
    costs, rule = CostModel(), Rule()

    dir_rows, strat_rows = [], []
    for sym, df in frames.items():
        mkt_aligned = mkt.reindex(df["date"]).set_axis(df.index)
        for label, extra in (("stock only", None), ("stock + market", mkt_aligned)):
            ev, p_up = walk_forward(df, extra=extra)
            dir_rows.append({
                "symbol": sym, "features": label, "n": ev.n,
                "accuracy": ev.accuracy, "majority_baseline": ev.base_rate,
                "edge": ev.edge, "p_value": ev.p_value,
                "brier": ev.brier, "brier_baseline": ev.brier_base,
                "folds_beating_baseline": ev.folds_beating_base, "folds": ev.folds,
            })
            if extra is None:
                oos = p_up.notna().to_numpy()
                first = int(np.argmax(oos))
                close = df["close"].to_numpy()[first:]
                pos = positions_from_scores(p_up.to_numpy()[first:], rule)
                s = simulate(close, pos, costs)
                b = buy_and_hold(close, costs)
                strat_rows.append({
                    "symbol": sym, "days": len(close) - 1,
                    "strategy_net": s["net_return"], "strategy_gross": s["gross_return"],
                    "buy_hold_net": b["net_return"],
                    "strategy_max_dd": s["max_drawdown"], "buy_hold_max_dd": b["max_drawdown"],
                    "trades": s["trades"], "time_in_market": s["time_in_market"],
                })
    dir_df, strat_df = pd.DataFrame(dir_rows), pd.DataFrame(strat_rows)
    dir_df.to_csv(out_dir / "direction_walk_forward.csv", index=False)
    strat_df.to_csv(out_dir / "strategy_backtest.csv", index=False)
    (out_dir / "REPORT.md").write_text(report(dir_df, strat_df, rule, costs), encoding="utf-8")
    return dir_df


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def report(dir_df: pd.DataFrame, strat_df: pd.DataFrame, rule: Rule, costs: CostModel) -> str:
    lines = ["# Results (auto-generated)", "",
             "Walk-forward, 5 expanding folds, out-of-sample only. Flat days are dropped.", "",
             "## Direction model vs majority-class baseline", "",
             "| Symbol | Features | n | Accuracy | Baseline | Edge | p-value | Folds won |",
             "|---|---|---|---|---|---|---|---|"]
    for r in dir_df.itertuples():
        lines.append(f"| {r.symbol} | {r.features} | {r.n} | {_pct(r.accuracy)} | "
                     f"{_pct(r.majority_baseline)} | {r.edge * 100:+.1f} pts | {r.p_value:.2f} | "
                     f"{r.folds_beating_baseline}/{r.folds} |")
    sig = int(((dir_df.p_value < 0.05) & (dir_df.edge > 0)).sum())
    lines += ["", f"Significant positive edge (p<0.05): **{sig} of {len(dir_df)}** rows. "
              "With this many tests, one or two at p<0.05 is expected by chance alone.", "",
              "## Strategy vs buy-and-hold (net of costs)", "",
              f"Rule fixed in advance: enter when P(up) > {rule.enter}, exit at <= {rule.exit}, "
              f"hold at least {rule.min_hold} days. Costs per side {costs.per_side * 100:.3f}%, "
              f"capital gains tax {costs.cgt * 100:.1f}% on profit.", "",
              "| Symbol | Strategy net | Strategy gross | Buy & hold net | Strategy max DD | Trades | In market |",
              "|---|---|---|---|---|---|---|"]
    for r in strat_df.itertuples():
        lines.append(f"| {r.symbol} | {_pct(r.strategy_net)} | {_pct(r.strategy_gross)} | "
                     f"{_pct(r.buy_hold_net)} | {_pct(r.strategy_max_dd)} | {r.trades} | "
                     f"{_pct(r.time_in_market)} |")
    wins = int((strat_df.strategy_net > strat_df.buy_hold_net).sum())
    lines += ["", f"Strategy beat buy-and-hold after costs on **{wins} of {len(strat_df)}** stocks.", "",
              "Limits: no slippage, no liquidity limits, trades fill at the close, small sample, "
              "one market regime. Verify current NEPSE fee rates. Not financial advice.", ""]
    return "\n".join(lines)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    run(data_dir, out_dir)
    print((out_dir / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
