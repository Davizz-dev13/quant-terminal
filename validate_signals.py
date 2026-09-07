from pathlib import Path
import sys
import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data import history
from core.backtest import backtest_strategy, buy_and_hold_backtest, ALL_STRATEGIES

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
tickers = sorted(set(CFG["market"]["universe"] + CFG["market"]["equities"]))

bt_cfg = CFG.get("backtest", {})
cost = float(bt_cfg.get("transaction_cost", 0.001))
cash_rate = float(bt_cfg.get("cash_rate_annual", 0.0))
voltarget_cfg = bt_cfg.get("voltarget", {})
confirmation_days = int(CFG.get("signals", {}).get("sma200", {}).get("confirmation_days", 1))

# Non-investable tickers: kept in the per-asset table and CSV, excluded from
# aggregate means (^VIX is an index, CL=F a futures continuous contract).
EXCLUDE_FROM_AGGREGATES = {"^VIX", "CL=F"}

rows = []
for ticker in tickers:
    try:
        df = history(ticker, "10y", "1d")
        _, bh = buy_and_hold_backtest(df)
        for strategy in ALL_STRATEGIES:
            cd = confirmation_days if strategy == "SMA200" else 1
            _, stats = backtest_strategy(df, strategy, cost=cost, cash_rate=cash_rate,
                                         confirmation_days=cd, voltarget=voltarget_cfg)
            rows.append({
                "Ticker": ticker,
                "Strategy": strategy,
                **stats,
                "BH_CAGR": bh["CAGR"], "BH_Sharpe": bh["Sharpe"], "BH_Vol": bh["Vol"],
                "BH_MaxDD": bh["MaxDD"], "BH_TotalReturn": bh["TotalReturn"],
                "Alpha_CAGR": stats["CAGR"] - bh["CAGR"],
                "Sharpe_Delta": stats["Sharpe"] - bh["Sharpe"],
                "DD_Improvement": stats["MaxDD"] - bh["MaxDD"],
                "Sharpe_Beat_BH": stats["Sharpe"] > bh["Sharpe"],
                "CAGR_Beat_BH": stats["CAGR"] > bh["CAGR"],
                "DD_Better_BH": stats["MaxDD"] > bh["MaxDD"],
            })
    except Exception as e:
        print(f"{ticker}: {e}")

out = pd.DataFrame(rows)
out_path = ROOT / "data" / "validation_results.csv"
out.to_csv(out_path, index=False)

if out.empty:
    print("No validation results generated.")
    raise SystemExit(1)

out["Investable"] = ~out["Ticker"].isin(EXCLUDE_FROM_AGGREGATES)
inv = out[out["Investable"]]

cols = ["CAGR", "Sharpe", "Vol", "MaxDD", "TotalReturn", "Trades", "Exposure", "Alpha_CAGR", "Sharpe_Delta", "DD_Improvement"]
print(out.drop(columns=["Investable"]).to_string(index=False))
print(f"\nAggregate by strategy (mean across investable assets; excluded: {sorted(EXCLUDE_FROM_AGGREGATES)}):")
print(inv.groupby("Strategy")[cols].mean(numeric_only=True).sort_values("Sharpe", ascending=False).to_string())
print("\nConsistency by strategy (investable assets only):")
consistency = inv.groupby("Strategy").agg(
    Assets=("Ticker", "count"),
    Sharpe_Beat_BH=("Sharpe_Beat_BH", "mean"),
    CAGR_Beat_BH=("CAGR_Beat_BH", "mean"),
    DD_Better_BH=("DD_Better_BH", "mean"),
).sort_values("Sharpe_Beat_BH", ascending=False)
print(consistency.to_string())
