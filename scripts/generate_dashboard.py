#!/usr/bin/env python3
"""Regenerate docs/data.json for the Quant Terminal dashboard.
Runs nightly via GitHub Actions. Uses the project's own signal engine and
backtest conventions (transaction cost + cash rate from config/settings.yaml).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data import history
from core.signals import strategy_position
from core.backtest import backtest_strategy, buy_and_hold_backtest
from scripts.david_sma200w import david_weekly_daily_returns

UNIVERSE = ["SPY", "QQQ", "GLD", "GC=F", "CL=F", "TLT", "AMD", "TSM", "ASML", "AVGO", "BTC-USD"]
# SMA200W+RSI = la estrategia de David (semanal): cruce a la baja de la SMA200W,
# salida RSI(14)W armada + caida >5%. El resto son diarias del motor.
STRATEGIES = ["SMA200W+RSI", "ROC60", "Donchian20"]
DAVID_STRATEGY = "SMA200W+RSI"

CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
COST = float(CFG.get("backtest", {}).get("transaction_cost", 0.001))
CASH = float(CFG.get("backtest", {}).get("cash_rate_annual", 0.02))
MAG7W_CFG = CFG.get("mag7w", {})
RSI_DEFAULT = float(MAG7W_CFG.get("rsi_level", 70))
RSI_BY_ASSET = {k: float(v) for k, v in MAG7W_CFG.get("rsi_level_by_asset", {}).items()}


def stats(ret: pd.Series) -> dict:
    ret = ret.fillna(0.0)
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    sharpe = float(ret.mean() / sd * np.sqrt(252)) if sd and sd > 1e-12 else None
    ytd = ret[ret.index >= f"{ret.index[-1].year}-01-01"]
    return {
        "sharpe": sharpe,
        "max_dd": round(float(dd.min()) * 100, 2),
        "total_return": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "ytd_return": round(((1 + ytd).prod() - 1) * 100, 2) if len(ytd) else 0.0,
    }


def main():
    data, rets, bh_rets, closes = {}, {}, {}, {}
    errors = []
    for t in UNIVERSE:
        try:
            df = history(t, "10y", "1d", refresh=True)
            closes[t] = df["Close"]
            bh_rets[t] = df["Close"].pct_change().fillna(0.0)
            for s in STRATEGIES:
                if s == DAVID_STRATEGY:
                    # historico completo para calentar la SMA200W; metricas en la ventana 10y
                    df_full = history(t, "max", "1d", refresh=True)
                    r_full, _ = david_weekly_daily_returns(
                        df_full, COST, CASH, RSI_BY_ASSET.get(t, RSI_DEFAULT))
                    rets[(t, s)] = r_full.reindex(df.index).fillna(0.0)
                else:
                    r, st = backtest_strategy(df, s, cost=COST, cash_rate=CASH)
                    rets[(t, s)] = r
        except Exception as e:
            errors.append(f"{t}: {e}")

    idx = closes[UNIVERSE[0]].index
    dates = [str(d.date()) for d in idx]
    year = idx[-1].year

    GO_LIVE = "2026-09-07"  # inicio del paper trading en vivo (arranque del servicio)
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "universe": UNIVERSE,
           "go_live": GO_LIVE,
           "strategies": {}, "per_asset": {}, "errors": errors}

    for s in STRATEGIES:
        portfolio = pd.DataFrame({t: rets[(t, s)] for t in UNIVERSE if (t, s) in rets}).mean(axis=1)
        eq = (1 + portfolio).cumprod()
        bh = pd.DataFrame(bh_rets).mean(axis=1)
        bh_eq = (1 + bh).cumprod()
        out["strategies"][s] = {
            "equity": [round(float(v), 4) for v in eq],
            "bh_equity": [round(float(v), 4) for v in bh_eq],
            **stats(portfolio),
        }
        for t in UNIVERSE:
            if (t, s) not in rets:
                continue
            e = (1 + rets[(t, s)]).cumprod()
            b = (1 + bh_rets[t]).cumprod()
            out["per_asset"][f"{t}|{s}"] = {
                "equity": [round(float(v), 4) for v in e],
                "bh_equity": [round(float(v), 4) for v in b],
                **stats(rets[(t, s)]),
            }

    # Paper trading en vivo: curvas desde el go-live, normalizadas a 1.
    live = {"start": GO_LIVE, "dates": [], "strategies": {}, "bh": []}
    live_idx = [i for i, d in enumerate(dates) if d > GO_LIVE]
    if live_idx:
        live["dates"] = [dates[i] for i in live_idx]
        for s in STRATEGIES:
            portfolio = pd.DataFrame({t: rets[(t, s)] for t in UNIVERSE if (t, s) in rets}).mean(axis=1)
            seg = portfolio.iloc[live_idx]
            eq = (1 + seg.fillna(0.0)).cumprod()
            live["strategies"][s] = [round(float(v), 4) for v in eq]
        bh = pd.DataFrame(bh_rets).mean(axis=1).iloc[live_idx]
        bh_eq = (1 + bh.fillna(0.0)).cumprod()
        live["bh"] = [round(float(v), 4) for v in bh_eq]
    out["live"] = live

    bh_all = pd.DataFrame(bh_rets).mean(axis=1)
    bh_ytd = bh_all[bh_all.index >= f"{year}-01-01"]
    out["bh_ytd"] = round(((1 + bh_ytd.fillna(0.0)).prod() - 1) * 100, 2) if len(bh_ytd) else 0.0
    out["dates"] = dates
    out["ytd_start"] = f"{year}-01-01"

    p = ROOT / "docs" / "data.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes), errors: {errors}")


if __name__ == "__main__":
    main()
