#!/usr/bin/env python3
"""Generador del dashboard Mag7: SMA200 y SMA50 (cruces) sobre las 7 magnificas.
Misma filosofia que el panel principal: backtest etiquetado, walk-forward OOS
por pliegues anuales y paper trading en vivo desde el go-live de la variante.
Salida: docs/mag7/mag7.json
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
from core.backtest import backtest_strategy

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]
STRATEGIES = ["SMA200", "SMA50"]
GO_LIVE = "2026-09-08"  # arranque de la variante Mag7
START_YEAR = 2018

CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
COST = float(CFG.get("backtest", {}).get("transaction_cost", 0.001))
CASH = float(CFG.get("backtest", {}).get("cash_rate_annual", 0.02))


def stats(ret: pd.Series) -> dict:
    ret = ret.fillna(0.0)
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    ytd = ret[ret.index >= f"{ret.index[-1].year}-01-01"]
    return {
        "sharpe": float(ret.mean() / sd * np.sqrt(252)) if sd and sd > 1e-12 else None,
        "max_dd": round(float(dd.min()) * 100, 2),
        "total_return": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "ytd_return": round(((1 + ytd).prod() - 1) * 100, 2) if len(ytd) else 0.0,
    }


def fold_stats(ret: pd.Series) -> dict | None:
    ret = ret.fillna(0.0)
    if len(ret) < 20:
        return None
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    return {
        "sharpe": round(float(ret.mean() / sd * np.sqrt(252)), 2) if sd and sd > 1e-12 else None,
        "return_pct": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "max_dd": round(float(dd.min()) * 100, 2),
    }


def main():
    rets, bh_rets, positions, closes = {}, {}, {}, {}
    errors = []
    for t in UNIVERSE:
        try:
            df = history(t, "10y", "1d", refresh=True)
            closes[t] = df["Close"]
            bh_rets[t] = df["Close"].pct_change().fillna(0.0)
            for s in STRATEGIES:
                r, st = backtest_strategy(df, s, cost=COST, cash_rate=CASH)
                rets[(t, s)] = r
                from core.signals import strategy_position
                positions[(t, s)] = strategy_position(df, s)
        except Exception as e:
            errors.append(f"{t}: {e}")

    idx = closes[UNIVERSE[0]].index
    dates = [str(d.date()) for d in idx]
    year = idx[-1].year

    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "universe": UNIVERSE,
           "go_live": GO_LIVE, "ytd_start": f"{year}-01-01", "dates": dates,
           "strategies": {}, "per_asset": {}, "walkforward": {}, "signals": [],
           "live": {"start": GO_LIVE, "dates": [], "strategies": {}, "bh": []},
           "errors": errors}

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

        # Walk-forward OOS anual sobre la cartera equal-weight
        folds, fold_sharpes, parts = [], [], []
        for y in range(START_YEAR, year + 1):
            test = portfolio[(portfolio.index >= f"{y}-01-01") & (portfolio.index <= f"{y}-12-31")]
            fst = fold_stats(test)
            if fst is None:
                continue
            folds.append({"year": y, **fst})
            if fst["sharpe"] is not None:
                fold_sharpes.append(fst["sharpe"])
            parts.append(test)
        oos = pd.concat(parts) if parts else pd.Series(dtype=float)
        oos_eq = (1 + oos.fillna(0.0)).cumprod()
        oos_dd = oos_eq / oos_eq.cummax() - 1
        sd = oos.std(ddof=1)
        out["walkforward"][s] = {
            "folds": folds,
            "agg": {
                "n_folds": len(folds),
                "oos_total_return": round((float(oos_eq.iloc[-1]) - 1) * 100, 2) if len(oos_eq) else None,
                "oos_sharpe": round(float(oos.mean() / sd * np.sqrt(252)), 2) if sd and sd > 1e-12 else None,
                "oos_max_dd": round(float(oos_dd.min()) * 100, 2) if len(oos_dd) else None,
                "median_fold_sharpe": round(float(np.median(fold_sharpes)), 2) if fold_sharpes else None,
                "pct_folds_positive": round(100 * np.mean([f["return_pct"] > 0 for f in folds]), 0) if folds else None,
            },
        }

    # Paper trading en vivo desde el go-live de la variante
    live_idx = [i for i, d in enumerate(dates) if d > GO_LIVE]
    if live_idx:
        out["live"]["dates"] = [dates[i] for i in live_idx]
        for s in STRATEGIES:
            portfolio = pd.DataFrame({t: rets[(t, s)] for t in UNIVERSE if (t, s) in rets}).mean(axis=1)
            eq = (1 + portfolio.iloc[live_idx].fillna(0.0)).cumprod()
            out["live"]["strategies"][s] = [round(float(v), 4) for v in eq]
        bh = pd.DataFrame(bh_rets).mean(axis=1).iloc[live_idx]
        out["live"]["bh"] = [round(float(v), 4) for v in (1 + bh.fillna(0.0)).cumprod()]

    # Ultimas senales (cambios de regimen de los ultimos 30 cambios)
    for t in UNIVERSE:
        for s in STRATEGIES:
            if (t, s) not in positions:
                continue
            pos = positions[(t, s)]
            chg = pos.diff().fillna(0) != 0
            for d in pos.index[chg][-30:]:
                i = pos.index.get_loc(d)
                out["signals"].append({
                    "date": str(d.date()), "asset": t, "strategy": s,
                    "side": "LONG" if int(pos.iloc[i]) else "EXIT",
                    "price": round(float(closes[t].iloc[i]), 2),
                })
    out["signals"].sort(key=lambda x: x["date"], reverse=True)
    out["signals"] = out["signals"][:40]

    p = ROOT / "docs" / "mag7" / "mag7.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes), errors: {errors}")
    for s in STRATEGIES:
        print(s, out["walkforward"][s]["agg"])


if __name__ == "__main__":
    main()
