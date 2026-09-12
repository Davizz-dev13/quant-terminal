#!/usr/bin/env python3
"""Generador del apartado "Rotacion Semis" (estrategia #4 de David).
Salida: docs/semis/semis.json - vivo (go-live 2026-09-09), walk-forward anual
OOS y backtest 10y de referencia vs buy & hold equiponderado de los 5 semis.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.semis_core import SEMIS, target_weights, backtest, rotation_log

GO_LIVE = "2026-09-09"
WINDOW_YEARS = 10
START_YEAR = 2018

CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
COST = float(CFG.get("backtest", {}).get("transaction_cost", 0.001))
CASH = float(CFG.get("backtest", {}).get("cash_rate_annual", 0.02))


def stats_d(ret: pd.Series) -> dict:
    ret = ret.fillna(0.0)
    if len(ret) < 10:
        return {}
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    ytd = ret[ret.index >= f"{ret.index[-1].year}-01-01"]
    return {
        "sharpe": round(float(ret.mean() / sd * np.sqrt(252)), 2) if sd and sd > 1e-12 else None,
        "max_dd": round(float(dd.min()) * 100, 2),
        "total_return": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "ytd_return": round(((1 + ytd).prod() - 1) * 100, 2) if len(ytd) else 0.0,
    }


def folds_d(ret: pd.Series, start_year: int) -> list[dict]:
    out = []
    for y in range(start_year, ret.index[-1].year + 1):
        seg = ret[(ret.index >= f"{y}-01-01") & (ret.index <= f"{y}-12-31")].fillna(0.0)
        if len(seg) < 60:
            continue
        eq = (1 + seg).cumprod()
        dd = eq / eq.cummax() - 1
        sd = seg.std(ddof=1)
        out.append({"year": y,
                    "sharpe": round(float(seg.mean() / sd * np.sqrt(252)), 2) if sd and sd > 1e-12 else None,
                    "return_pct": round((float(eq.iloc[-1]) - 1) * 100, 2),
                    "max_dd": round(float(dd.min()) * 100, 2)})
    return out


def download(t: str) -> pd.Series:
    df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"sin datos para {t}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df["Close"]


def main():
    errors, closes = [], {}
    for t in SEMIS:
        try:
            closes[t] = download(t)
        except Exception as e:
            errors.append(f"{t}: {e}")
    C = pd.DataFrame(closes).dropna()
    W_full = target_weights(C)
    ret_full = backtest(C, W_full, COST, CASH)
    bh_full = C.pct_change().fillna(0.0).mean(axis=1)

    wstart = C.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
    Cw = C[C.index >= wstart]
    W = W_full[W_full.index >= wstart]
    ret = ret_full[ret_full.index >= wstart]
    bh = bh_full[bh_full.index >= wstart]

    eq = (1 + ret.fillna(0.0)).cumprod()
    bh_eq = (1 + bh.fillna(0.0)).cumprod()

    live_idx = [i for i, d in enumerate(Cw.index) if str(d.date()) > GO_LIVE]
    live = {"start": GO_LIVE, "dates": [], "equity": [], "bh": []}
    if live_idx:
        seg = ret.fillna(0.0).iloc[live_idx]
        live["dates"] = [str(d.date()) for d in Cw.index[live_idx]]
        live["equity"] = [round(float(v), 4) for v in (1 + seg).cumprod()]
        bseg = bh.fillna(0.0).iloc[live_idx]
        live["bh"] = [round(float(v), 4) for v in (1 + bseg).cumprod()]

    cur = W.iloc[-1]
    holdings = [{"asset": t, "weight": round(float(w), 4),
                 "price": round(float(Cw[t].iloc[-1]), 2)}
                for t, w in cur.items() if w > 0]
    invested = round(float(cur.sum()), 4)

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "universe": SEMIS, "go_live": GO_LIVE,
        "rules": {
            "name": "roc126 top2 vol-escalada",
            "selection": "Top-2 por ROC de 126 dias, solo si ROC > 0 (si ninguno, 100% cash)",
            "sizing": "Reparto equitativo entre patas; cada pata escalada por min(1, 0.10/vol20)",
            "execution": "Senal al cierre, posicion desde la barra siguiente",
            "cost": COST, "cash_rate": CASH},
        "dates": [str(d.date()) for d in Cw.index],
        "equity": [round(float(v), 4) for v in eq],
        "bh_equity": [round(float(v), 4) for v in bh_eq],
        "stats": stats_d(ret),
        "bh_stats": stats_d(bh),
        "folds": folds_d(ret, START_YEAR),
        "bh_folds": {f["year"]: f["return_pct"] for f in folds_d(bh, START_YEAR)},
        "live": live,
        "holdings": holdings, "invested": invested,
        "rotation_log": rotation_log(W),
        "errors": errors,
    }
    pos_folds = [f for f in out["folds"] if f["return_pct"] > 0]
    out["pct_folds_positive"] = round(100 * len(pos_folds) / len(out["folds"]), 0) if out["folds"] else None

    p = ROOT / "docs" / "semis" / "semis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print("stats:", out["stats"])
    print("bh:", out["bh_stats"])
    print("holdings:", holdings, "invested:", invested)
    print("folds:", [(f["year"], f["return_pct"]) for f in out["folds"]])
    print("errors:", errors)


if __name__ == "__main__":
    main()
