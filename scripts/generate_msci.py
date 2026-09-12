#!/usr/bin/env python3
"""Generador del apartado "MomVol MSCI World" (pata mundo del terminal).
Salida: docs/msci/msci.json - vivo (go-live 2026-09-10), walk-forward anual
OOS y backtest de referencia vs buy & hold de URTH (proxy MSCI World, 2012+).
Regla MomVol (canonica del harness de investigacion): largo si ROC126 > 0,
peso min(1, 0.10/vol20), senal al cierre ejecutable desde la barra siguiente,
coste 0.1% por cambio, cash remunerado segun config.
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

TICKER = "URTH"
NAME = "MomVol MSCI World"
ROC_DAYS = 126
VOL_DAYS = 20
TARGET_VOL = 0.10
GO_LIVE = "2026-09-10"
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
    yrs = (ret.index[-1] - ret.index[0]).days / 365.25
    ytd = ret[ret.index >= f"{ret.index[-1].year}-01-01"]
    return {
        "sharpe": round(float(ret.mean() / sd * np.sqrt(252)), 2) if sd and sd > 1e-12 else None,
        "max_dd": round(float(dd.min()) * 100, 2),
        "total_return": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "cagr": round(((float(eq.iloc[-1])) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else None,
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


def momvol_weights(close: pd.Series) -> pd.Series:
    """Peso objetivo diario: min(1, 0.10/vol20) si ROC126 > 0; si no, 0."""
    roc = close.pct_change(ROC_DAYS)
    vol = close.pct_change().rolling(VOL_DAYS).std(ddof=1) * np.sqrt(252)
    w = (0.10 / vol).clip(upper=1.0).fillna(0.0)
    return w.where(roc > 0, 0.0).fillna(0.0)


def backtest(close: pd.Series, w: pd.Series, cost: float, cash_rate: float) -> pd.Series:
    held = w.shift(1).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    cash_daily = cash_rate / 252
    return asset_ret * held + cash_daily * (1 - held.clip(upper=1.0)) - turnover * cost


def main():
    errors = []
    try:
        close = download(TICKER)
    except Exception as e:
        print(json.dumps({"errors": [str(e)]}))
        raise
    w_full = momvol_weights(close)
    ret_full = backtest(close, w_full, COST, CASH)
    bh_full = close.pct_change().fillna(0.0)

    wstart = close.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
    c = close[close.index >= wstart]
    ret = ret_full[ret_full.index >= wstart]
    bh = bh_full[bh_full.index >= wstart]
    w = w_full[w_full.index >= wstart]

    eq = (1 + ret.fillna(0.0)).cumprod()
    bh_eq = (1 + bh.fillna(0.0)).cumprod()

    live_idx = [i for i, d in enumerate(c.index) if str(d.date()) > GO_LIVE]
    live = {"start": GO_LIVE, "dates": [], "equity": [], "bh": []}
    if live_idx:
        seg = ret.fillna(0.0).iloc[live_idx]
        live["dates"] = [str(d.date()) for d in c.index[live_idx]]
        live["equity"] = [round(float(v), 4) for v in (1 + seg).cumprod()]
        bseg = bh.fillna(0.0).iloc[live_idx]
        live["bh"] = [round(float(v), 4) for v in (1 + bseg).cumprod()]

    out = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "ticker": TICKER, "go_live": GO_LIVE,
        "rules": {
            "name": "MomVol",
            "selection": f"Largo si ROC de {ROC_DAYS} dias > 0; si no, cash",
            "sizing": f"Peso min(1, {TARGET_VOL}/vol20)",
            "execution": "Senal al cierre, posicion desde la barra siguiente",
            "cost": COST, "cash_rate": CASH,
            "note": "Proxy URTH (iShares MSCI World), datos desde 2012 - ventana mas corta que el resto del terminal"},
        "dates": [str(d.date()) for d in c.index],
        "equity": [round(float(v), 4) for v in eq],
        "bh_equity": [round(float(v), 4) for v in bh_eq],
        "stats": stats_d(ret),
        "bh_stats": stats_d(bh),
        "folds": folds_d(ret, START_YEAR),
        "bh_folds": {f["year"]: f["return_pct"] for f in folds_d(bh, START_YEAR)},
        "live": live,
        "position": {"long": bool(w_full.iloc[-1] > 0), "weight": round(float(w_full.iloc[-1]), 4),
                     "price": round(float(close.iloc[-1]), 2)},
        "errors": errors,
    }
    pos_folds = [f for f in out["folds"] if f["return_pct"] > 0]
    out["pct_folds_positive"] = round(100 * len(pos_folds) / len(out["folds"]), 0) if out["folds"] else None

    p = ROOT / "docs" / "msci" / "msci.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    print("stats:", out["stats"])
    print("bh:", out["bh_stats"])
    print("position:", out["position"])
    print("folds:", [(f["year"], f["return_pct"]) for f in out["folds"]])
    print("pct_pos:", out["pct_folds_positive"])


if __name__ == "__main__":
    main()
