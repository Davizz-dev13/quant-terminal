#!/usr/bin/env python3
"""Generador del apartado "Mag7 SMA200 semanal" (estrategia contrarian de David).
Universo: 7 magnificas + GLD + BTC-USD. Salida: docs/mag7/mag7.json
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

from scripts.mag7w_core import weekly_frame, weekly_position, trades_from_position, weekly_backtest

UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "GLD", "BTC-USD"]
GO_LIVE = "2026-09-08"
WINDOW_YEARS = 10  # metricas/graficas solo ultimos 10 años; el historico previo solo calienta la SMA200W

CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
COST = float(CFG.get("backtest", {}).get("transaction_cost", 0.001))
CASH = float(CFG.get("backtest", {}).get("cash_rate_annual", 0.02))


def download(t: str) -> pd.DataFrame:
    df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"sin datos para {t}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def stats_w(ret: pd.Series) -> dict:
    ret = ret.fillna(0.0)
    if len(ret) < 10:
        return {}
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    ytd = ret[ret.index >= f"{ret.index[-1].year}-01-01"]
    return {
        "sharpe": round(float(ret.mean() / sd * np.sqrt(52)), 2) if sd and sd > 1e-12 else None,
        "max_dd": round(float(dd.min()) * 100, 2),
        "total_return": round((float(eq.iloc[-1]) - 1) * 100, 2),
        "ytd_return": round(((1 + ytd).prod() - 1) * 100, 2) if len(ytd) else 0.0,
    }


def folds_w(ret: pd.Series, start_year: int) -> list[dict]:
    out = []
    for y in range(start_year, ret.index[-1].year + 1):
        seg = ret[(ret.index >= f"{y}-01-01") & (ret.index <= f"{y}-12-31")].fillna(0.0)
        if len(seg) < 20:
            continue
        eq = (1 + seg).cumprod()
        dd = eq / eq.cummax() - 1
        sd = seg.std(ddof=1)
        out.append({"year": y,
                    "sharpe": round(float(seg.mean() / sd * np.sqrt(52)), 2) if sd and sd > 1e-12 else None,
                    "return_pct": round((float(eq.iloc[-1]) - 1) * 100, 2),
                    "max_dd": round(float(dd.min()) * 100, 2)})
    return out


def main():
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "universe": UNIVERSE,
           "go_live": GO_LIVE, "rules": {
               "entry": "Cruce a la baja de la SMA200 semanal (cierre semanal)",
               "exit": "RSI(14) semanal > 70 arma la salida; venta al caer >5% desde el maximo posterior",
               "execution": "Senal al cierre de la semana, posicion desde la semana siguiente",
               "cost": COST, "cash_rate": CASH},
           "assets": {}, "portfolio": {}, "live": [], "errors": []}

    port_rets, bh_rets = {}, {}
    for t in UNIVERSE:
        try:
            df = download(t)
            x = weekly_frame(df)
            pos = weekly_position(x)
            ret_full = weekly_backtest(x, pos, COST, CASH)
            bh_full = x["Close"].pct_change().fillna(0.0)
            wstart = x.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
            trades = [tr for tr in trades_from_position(x, pos)
                      if tr["entry_date"] >= str(wstart.date())]
            x = x[x.index >= wstart]
            pos = pos[pos.index >= wstart]
            ret = ret_full[ret_full.index >= wstart]
            bh = bh_full[bh_full.index >= wstart]
            bh_stats = stats_w(bh)
            closed = [tr for tr in trades if not tr["open"]]
            wins = [tr for tr in closed if tr["return_pct"] > 0]
            port_rets[t], bh_rets[t] = ret, bh
            out["assets"][t] = {
                "dates": [str(d.date()) for d in x.index],
                "close": [round(float(v), 2) for v in x["Close"]],
                "sma200": [None if np.isnan(v) else round(float(v), 2) for v in x["SMA200"]],
                "trades": trades,
                "stats": {**stats_w(ret),
                          "n_trades": len(closed),
                          "win_rate": round(100 * len(wins) / len(closed), 0) if closed else None,
                          "avg_trade": round(float(np.mean([tr["return_pct"] for tr in closed])), 2) if closed else None,
                          "exposure": round(float(pos.mean()) * 100, 0),
                          "bh_total_return": bh_stats.get("total_return"),
                          "bh_sharpe": bh_stats.get("sharpe"),
                          "bh_max_dd": bh_stats.get("max_dd")},
            }
            cur = pos.iloc[-1]
            if cur == 1:
                open_tr = trades[-1] if trades and trades[-1]["open"] else None
                out["live"].append({"asset": t, "state": "LONG",
                                    "entry_date": open_tr["entry_date"] if open_tr else None,
                                    "entry_price": open_tr["entry_price"] if open_tr else None,
                                    "current_price": round(float(x["Close"].iloc[-1]), 2),
                                    "unrealized_pct": open_tr["return_pct"] if open_tr else None})
            else:
                out["live"].append({"asset": t, "state": "FLAT", "current_price": round(float(x["Close"].iloc[-1]), 2)})
        except Exception as e:
            out["errors"].append(f"{t}: {e}")

    if port_rets:
        portfolio = pd.DataFrame(port_rets).mean(axis=1).dropna()
        bhp = pd.DataFrame(bh_rets).mean(axis=1).dropna()
        wstart = portfolio.index[-1] - pd.DateOffset(years=WINDOW_YEARS)
        portfolio = portfolio[portfolio.index >= wstart]
        bhp = bhp[bhp.index >= wstart]
        eq = (1 + portfolio.fillna(0.0)).cumprod()
        bh_eq = (1 + bhp.fillna(0.0)).cumprod()
        fl = folds_w(portfolio, portfolio.index[0].year)
        bhfl = {f["year"]: f for f in folds_w(bhp, bhp.index[0].year)}
        for f in fl:
            f["bh_return_pct"] = bhfl.get(f["year"], {}).get("return_pct")
        pos_folds = [f for f in fl if f["return_pct"] > 0]
        out["portfolio"] = {
            "dates": [str(d.date()) for d in portfolio.index],
            "equity": [round(float(v), 4) for v in eq],
            "bh_equity": [round(float(v), 4) for v in bh_eq],
            "stats": stats_w(portfolio),
            "bh_stats": stats_w(bhp),
            "folds": fl,
            "pct_folds_positive": round(100 * len(pos_folds) / len(fl), 0) if fl else None,
        }

    p = ROOT / "docs" / "mag7" / "mag7.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes)")
    for t, a in out["assets"].items():
        print(t, a["stats"])
    print("errors:", out["errors"])


if __name__ == "__main__":
    main()
