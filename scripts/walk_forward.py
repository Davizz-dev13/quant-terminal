#!/usr/bin/env python3
"""Validacion walk-forward de las estrategias del Quant Terminal.

Las estrategias son reglas sin parametros ajustables, asi que la validacion
honesta es out-of-sample por ventanas sucesivas: para cada ano de prueba se
evalua la estrategia solo con datos disponibles hasta ese momento (la regla no
cambia) y se miden Sharpe, rentabilidad y drawdown del periodo de prueba. Se
reportan los pliegues anuales y el agregado OOS, que es lo mas parecido a
"haberlo operado de verdad" sin mirar el futuro.

Salida: docs/walkforward.json
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

UNIVERSE = ["SPY", "QQQ", "GLD", "GC=F", "CL=F", "TLT", "AMD", "TSM", "ASML", "AVGO", "BTC-USD"]
STRATEGIES = ["EMA20>EMA50", "EMA10>EMA30", "EMA50", "Donchian20", "SMA200", "ROC60", "MeanReversionZ"]

CFG = yaml.safe_load(open(ROOT / "config/settings.yaml", encoding="utf8"))
COST = float(CFG.get("backtest", {}).get("transaction_cost", 0.001))
CASH = float(CFG.get("backtest", {}).get("cash_rate_annual", 0.02))
START_YEAR = 2018  # primer ano de prueba; lo anterior es "entrenamiento"


def fold_stats(ret: pd.Series) -> dict:
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
    rets = {}
    errors = []
    for t in UNIVERSE:
        try:
            df = history(t, "10y", "1d")  # cache del generator; sin refresh
            for s in STRATEGIES:
                r, _ = backtest_strategy(df, s, cost=COST, cash_rate=CASH)
                rets[(t, s)] = r
        except Exception as e:
            errors.append(f"{t}: {e}")

    last_year = max(r.index[-1].year for r in rets.values())
    years = list(range(START_YEAR, last_year + 1))
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "start_year": START_YEAR,
           "note": "Estrategias de reglas fijas: cada pliego anual es out-of-sample respecto a todo lo anterior.",
           "strategies": {}, "errors": errors}

    for s in STRATEGIES:
        portfolio = pd.DataFrame({t: rets[(t, s)] for t in UNIVERSE if (t, s) in rets}).mean(axis=1)
        folds, fold_sharpes = [], []
        oos_parts = []
        for y in years:
            test = portfolio[(portfolio.index >= f"{y}-01-01") & (portfolio.index <= f"{y}-12-31")]
            st = fold_stats(test)
            if st is None:
                continue
            folds.append({"year": y, **st})
            if st["sharpe"] is not None:
                fold_sharpes.append(st["sharpe"])
            oos_parts.append(test)
        oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
        oos_eq = (1 + oos.fillna(0.0)).cumprod()
        oos_dd = oos_eq / oos_eq.cummax() - 1
        sd = oos.std(ddof=1)
        out["strategies"][s] = {
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

    p = ROOT / "docs" / "walkforward.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} ({p.stat().st_size} bytes), errors: {errors}")


if __name__ == "__main__":
    main()
