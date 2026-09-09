"""Estrategia #4 "Rotacion Semis" (nombre canonico: roc126 top2 vol-escalada).

Universo: NVDA, AVGO, AMD, TSM, ASML.
Cada dia: ROC de 126 dias de los 5; se mantienen los 2 con mayor ROC, solo si
su ROC > 0 (si ninguno es positivo, 100% cash). Reparto equitativo entre las
patas seleccionadas; cada pata se escala por min(1, 0.25/vol20). Coste 0.1%
por cambio de pata; cash remunerado segun config.
Senal al cierre, ejecutable desde la barra siguiente (convencion del proyecto).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SEMIS = ["NVDA", "AVGO", "AMD", "TSM", "ASML"]
ROC_DAYS = 126
VOL_DAYS = 20
TARGET_VOL = 0.25


def target_weights(closes: pd.DataFrame) -> pd.DataFrame:
    """Pesos objetivo diarios (senal al cierre de ese dia)."""
    roc = closes.pct_change(ROC_DAYS)
    vol = closes.pct_change().rolling(VOL_DAYS).std(ddof=1) * np.sqrt(252)
    W = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i in range(len(closes)):
        r = roc.iloc[i].dropna()
        pos = r[r > 0].sort_values(ascending=False)
        sel = list(pos.index[:2])
        if not sel:
            continue
        n = len(sel)
        for t in sel:
            v = vol[t].iloc[i]
            scale = min(1.0, TARGET_VOL / v) if v and v > 1e-9 else 0.0
            W.iloc[i, W.columns.get_loc(t)] = scale / n
    return W


def backtest(closes: pd.DataFrame, W: pd.DataFrame, cost: float, cash_rate: float) -> pd.Series:
    held = W.shift(1).fillna(0.0)
    asset_ret = closes.pct_change().fillna(0.0)
    invested = held.sum(axis=1).clip(upper=1.0)
    turnover = held.diff().abs().fillna(held.abs()).sum(axis=1)
    cash_daily = cash_rate / 252
    return (asset_ret * held).sum(axis=1) + cash_daily * (1 - invested) - turnover * cost


def rotation_log(W: pd.DataFrame) -> list[dict]:
    """Cambios en el conjunto de patas mantenidas."""
    out, prev = [], None
    for d, row in W.iterrows():
        sel = frozenset(t for t, w in row.items() if w > 0)
        if sel != prev:
            out.append({"date": str(d.date()),
                        "legs": sorted(sel),
                        "weights": {t: round(float(row[t]), 4) for t in sorted(sel)}})
            prev = sel
    return out
