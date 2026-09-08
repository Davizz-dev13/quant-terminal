"""Estrategia Mag7 SMA200 semanal (contrarian) de David.

Entrada: el cierre semanal cruza A LA BAJA la SMA200 semanal.
Salida (regla de David): cuando el RSI(14) semanal supera 70 (sobrecompra) la
posicion queda "armada"; desde ese momento, si el cierre cae mas de un 5% desde
el maximo cierre posterior al armado, se vende.

Senales al cierre de la semana; la posicion se ejecuta desde la semana
siguiente (misma convencion que el resto del proyecto). Coste por operacion y
remuneracion del cash segun config/settings.yaml.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ENTRY_DROP = 0.0        # cruce simple a la baja
RSI_LEVEL = 70.0
EXIT_DD = 0.05          # caida del 5% desde el maximo tras sobrecompra


def weekly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    c = daily["Close"].resample("W-FRI").last().dropna()
    x = pd.DataFrame({"Close": c})
    x["SMA200"] = c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))
    return x


def weekly_position(x: pd.DataFrame) -> pd.Series:
    """0/1 por semana (senal al cierre de esa semana)."""
    pos = pd.Series(0, index=x.index, dtype=int)
    state, armed, peak = 0, False, np.nan
    c, sma, rsi = x["Close"], x["SMA200"], x["RSI14"]
    for i in range(1, len(x)):
        if state == 0:
            if not np.isnan(sma.iloc[i]) and c.iloc[i - 1] > sma.iloc[i - 1] and c.iloc[i] < sma.iloc[i]:
                state, armed, peak = 1, False, np.nan
        else:
            if not armed and not np.isnan(rsi.iloc[i]) and rsi.iloc[i] >= RSI_LEVEL:
                armed, peak = True, c.iloc[i]
            if armed:
                peak = max(peak, c.iloc[i])
                if c.iloc[i] <= peak * (1 - EXIT_DD):
                    state, armed, peak = 0, False, np.nan
        pos.iloc[i] = state
    return pos


def trades_from_position(x: pd.DataFrame, pos: pd.Series) -> list[dict]:
    """Operaciones cerradas/abiertas segun las senales (precios del cierre de senal)."""
    trades, entry_i = [], None
    for i in range(len(pos)):
        if pos.iloc[i] == 1 and (i == 0 or pos.iloc[i - 1] == 0):
            entry_i = i
        elif pos.iloc[i] == 0 and i > 0 and pos.iloc[i - 1] == 1 and entry_i is not None:
            e, x_ = x["Close"].iloc[entry_i], x["Close"].iloc[i]
            trades.append({
                "entry_date": str(x.index[entry_i].date()), "entry_price": round(float(e), 2),
                "exit_date": str(x.index[i].date()), "exit_price": round(float(x_), 2),
                "return_pct": round((x_ / e - 1) * 100, 2), "open": False,
            })
            entry_i = None
    if entry_i is not None:
        e = x["Close"].iloc[entry_i]
        last = x["Close"].iloc[-1]
        trades.append({
            "entry_date": str(x.index[entry_i].date()), "entry_price": round(float(e), 2),
            "exit_date": None, "exit_price": None,
            "return_pct": round((last / e - 1) * 100, 2), "open": True,
        })
    return trades


def weekly_backtest(x: pd.DataFrame, pos: pd.Series, cost: float, cash_rate: float) -> pd.Series:
    held = pos.shift(1).fillna(0.0).astype(float)  # ejecutable desde la semana siguiente
    turnover = held.diff().abs().fillna(held.abs())
    asset_ret = x["Close"].pct_change().fillna(0.0)
    cash_w = cash_rate / 52
    return asset_ret * held + cash_w * (1 - held) - turnover * cost
