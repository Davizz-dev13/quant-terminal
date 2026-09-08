"""La estrategia de David aplicada al universo del panel principal.

Entrada: el cierre semanal cruza a la baja la SMA200 semanal.
Salida: RSI(14) semanal supera el nivel configurado (70 por defecto; el oro
usa 87 segun config/settings.yaml -> mag7w.rsi_level_by_asset) y luego el
cierre cae mas de un 5% desde el maximo posterior.

El historico completo calienta la SMA200 semanal; la posicion semanal
(ejecutable desde la semana siguiente) se traslada a barras diarias para
combinarse con el resto de estrategias del panel.
"""
from __future__ import annotations
import pandas as pd

from scripts.mag7w_core import weekly_frame, weekly_position


def david_weekly_daily_returns(daily: pd.DataFrame, cost: float, cash_rate: float,
                               rsi_level: float = 70.0):
    """Devuelve (retorno_diario, posicion_diaria) de la estrategia de David."""
    x = weekly_frame(daily)
    pos_w = weekly_position(x, rsi_level)
    held_w = pos_w.shift(1).fillna(0.0).astype(float)  # ejecutable la semana siguiente
    held_d = held_w.reindex(daily.index, method="ffill").fillna(0.0)
    asset_ret = daily["Close"].pct_change().fillna(0.0)
    turnover = held_d.diff().abs().fillna(held_d.abs())
    cash_daily = cash_rate / 252
    ret = asset_ret * held_d + cash_daily * (1.0 - held_d) - turnover * cost
    return ret, held_d
