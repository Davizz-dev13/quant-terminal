from __future__ import annotations
import numpy as np
import pandas as pd
from .signals import strategy_weights, STRATEGY_NAMES, VOLTARGET_STRATEGIES

ALL_STRATEGIES = STRATEGY_NAMES + VOLTARGET_STRATEGIES


def _stats(ret: pd.Series, positions: pd.Series | None = None) -> dict:
    ret = ret.fillna(0.0)
    if not len(ret):
        return {}
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    sd = ret.std(ddof=1)
    vol = sd * np.sqrt(252)
    # Guard against degenerate (near-constant) return series, e.g. a strategy
    # that never trades and only earns the cash rate.
    sharpe = ret.mean() / sd * np.sqrt(252) if sd and sd > 1e-12 else np.nan
    trades = int((positions.diff().abs().fillna(positions.abs()) > 1e-12).sum()) if positions is not None else 0
    years = max((ret.index[-1] - ret.index[0]).days / 365.25, 1 / 252) if hasattr(ret.index, "__getitem__") else len(ret) / 252
    return {
        "CAGR": eq.iloc[-1] ** (1 / years) - 1,
        "Sharpe": sharpe,
        "Vol": vol,
        "MaxDD": dd.min(),
        "Trades": trades,
        "TotalReturn": eq.iloc[-1] - 1,
        "Exposure": float(positions.mean()) if positions is not None else np.nan,
    }


def buy_and_hold_backtest(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    close = df["Close"]
    ret = close.pct_change().fillna(0.0)
    positions = pd.Series(1.0, index=close.index, dtype=float)
    return ret, _stats(ret, positions)


def backtest_strategy(df: pd.DataFrame, strategy: str, cost: float = 0.001,
                      cash_rate: float = 0.0, confirmation_days: int = 1,
                      voltarget: dict | None = None) -> tuple[pd.Series, dict]:
    """Daily backtest. Signals observed at today's close are executable from the
    next bar onward. Uninvested cash earns `cash_rate` (annualized, simple) so
    low-exposure strategies are not unfairly penalized vs buy & hold."""
    if strategy not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    pos = strategy_weights(df, strategy, confirmation_days=confirmation_days, voltarget=voltarget)
    held = pos.shift(1).fillna(0.0).astype(float)
    turnover = held.diff().abs().fillna(held.abs())
    asset_ret = df["Close"].pct_change().fillna(0.0)
    cash_daily = cash_rate / 252
    ret = asset_ret * held + cash_daily * (1.0 - held) - turnover * cost
    return ret, _stats(ret, held)


def sma200_backtest(df: pd.DataFrame, cost=0.001, confirmation=1, cash_rate=0.0):
    return backtest_strategy(df, "SMA200", cost=cost,
                             confirmation_days=confirmation, cash_rate=cash_rate)
