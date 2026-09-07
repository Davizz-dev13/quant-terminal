import numpy as np
import pandas as pd


def metrics(df: pd.DataFrame) -> dict:
    c = df["Close"].dropna()
    r = c.pct_change().dropna()
    dd = c / c.cummax() - 1
    ann = 252
    vol = r.std() * np.sqrt(ann)
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() else np.nan
    downside = r[r < 0].std() * np.sqrt(ann)
    sortino = r.mean() * ann / downside if downside else np.nan
    return {
        "Price": c.iloc[-1], "Return 1D": r.iloc[-1], "Return 1M": c.pct_change(21).iloc[-1] if len(c) > 21 else np.nan,
        "Return 1Y": c.pct_change(252).iloc[-1] if len(c) > 252 else np.nan,
        "Volatility": vol, "Sharpe": sharpe, "Sortino": sortino,
        "Max Drawdown": dd.min(), "Skew": r.skew(), "Kurtosis": r.kurtosis(),
        "SMA20": c.rolling(20).mean().iloc[-1], "SMA50": c.rolling(50).mean().iloc[-1], "SMA200": c.rolling(200).mean().iloc[-1],
    }


def signal_frame(df: pd.DataFrame, lookback=20, trend_sma=200) -> pd.DataFrame:
    x = df.copy()
    c = x["Close"]
    x["SMA20"] = c.rolling(20).mean()
    x["SMA50"] = c.rolling(50).mean()
    x["SMA200"] = c.rolling(trend_sma).mean()
    x["EMA10"] = c.ewm(span=10, adjust=False).mean()
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()
    x["EMA30"] = c.ewm(span=30, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA100"] = c.ewm(span=100, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()
    x["STD20"] = c.rolling(lookback).std(ddof=1)
    x["Z20"] = (c - x["SMA20"]) / x["STD20"]

    # ROC windows used by the momentum strategies.
    for n in (20, 60, 120, 252):
        x[f"ROC{n}"] = c.pct_change(n)
    # Donchian channels used by the breakout strategies (20/50/100/200).
    for n in (20, 50, 100, 200):
        x[f"DonchianHigh{n}"] = c.shift(1).rolling(n).max()
        x[f"DonchianLow{n}"] = c.shift(1).rolling(n).min()

    prev = c.shift(1)
    tr = pd.concat([(x["High"] - x["Low"]), (x["High"] - prev).abs(), (x["Low"] - prev).abs()], axis=1).max(axis=1)
    x["TR"] = tr
    x["ATR14"] = tr.rolling(14).mean()
    x["ATR_pct"] = x["ATR14"] / c * 100
    x["ATR_pct_Q75_252"] = x["ATR_pct"].rolling(252).quantile(0.75)

    # RSI(14), Wilder-style smoothing via exponential moving averages.
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI14"] = 100 - (100 / (1 + rs))

    # ADX(14), using Wilder-style exponential smoothing.
    up_move = x["High"].diff()
    down_move = -x["Low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    if "Volume" in x.columns:
        x["VolumeSMA20"] = x["Volume"].rolling(20).mean()
        x["VolumeRatio20"] = x["Volume"] / x["VolumeSMA20"]
    else:
        x["VolumeSMA20"] = np.nan
        x["VolumeRatio20"] = np.nan

    x["AboveSMA200"] = c > x["SMA200"]
    return x
