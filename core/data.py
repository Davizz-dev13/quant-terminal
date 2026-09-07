from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

def _path(ticker: str, period: str, interval: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("=", "_").replace("/", "_")
    return CACHE / f"{safe}_{period}_{interval}.parquet"

def history(ticker: str, period: str = "10y", interval: str = "1d", refresh: bool = False) -> pd.DataFrame:
    p = _path(ticker, period, interval)
    if p.exists() and not refresh and time.time() - p.stat().st_mtime < 6 * 3600:
        return pd.read_parquet(p)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.title() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.to_parquet(p)
    return df

def latest_snapshot(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            d = history(t, "1y", "1d")
            c = d["Close"]
            rows.append({
                "Ticker": t,
                "Price": float(c.iloc[-1]),
                "1D %": float(c.pct_change().iloc[-1] * 100),
                "1W %": float(c.pct_change(5).iloc[-1] * 100),
                "1M %": float(c.pct_change(21).iloc[-1] * 100),
                "SMA200": float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None,
                "Dist SMA200 %": float((c.iloc[-1] / c.rolling(200).mean().iloc[-1] - 1) * 100) if len(c) >= 200 else None,
            })
        except Exception as e:
            rows.append({"Ticker": t, "Error": str(e)})
    return pd.DataFrame(rows)
