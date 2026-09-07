import os, sys
from pathlib import Path
import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import history, latest_snapshot
from core.metrics import metrics, signal_frame
from core.backtest import backtest_strategy, buy_and_hold_backtest

CFG = yaml.safe_load(open(ROOT/"config/settings.yaml", encoding="utf8"))
UNIVERSE = CFG["market"]["universe"] + CFG["market"]["equities"]
BT_CFG = CFG.get("backtest", {})
CONFIRMATION_DAYS = int(CFG.get("signals", {}).get("sma200", {}).get("confirmation_days", 1))

st.set_page_config(page_title="Quant Terminal", layout="wide")
st.title("Quant Terminal")
st.caption("Market data + quantitative metrics + validated signal research")

with st.sidebar:
    st.header("Asset")
    ticker = st.selectbox("Ticker", sorted(set(UNIVERSE)))
    period = st.selectbox("History", ["1y", "3y", "5y", "10y"], index=2)
    refresh = st.button("Refresh data")

try:
    df = history(ticker, period, refresh=refresh)
    x = signal_frame(df)
except Exception as e:
    st.error(str(e)); st.stop()

m = metrics(df)
cols = st.columns(6)
for col, (k,v) in zip(cols, [("Price",m["Price"]),("1D",m["Return 1D"]),("1M",m["Return 1M"]),("Vol",m["Volatility"]),("Sharpe",m["Sharpe"]),("Max DD",m["Max Drawdown"])]):
    col.metric(k, f"{v:.2%}" if k in {"1D","1M","Vol","Max DD"} else f"{v:.2f}")

p = go.Figure()
p.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name=ticker))
p.add_trace(go.Scatter(x=x.index, y=x.SMA200, name="SMA200"))
p.update_layout(height=560, xaxis_rangeslider_visible=False)
st.plotly_chart(p, use_container_width=True)

t1,t2,t3 = st.tabs(["Metrics", "Market overview", "Backtest"])
with t1:
    st.dataframe({k:[v] for k,v in m.items()}, use_container_width=True)
    st.write("Latest quantitative state")
    st.dataframe(x[["Close","SMA20","SMA50","SMA200","Z20","ATR_pct","AboveSMA200"]].tail(10), use_container_width=True)
with t2:
    st.dataframe(latest_snapshot(sorted(set(UNIVERSE))), use_container_width=True)
with t3:
    cost = st.number_input("Transaction cost", min_value=0.0, max_value=0.02,
                           value=float(BT_CFG.get("transaction_cost", 0.001)), step=0.0005, format="%.4f")
    cash_rate = st.number_input("Cash rate (annual)", min_value=0.0, max_value=0.20,
                                value=float(BT_CFG.get("cash_rate_annual", 0.0)), step=0.005, format="%.3f")
    rows = []
    curves = {}
    _, bh = buy_and_hold_backtest(df)
    for strategy in ["SMA200", "SMA50", "EMA50", "SMA50>SMA200", "EMA20>EMA50", "SMA200+Volume",
                     "Donchian50", "MeanReversionZ", "VolTarget10", "VolTarget12"]:
        cd = CONFIRMATION_DAYS if strategy == "SMA200" else 1
        r, stats = backtest_strategy(df, strategy, cost=cost, cash_rate=cash_rate,
                                     confirmation_days=cd, voltarget=BT_CFG.get("voltarget", {}))
        rows.append({"Strategy": strategy, **stats, "BH_CAGR": bh["CAGR"], "BH_Sharpe": bh["Sharpe"], "Alpha_CAGR": stats["CAGR"] - bh["CAGR"], "Sharpe_Delta": stats["Sharpe"] - bh["Sharpe"], "DD_Improvement": stats["MaxDD"] - bh["MaxDD"]})
        curves[strategy] = (1 + r).cumprod()
    rows.append({"Strategy":"Buy & Hold", **bh})
    st.dataframe(rows, use_container_width=True)
    st.line_chart(pd.DataFrame(curves))
