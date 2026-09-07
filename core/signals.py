from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metrics import signal_frame


@dataclass
class Signal:
    ticker: str
    kind: str
    side: str
    date: str
    price: float
    score: int
    reason: str


# Signal Lab: intentionally broad, so validation can tell us which families
# deserve further research rather than selecting a winner in advance.
STRATEGY_NAMES = (
    "SMA200", "SMA50", "EMA50", "SMA50>SMA200", "EMA20>EMA50",
    "SMA200+Volume", "ROC60", "ROC120", "ROC252", "Donchian20",
    "Donchian50", "Donchian100", "Donchian200", "EMA10>EMA30", "EMA20>EMA100",
    "SMA200+ROC60", "SMA200+RSI50", "SMA200+ADX25", "Donchian50+Volume",
    "ATRfilter+SMA200", "EMA20>EMA50+ADX25", "MeanReversionZ",
)

# Backtest-only strategies with continuous position sizing (no alert signals):
# vol-target weight = target_vol / realized_vol, clipped, rebalanced every N sessions.
VOLTARGET_STRATEGIES = ("VolTarget10", "VolTarget12")
VOLTARGET_PARAMS = {"VolTarget10": 0.10, "VolTarget12": 0.12}

STATEFUL_STRATEGIES = {
    "SMA200+Volume", "Donchian20", "Donchian50", "Donchian100",
    "Donchian200", "Donchian50+Volume", "MeanReversionZ",
}


def _raw_regime(x: pd.DataFrame, strategy: str) -> pd.Series:
    c = x["Close"]
    if strategy == "SMA200": return c > x["SMA200"]
    if strategy == "SMA50": return c > x["SMA50"]
    if strategy == "EMA50": return c > x["EMA50"]
    if strategy == "SMA50>SMA200": return x["SMA50"] > x["SMA200"]
    if strategy == "EMA20>EMA50": return x["EMA20"] > x["EMA50"]
    if strategy == "ROC60": return x["ROC60"] > 0
    if strategy == "ROC120": return x["ROC120"] > 0
    if strategy == "ROC252": return x["ROC252"] > 0
    if strategy == "EMA10>EMA30": return x["EMA10"] > x["EMA30"]
    if strategy == "EMA20>EMA100": return x["EMA20"] > x["EMA100"]
    if strategy == "SMA200+ROC60": return (c > x["SMA200"]) & (x["ROC60"] > 0)
    if strategy == "SMA200+RSI50": return (c > x["SMA200"]) & (x["RSI14"] > 50)
    if strategy == "SMA200+ADX25": return (c > x["SMA200"]) & (x["ADX14"] > 25)
    if strategy == "ATRfilter+SMA200": return (c > x["SMA200"]) & (x["ATR_pct"] <= x["ATR_pct_Q75_252"])
    if strategy == "EMA20>EMA50+ADX25": return (x["EMA20"] > x["EMA50"]) & (x["ADX14"] > 25)
    # For volume-confirmed breakouts, Donchian channels and mean reversion,
    # positions are stateful and handled in _stateful_position/strategy_position.
    if strategy == "SMA200+Volume": return c > x["SMA200"]
    if strategy == "Donchian20": return c > x["DonchianHigh20"]
    if strategy == "Donchian50": return c > x["DonchianHigh50"]
    if strategy == "Donchian100": return c > x["DonchianHigh100"]
    if strategy == "Donchian200": return c > x["DonchianHigh200"]
    if strategy == "Donchian50+Volume": return c > x["DonchianHigh50"]
    raise ValueError(f"Unknown strategy: {strategy}")


def _apply_confirmation(regime: pd.Series, days: int) -> pd.Series:
    """Only change state after the raw regime holds for `days` consecutive sessions."""
    regime = regime.fillna(False).astype(int)
    if days <= 1:
        return regime
    out = []
    cur = 0
    prev = int(regime.iloc[0]) if len(regime) else 0
    streak = 0
    for v in regime:
        v = int(v)
        streak = streak + 1 if v == prev else 1
        prev = v
        if streak >= days:
            cur = v
        out.append(cur)
    return pd.Series(out, index=regime.index, dtype=int)


def _stateful_position(x: pd.DataFrame, strategy: str) -> pd.Series:
    pos = pd.Series(0, index=x.index, dtype=int)
    for i in range(len(x)):
        if i == 0:
            continue
        c = x["Close"].iloc[i]
        if strategy == "SMA200+Volume":
            enter = bool(c > x["SMA200"].iloc[i] and x["VolumeRatio20"].iloc[i] >= 1.5)
            exit_ = bool(c <= x["SMA200"].iloc[i])
        elif strategy == "Donchian20":
            enter = bool(c > x["DonchianHigh20"].iloc[i])
            exit_ = bool(c < x["DonchianLow20"].iloc[i])
        elif strategy == "Donchian50":
            enter = bool(c > x["DonchianHigh50"].iloc[i])
            exit_ = bool(c < x["DonchianLow50"].iloc[i])
        elif strategy == "Donchian100":
            enter = bool(c > x["DonchianHigh100"].iloc[i])
            exit_ = bool(c < x["DonchianLow100"].iloc[i])
        elif strategy == "Donchian200":
            enter = bool(c > x["DonchianHigh200"].iloc[i])
            exit_ = bool(c < x["DonchianLow200"].iloc[i])
        elif strategy == "Donchian50+Volume":
            enter = bool(c > x["DonchianHigh50"].iloc[i] and x["VolumeRatio20"].iloc[i] >= 1.5)
            exit_ = bool(c < x["DonchianLow50"].iloc[i])
        elif strategy == "MeanReversionZ":
            # Long-only oversold setup: enter on a deep 20-day z-score dip while
            # the long-term trend is up and daily ranges are meaningful;
            # exit when price reverts to its 20-day mean or the trend breaks.
            enter = bool(
                x["Z20"].iloc[i] <= -2.25
                and c > x["SMA200"].iloc[i]
                and x["ATR_pct"].iloc[i] >= 1.0
            )
            exit_ = bool(x["Z20"].iloc[i] >= 0 or c <= x["SMA200"].iloc[i])
        else:
            pos.iloc[i] = pos.iloc[i - 1]
            continue
        if pos.iloc[i - 1] == 0 and enter:
            pos.iloc[i] = 1
        elif pos.iloc[i - 1] == 1 and exit_:
            pos.iloc[i] = 0
        else:
            pos.iloc[i] = pos.iloc[i - 1]
    return pos


def strategy_position(df: pd.DataFrame, strategy: str, confirmation_days: int = 1) -> pd.Series:
    x = signal_frame(df)
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy in STATEFUL_STRATEGIES:
        return _stateful_position(x, strategy)
    regime = _raw_regime(x, strategy)
    return _apply_confirmation(regime, confirmation_days)


def voltarget_weights(df: pd.DataFrame, target_vol: float, window: int = 20,
                      rebalance_days: int = 5, max_weight: float = 1.0) -> pd.Series:
    """Continuous vol-target sizing: weight = target_vol / realized_vol,
    clipped to [0, max_weight], updated only every `rebalance_days` sessions."""
    c = df["Close"]
    rv = c.pct_change().rolling(window).std(ddof=1) * np.sqrt(252)
    raw = (target_vol / rv).clip(lower=0.0, upper=max_weight)
    w = raw.iloc[::rebalance_days].reindex(raw.index).ffill()
    return w.fillna(0.0).astype(float)


def strategy_weights(df: pd.DataFrame, strategy: str, confirmation_days: int = 1,
                     voltarget: dict | None = None) -> pd.Series:
    """Position series used by the backtest: 0/1 for signal strategies,
    fractional weights for the VolTarget family."""
    if strategy in VOLTARGET_STRATEGIES:
        p = voltarget or {}
        return voltarget_weights(
            df, VOLTARGET_PARAMS[strategy],
            window=int(p.get("window", 20)),
            rebalance_days=int(p.get("rebalance_days", 5)),
            max_weight=float(p.get("max_weight", 1.0)),
        )
    return strategy_position(df, strategy, confirmation_days=confirmation_days).astype(float)


def recent_signals(ticker: str, df: pd.DataFrame, strategy: str,
                   lookback_days: int = 1, confirmation_days: int = 1) -> list[Signal]:
    """Signals fired on any of the last `lookback_days` sessions (oldest first),
    so a missed run does not lose a signal that is still recent."""
    if strategy not in STRATEGY_NAMES:
        raise ValueError(f"Unknown strategy: {strategy}")
    pos = strategy_position(df, strategy, confirmation_days=confirmation_days)
    if len(pos) < 2:
        return []
    x = signal_frame(df)
    out = []
    n = min(int(lookback_days), len(pos) - 1)
    for k in range(n, 0, -1):
        i = len(pos) - k
        last, prev = int(pos.iloc[i]), int(pos.iloc[i - 1])
        if last == prev:
            continue
        price = float(x["Close"].iloc[i])
        side = "LONG" if last else "EXIT"
        reason = f"Regime changed for {strategy}"
        if strategy in {"SMA200+Volume", "Donchian50+Volume"} and last:
            reason += f" (VolumeRatio20={x['VolumeRatio20'].iloc[i]:.2f}x)"
        out.append(Signal(ticker, strategy, side, str(x.index[i].date()), price, 85, reason))
    return out


def strategy_signal(ticker: str, df: pd.DataFrame, strategy: str,
                    confirmation_days: int = 1) -> Signal | None:
    sigs = recent_signals(ticker, df, strategy, lookback_days=1,
                          confirmation_days=confirmation_days)
    return sigs[-1] if sigs else None


def sma200_signal(ticker: str, df: pd.DataFrame, confirmation_days: int = 1):
    return strategy_signal(ticker, df, "SMA200", confirmation_days=confirmation_days)
