from __future__ import annotations
import argparse, os, time, json
from pathlib import Path
import yaml
from dotenv import load_dotenv
load_dotenv()

from core.data import history
from core.signals import recent_signals, STRATEGY_NAMES
from core.telegram import send

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT/"config/settings.yaml", encoding="utf8"))
STATE_FILE = ROOT/"data"/"alert_state.json"


def load_state():
    if STATE_FILE.exists(): return json.loads(STATE_FILE.read_text())
    return {}

def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=2))

def run_once(dry_run: bool = False):
    state = load_state()
    sent_log = state.get("_last_sent", {})
    alerts_cfg = CFG.get("alerts", {})
    lookback_days = int(alerts_cfg.get("lookback_days", 1))
    cooldown_hours = float(alerts_cfg.get("cooldown_hours", 0))
    only_new = bool(alerts_cfg.get("only_on_new_signal", True))
    confirmation_days = int(CFG.get("signals", {}).get("sma200", {}).get("confirmation_days", 1))
    now = time.time()

    tickers = sorted(set(CFG["market"]["universe"] + CFG["market"]["equities"]))
    signals = []
    for t in tickers:
        try:
            df = history(t, "2y", "1d")
            for strategy in STRATEGY_NAMES:
                cd = confirmation_days if strategy == "SMA200" else 1
                signals.extend(recent_signals(t, df, strategy, lookback_days=lookback_days, confirmation_days=cd))
        except Exception as e:
            print(f"{t}: {e}")

    sent = 0
    for s in signals:
        key = f"{s.ticker}:{s.kind}:{s.side}:{s.date}"
        base = f"{s.ticker}:{s.kind}:{s.side}"
        if only_new and key in state:
            continue
        last_ts = sent_log.get(base)
        if cooldown_hours and last_ts and now - last_ts < cooldown_hours * 3600:
            continue
        text = (f"QUANT ALERT\n{s.ticker} | {s.kind} | {s.side}\n"
                f"Price: {s.price:.4f}\nScore: {s.score}/100\n{s.reason}\nDate: {s.date}")
        if dry_run:
            print("-" * 40)
            print(text)
        else:
            send(text)
            state[key] = int(now)
            sent_log[base] = int(now)
        sent += 1

    if not dry_run:
        state["_last_sent"] = sent_log
        save_state(state)
    mode = "DRY-RUN: " if dry_run else ""
    print(f"{mode}Checked {len(tickers)} assets over the last {lookback_days} sessions; "
          f"{'would send' if dry_run else 'sent'} {sent} new candidate alerts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scan configured strategies and send Telegram alerts.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print alerts instead of sending them; does not touch alert_state.json.")
    args = ap.parse_args()
    run_once(dry_run=args.dry_run)
