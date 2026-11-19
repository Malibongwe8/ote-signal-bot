"""
Multi-timeframe live signal scanner for the OTE + SMC strategy.
================================================================
Scans 1m, 5m, 15m, 30m and 1h timeframes, runs the OTE+SMC strategy on each,
and emits ENTRY / EXIT signals. It NEVER stops after wins.

Delivery:
  - console + signals_log.jsonl (always)
  - Telegram via Bot API if --telegram-token and --telegram-chat are supplied
  - generic JSON webhook via --webhook

DISCLAIMER: Research / education only. NOT financial advice. No signal is
guaranteed. Trading risks substantial loss. Verify every signal yourself.
"""
import argparse
import atexit
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: pip install yfinance pandas numpy", file=sys.stderr)
    sys.exit(1)

from ote_strategy import (run_strategy, _atr, _fractal_swings, find_fvgs,
                          find_order_blocks, find_liquidity_sweeps,
                          htf_bias, in_killzone, ote_zones, price_in_ote, Trade)

STATE_FILE = "scanner_state.json"
LOG_FILE = "signals_log.jsonl"
LOCK_FILE = "/tmp/ote_scanner.lock"


def acquire_lock() -> bool:
    """Prevent two scanner instances from running at once (which duplicates
    signals). Returns False if another live instance already holds the lock."""
    if os.path.exists(LOCK_FILE):
        try:
            pid = int(open(LOCK_FILE).read().strip())
            os.kill(pid, 0)  # raises if the process is not alive
            print(f"Another scanner is already running (pid {pid}). Exiting.",
                  file=sys.stderr)
            return False
        except (ProcessLookupError, ValueError, OSError):
            pass  # stale lock, take it over
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(LOCK_FILE) and
                    int(open(LOCK_FILE).read().strip() or 0) == os.getpid() and
                    os.remove(LOCK_FILE))
    return True

# Default instruments: FX, crypto, and CBOE volatility indices.
# NOTE: volatility indices (^VIX etc.) are NOT directly tradeable — they are
# used here as market-context / risk filters, not as tradeable pairs.
# (^RVX and ^SKEW return no data on Yahoo and are excluded.)
DEFAULT_SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "BTC-USD",
    "^VIX", "^VXN", "^VXD", "^VVIX", "^VIX3M",
]

# timeframe -> (higher-timeframe for bias, yfinance max period allowed)
TF_CONFIG = {
    "1m":  ("5m",  "7d"),
    "5m":  ("15m", "60d"),
    "15m": ("1h",  "60d"),
    "30m": ("1h",  "60d"),
    "1h":  ("1d",  "730d"),
}


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"open_positions": {}, "seen_entries": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_log(signal: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(signal, default=str) + "\n")


def fetch_recent(symbol: str, ltf: str, htf: str, period: str):
    df_ltf = yf.download(symbol, period=period, interval=ltf, progress=False,
                         auto_adjust=True)
    df_htf = yf.download(symbol, period=period, interval=htf, progress=False,
                         auto_adjust=True)
    for df in (df_ltf, df_htf):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
    # align to required columns
    df_ltf = df_ltf[[c for c in ["open", "high", "low", "close"] if c in df_ltf.columns]]
    df_htf = df_htf[[c for c in ["open", "high", "low", "close"] if c in df_htf.columns]]
    df_ltf = df_ltf.dropna()
    df_htf = df_htf.dropna()
    return df_ltf, df_htf


def find_new_signal(df_ltf, df_htf, symbol, tf, rr, min_confluence, window=400):
    # Bound the analysis window so intraday (1m-30m) scans stay fast.
    if len(df_ltf) > window:
        df_ltf = df_ltf.iloc[-window:]
    atr = _atr(df_ltf, 14)
    swing_highs, swing_lows = _fractal_swings(df_ltf, lookback=3)
    fvgs = find_fvgs(df_ltf)
    obs = find_order_blocks(df_ltf, threshold=1.0)
    sweeps = find_liquidity_sweeps(df_ltf, swing_highs, swing_lows)
    bias = htf_bias(df_htf)

    i = len(df_ltf) - 1
    ts = df_ltf.index[i]
    c = df_ltf["close"].values
    if not in_killzone(ts):
        return None
    direction = {"bullish": "long", "bearish": "short"}.get(bias)
    if not direction:
        return None

    if direction == "long":
        rl = [s for s in swing_lows if s.index < i]
        rh = [s for s in swing_highs if s.index < i]
        if not rl or not rh:
            return None
        sw_low, sw_high = rl[-1], rh[-1]
        if sw_low.index >= sw_high.index or sw_high.price <= sw_low.price:
            return None
        zones = ote_zones(sw_low.price, sw_high.price, "long")
    else:
        rh = [s for s in swing_highs if s.index < i]
        rl = [s for s in swing_lows if s.index < i]
        if not rh or not rl:
            return None
        sw_high, sw_low = rh[-1], rl[-1]
        if sw_high.index >= sw_low.index or sw_high.price <= sw_low.price:
            return None
        zones = ote_zones(sw_high.price, sw_low.price, "short")

    if not price_in_ote(c[i], zones, direction):
        return None

    confluence = []
    for f in fvgs:
        if f.index < i and (f.bullish == (direction == "long")):
            if price_in_ote((f.top + f.bottom) / 2, zones, direction):
                confluence.append("FVG")
                break
    for ob in obs:
        if ob.index < i and (ob.bullish == (direction == "long")):
            if price_in_ote((ob.top + ob.bottom) / 2, zones, direction):
                confluence.append("OB")
                break
    for s in sweeps:
        if s.index < i:
            if (direction == "long" and s.kind == "SSL") or \
               (direction == "short" and s.kind == "BSL"):
                confluence.append(f"Sweep({s.kind})")
                break
    if len(confluence) < min_confluence:
        return None

    entry = float(c[i])
    a = atr[i]
    if a <= 0:
        return None
    # Structurally-correct SL: just beyond the swing that defined the OTE leg
    # (the invalidation level), capped at 3*ATR.
    if direction == "long":
        raw_stop = sw_low.price - 0.1 * a
        risk = min(entry - raw_stop, 3 * a)
        if risk <= 0:
            return None
        stop, target = entry - risk, entry + risk * rr
    else:
        raw_stop = sw_high.price + 0.1 * a
        risk = min(raw_stop - entry, 3 * a)
        if risk <= 0:
            return None
        stop, target = entry + risk, entry - risk * rr
    return {
        "type": "ENTRY", "symbol": symbol, "timeframe": tf,
        "direction": direction, "bar_time": str(ts),
        "entry": round(entry, 5), "stop": round(stop, 5),
        "target": round(target, 5), "rr": rr, "htf_bias": bias,
        "confluence": confluence,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def check_open_position(pos: dict, df_ltf) -> dict | None:
    last = df_ltf.iloc[-1]
    hi, lo = float(last["high"]), float(last["low"])
    d = pos["direction"]
    if d == "long":
        if lo <= pos["stop"]:
            return _exit_signal(pos, "loss", pos["stop"])
        if hi >= pos["target"]:
            return _exit_signal(pos, "win", pos["target"])
    else:
        if hi >= pos["stop"]:
            return _exit_signal(pos, "loss", pos["stop"])
        if lo <= pos["target"]:
            return _exit_signal(pos, "win", pos["target"])
    return None


def _exit_signal(pos, outcome, price):
    return {
        "type": "EXIT", "symbol": pos["symbol"], "timeframe": pos.get("timeframe"),
        "direction": pos["direction"], "outcome": outcome,
        "entry": pos["entry"], "exit": round(price, 5),
        "stop": pos.get("stop"), "target": pos.get("target"), "rr": pos.get("rr"),
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def _format(signal: dict) -> str:
    if signal["type"] == "ENTRY":
        return (f"OTE ENTRY [{signal.get('timeframe')}] {signal['symbol']} "
                f"{signal['direction'].upper()}\n"
                f"Entry {signal['entry']} | SL {signal['stop']} | TP {signal['target']}\n"
                f"Confluence: {'+'.join(signal['confluence'])} | bias {signal['htf_bias']}")
    return (f"OTE EXIT [{signal.get('timeframe')}] {signal['symbol']} "
            f"{signal['direction'].upper()} -> {signal['outcome'].upper()}\n"
            f"Entry {signal['entry']} | Exit {signal['exit']}")


def send_telegram(signal: dict, token: str, chat: str):
    try:
        import urllib.request
        import urllib.parse
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": _format(signal)}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
    except Exception as e:
        print(f"  (telegram send failed: {e})", file=sys.stderr)


def send_telegram_photo(signal, df_ltf, token, chat):
    """Render a TradingView-style chart and send it with the signal as caption."""
    try:
        import tempfile, os as _os
        from chart import render_chart
        path = _os.path.join(tempfile.gettempdir(),
                             f"sig_{signal.get('symbol','x')}_{signal.get('timeframe','')}.png")
        render_chart(df_ltf, signal, path)
        import urllib.request, uuid
        boundary = uuid.uuid4().hex
        caption = _format(signal)
        with open(path, "rb") as f:
            img = f.read()
        parts = []
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"chart.png\"\r\nContent-Type: image/png\r\n\r\n".encode())
        parts.append(img)
        parts.append(f"\r\n--{boundary}--\r\n".encode())
        body = b"".join(parts)
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendPhoto", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        urllib.request.urlopen(req, timeout=25)
    except Exception as e:
        print(f"  (telegram photo failed: {e})", file=sys.stderr)


def emit(signal: dict, webhook, tg_token, tg_chat, df_ltf=None):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] " + _format(signal).replace("\n", " | "),
          flush=True)
    append_log(signal)
    if tg_token and tg_chat:
        if df_ltf is not None:
            send_telegram_photo(signal, df_ltf, tg_token, tg_chat)
        else:
            send_telegram(signal, tg_token, tg_chat)
    if webhook:
        try:
            import urllib.request
            req = urllib.request.Request(webhook, data=json.dumps(signal).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"  (webhook failed: {e})", file=sys.stderr)


def scan_once(symbols, timeframes, rr, min_confluence, webhook, tg_token, tg_chat):
    state = load_state()
    open_positions = state.get("open_positions", {})
    seen = set(state.get("seen_entries", []))

    for symbol in symbols:
        for tf in timeframes:
            htf, period = TF_CONFIG[tf]
            try:
                df_ltf, df_htf = fetch_recent(symbol, tf, htf, period)
            except Exception as e:
                print(f"  (fetch failed {symbol} {tf}: {e})", file=sys.stderr)
                continue
            if len(df_ltf) < 60 or len(df_htf) < 60:
                continue

            key = f"{symbol}|{tf}"
            pos = open_positions.get(key)
            if pos:
                exit_sig = check_open_position(pos, df_ltf)
                if exit_sig:
                    emit(exit_sig, webhook, tg_token, tg_chat, df_ltf=df_ltf)
                    open_positions.pop(key, None)
                    continue

            sig = find_new_signal(df_ltf, df_htf, symbol, tf, rr, min_confluence)
            if sig and key not in open_positions:
                k = f"{key}|{sig['bar_time']}|{sig['direction']}"
                if k not in seen:
                    emit(sig, webhook, tg_token, tg_chat, df_ltf=df_ltf)
                    seen.add(k)
                    open_positions[key] = sig

    state["open_positions"] = open_positions
    state["seen_entries"] = list(seen)[-500:]
    save_state(state)


def run_test(tg_token, tg_chat, webhook):
    """Emit sample signals through the real delivery path so the user can see
    exactly how a signal will look. Nothing is sent unless Telegram creds exist."""
    now = datetime.now(timezone.utc).isoformat()
    samples = [
        {"type": "ENTRY", "symbol": "BTC-USD", "timeframe": "15m",
         "direction": "short", "bar_time": now, "entry": 81250.5,
         "stop": 81940.0, "target": 79871.5, "rr": 2.0, "htf_bias": "bearish",
         "confluence": ["FVG", "Sweep(BSL)"], "detected_at": now},
        {"type": "EXIT", "symbol": "BTC-USD", "timeframe": "15m",
         "direction": "short", "outcome": "win", "entry": 81250.5,
         "exit": 79871.5, "detected_at": now},
    ]
    print("=== SIGNAL PREVIEW (exactly how each alert is formatted) ===\n")
    for s in samples:
        print("-" * 44)
        print(_format(s))
        print("-" * 44 + "\n")
    if tg_token and tg_chat:
        for s in samples:
            send_telegram(s, tg_token, tg_chat)
        print("Sent 2 test messages to Telegram.")
    else:
        print("No Telegram token configured -> PREVIEW ONLY, nothing was sent.")


def main():
    ap = argparse.ArgumentParser(description="Multi-TF OTE+SMC signal scanner")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframes", default="1m,5m,15m,30m,1h")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--min-confluence", type=int, default=2)
    ap.add_argument("--interval", type=int, default=60, help="seconds between scans")
    ap.add_argument("--watch", action="store_true", help="loop forever")
    ap.add_argument("--test", action="store_true",
                    help="preview how signals are formatted (no real scan)")
    ap.add_argument("--webhook", default=None)
    ap.add_argument("--telegram-token", default=None)
    ap.add_argument("--telegram-chat", default=None)
    args = ap.parse_args()

    if args.test:
        run_test(args.telegram_token, args.telegram_chat, args.webhook)
        return

    if not acquire_lock():
        sys.exit(1)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip() in TF_CONFIG]
    tgt = "Telegram" if (args.telegram_token and args.telegram_chat) else \
          ("webhook" if args.webhook else "console only")
    print(f"Scanner: {symbols} | TF {timeframes} | RR {args.rr} | delivery: {tgt}")
    print(f"Mode: {'continuous' if args.watch else 'single scan'}\n")

    args_kw = dict(webhook=args.webhook, tg_token=args.telegram_token,
                   tg_chat=args.telegram_chat)
    scan_once(symbols, timeframes, args.rr, args.min_confluence, **args_kw)
    if args.watch:
        while True:
            time.sleep(args.interval)
            try:
                scan_once(symbols, timeframes, args.rr, args.min_confluence, **args_kw)
            except KeyboardInterrupt:
                print("\nStopped by user.")
                break
            except Exception as e:
                print(f"  (scan error: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
