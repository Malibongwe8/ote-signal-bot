"""Deriv synthetic-index market data adapter.

Pulls candles for Deriv's synthetic indices (Volatility 10/25/50/75/90/100) over
Deriv's public WebSocket API. Market data needs no account or token -- only an
app_id.

IMPORTANT: Deriv synthetics are deliberately random-walk generators. There is no
real economic process behind them, so technical/price-action patterns have NO
predictive edge on them by construction. They are included for completeness and
context only -- signals on them are noise, not opportunities.

Endpoint: wss://ws.derivws.com/websockets/v3?app_id=<APP_ID>
"""
import json

import pandas as pd

# Deriv symbols for the Volatility N Index synthetics.
DERIV_SYMBOLS = {
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index",
    "R_50": "Volatility 50 Index",
    "R_75": "Volatility 75 Index",
    "R_90": "Volatility 90 Index",
    "R_100": "Volatility 100 Index",
}

# Scanner timeframe -> Deriv candle granularity (seconds).
TF_TO_GRANULARITY = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400,
}

DERIV_WS = "wss://ws.derivws.com/websockets/v3"
APP_ID = 1089   # Deriv's public app_id; no auth needed for market data


def _fetch_candles(symbol: str, granularity: int, count: int = 500):
    """One short-lived WebSocket round-trip -> OHLC DataFrame (UTC index)."""
    import websocket
    ws = websocket.create_connection(f"{DERIV_WS}?app_id={APP_ID}", timeout=30)
    try:
        ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": granularity,
            "style": "candles",
        }))
        raw = ws.recv()
    finally:
        ws.close()

    msg = json.loads(raw)
    if "candles" not in msg:
        raise RuntimeError(f"Deriv error: {msg.get('error', msg)}")

    df = pd.DataFrame(msg["candles"])
    df["time"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close"]].astype(float)
    return df


def fetch_deriv(symbol: str, ltf: str, htf: str, count: int = 500):
    """Return (df_ltf, df_htf) for a Deriv synthetic, matching fetch_recent()."""
    gl = TF_TO_GRANULARITY.get(ltf, 900)
    gh = TF_TO_GRANULARITY.get(htf, 3600)
    df_ltf = _fetch_candles(symbol, gl, count)
    df_htf = _fetch_candles(symbol, gh, count)
    return df_ltf, df_htf


def check_connectivity(count: int = 100):
    """Try every synthetic once; return a short human-readable status string."""
    parts = []
    for sym in DERIV_SYMBOLS:
        try:
            df, _ = fetch_deriv(sym, "15m", "1h", count)
            parts.append(f"{sym}:{len(df)}bars")
        except Exception as e:
            parts.append(f"{sym}:ERR({str(e)[:40]})")
    return "Deriv check: " + ", ".join(parts)
