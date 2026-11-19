# OTE + SMC Signal Bot

Scans forex pairs, BTC and gold for OTE (Optimal Trade Entry) + SMC setups and
sends them to Telegram — as a message **and** a TradingView-style chart image.

Runs on **GitHub Actions**, scheduled every 5 minutes, so it keeps running
whether or not your own computer is on.

## What it scans

- **Symbols (28):** 26 FX pairs (majors + crosses), `BTC-USD`, `GC=F` (gold)
- **Timeframes:** 5m, 15m, 1h
- **Directions:** both long and short
- **Strategy:** fib OTE zone (0.62–0.79) + HTF bias + multi-timeframe structure
  agreement + confluence (FVG / Order Block / Liquidity Sweep), entries inside
  London & NY killzones. **Dynamic stop** sits beyond the swing that defined the
  OTE leg plus a volatility (ATR) buffer. **TP1 = 1R** (scale out 50%, move SL to
  breakeven), **TP2 = the opposing structural extreme, at least 2R**.

## Setup (one time)

1. **Add your Telegram credentials as repo secrets** (Settings → Secrets and
   variables → Actions → New repository secret):
   - `TELEGRAM_TOKEN` — your bot token from @BotFather
   - `TELEGRAM_CHAT` — your chat ID
2. Go to the **Actions** tab and enable workflows. The schedule starts by itself.
3. To test immediately, open **Actions → OTE Signal Scan → Run workflow**.

## Run it locally

```bash
pip install -r requirements.txt
python signal_scanner.py \
  --symbols "EURUSD=X,GBPUSD=X,BTC-USD,GC=F" \
  --timeframes 5m,15m,1h \
  --telegram-token "YOUR_TOKEN" --telegram-chat "YOUR_CHAT"
```

Add `--watch` to loop continuously instead of a single scan.

## Honest disclaimer

This is a **research / paper-trading tool**. A corrected backtest (no lookahead
bias, structurally-placed stops) showed this strategy is **net negative before
costs** on most instruments. It is **not** a guaranteed money-maker and no
strategy can avoid losing trades. Do not trade real money on these signals
without independent validation.
