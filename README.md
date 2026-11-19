# OTE + SMC Signal Bot

Scans forex pairs, BTC and gold for OTE (Optimal Trade Entry) + SMC setups and
sends them to Telegram — as a message **and** a TradingView-style chart image.

Runs on **GitHub Actions**, scheduled every 5 minutes, so it keeps running
whether or not your own computer is on.

## What it scans

- **Symbols (50):** 26 FX pairs (majors + crosses), `BTC-USD`, `GC=F` (gold),
  9 volatility indices (`^VIX ^VXN ^VXD ^VVIX ^VIX3M ^VXAPL ^OVX ^VXEEM ^MOVE`),
  13 equity/regional indices (`^GSPC ^NDX ^DJI ^RUT ^GDAXI ^FTSE ^N225 ^HSI ^AXJO ^FCHI ^STOXX50E ^IBEX ^KS11`)
- **Timeframes:** 5m, 15m, 1h
- **Directions:** both long and short
- **Strategy:** fib OTE zone (0.62–0.79) + HTF bias + multi-timeframe structure
  agreement + confluence (FVG / Order Block / Liquidity Sweep), entries inside
  London & NY killzones. **Dynamic stop** sits beyond the swing that defined the
  OTE leg plus a volatility (ATR) buffer. **TP1 = 1R** (scale out 50%, move SL to
  breakeven), **TP2 = the opposing structural extreme, at least 2R**.

## Telegram commands

Message the bot (in the configured chat) any of these — it replies within ~5 minutes
(it polls for commands at the start of every scan):

| Command | What it does |
|---|---|
| `/help` `/start` | List the commands |
| `/status` | Scanner status, symbol/TF count, open positions, last scan time |
| `/positions` | List currently open trades (entry / SL / TP2) |
| `/last` | Show the last 5 signals |
| `/symbols` | List the instruments being scanned |
| `/test` | Send a sample signal with a chart |
| `/mute` | Pause signal alerts (scanning continues silently) |
| `/unmute` | Resume signal alerts |

## Staying alive / self-healing

- **`scan.yml`** runs the scan every 5 minutes on GitHub's servers.
- **`watchdog.yml`** is the self-healing supervisor. Every 15 minutes it checks
  whether `scan.yml` has actually run. If it has **stalled** (no run for 25 min)
  or GitHub **disabled** it (e.g. 60-day inactivity), the watchdog **re-enables
  and re-triggers** the scan and pings you on Telegram. The scanner process also
  catches its own errors and keeps looping rather than dying.
- **Honest limit:** GitHub cron is best-effort. Runs can be delayed, and the
  watchdog runs on the same scheduler, so this is a strong safety net — **not an
  absolute uptime guarantee.** For true 24/7 you'd still want a VPS.

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
