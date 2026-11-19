"""
OTE (Optimal Trade Entry) + SMC (Smart Money Concepts) Strategy Engine
======================================================================
Implements the OTE framework from the slides:
  - Fibonacci OTE zone (0.62 / 0.705 / 0.79 retracement)
  - Higher-timeframe bias (Daily / H4)
  - PD Arrays: FVG (Fair Value Gap), Order Blocks, Liquidity Sweeps (SSL/BSL)
  - Killzone filtering (London 02-05 NY, NY 07-10 NY)
  - Lower-timeframe confirmation entry

DISCLAIMER: This is research / backtesting code. It is not financial advice.
Past backtested performance does not guarantee future results. Trading
involves substantial risk of loss.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Swing:
    index: int
    price: float
    kind: str  # "high" or "low"


@dataclass
class FVG:
    index: int          # middle candle index of the 3-candle gap
    top: float
    bottom: float
    bullish: bool       # True = bullish FVG (gap up), False = bearish


@dataclass
class OrderBlock:
    index: int
    top: float
    bottom: float
    bullish: bool       # True = bullish OB (last down candle before up move)


@dataclass
class LiquiditySweep:
    index: int
    price: float
    kind: str           # "SSL" (sell-side, lows swept) or "BSL" (buy-side, highs swept)


@dataclass
class Trade:
    entry_index: int
    entry_price: float
    direction: str      # "long" or "short"
    stop: float
    target: float
    risk_reward: float
    pnl_pct: float = 0.0
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    outcome: str = "open"
    confluence: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fractal_swings(df: pd.DataFrame, lookback: int = 2) -> Tuple[List[Swing], List[Swing]]:
    """Detect swing highs and lows using a fractal window of `lookback` bars."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs: List[Swing] = []
    swing_lows: List[Swing] = []

    for i in range(lookback, n - lookback):
        if highs[i] == max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append(Swing(i, highs[i], "high"))
        if lows[i] == min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append(Swing(i, lows[i], "low"))
    return swing_highs, swing_lows


def find_fvgs(df: pd.DataFrame) -> List[FVG]:
    """Fair Value Gaps: 3-candle imbalance where middle candle leaves a gap."""
    fvgs: List[FVG] = []
    h = df["high"].values
    l = df["low"].values
    for i in range(1, len(df) - 1):
        # Bullish FVG: low[i+1] > high[i-1]
        if l[i + 1] > h[i - 1]:
            fvgs.append(FVG(i, l[i + 1], h[i - 1], bullish=True))
        # Bearish FVG: high[i+1] < low[i-1]
        elif h[i + 1] < l[i - 1]:
            fvgs.append(FVG(i, l[i - 1], h[i + 1], bullish=False))
    return fvgs


def find_order_blocks(df: pd.DataFrame, threshold: float = 1.0) -> List[OrderBlock]:
    """
    Order block: the last candle of opposite color before an aggressive move
    whose body exceeds `threshold` times the ATR.
    """
    obs: List[OrderBlock] = []
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    atr = _atr(df, 14)
    for i in range(1, len(df)):
        body = abs(c[i] - o[i])
        move = body
        if atr[i] <= 0:
            continue
        if move < threshold * atr[i]:
            continue
        # Bullish move: OB is the last bearish candle before it
        if c[i] > o[i]:
            prev_bearish = c[i - 1] < o[i - 1]
            if prev_bearish:
                obs.append(OrderBlock(i - 1, h[i - 1], l[i - 1], bullish=True))
        # Bearish move: OB is the last bullish candle before it
        elif c[i] < o[i]:
            prev_bullish = c[i - 1] > o[i - 1]
            if prev_bullish:
                obs.append(OrderBlock(i - 1, h[i - 1], l[i - 1], bullish=False))
    return obs


def find_liquidity_sweeps(df: pd.DataFrame, swing_highs: List[Swing],
                          swing_lows: List[Swing]) -> List[LiquiditySweep]:
    """
    Liquidity sweep: price takes out a prior swing high/low then closes back
    inside the range (wick through the level).
      BSL = buy-side liquidity swept (high taken)
      SSL = sell-side liquidity swept (low taken)
    """
    sweeps: List[LiquiditySweep] = []
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    high_levels = [sw.price for sw in swing_highs]
    low_levels = [sw.price for sw in swing_lows]

    for i in range(1, len(df)):
        # BSL: wick above a prior swing high, close below it
        for lv in high_levels:
            if h[i] > lv and c[i] < lv:
                sweeps.append(LiquiditySweep(i, lv, "BSL"))
                break
        # SSL: wick below a prior swing low, close above it
        for lv in low_levels:
            if l[i] < lv and c[i] > lv:
                sweeps.append(LiquiditySweep(i, lv, "SSL"))
                break
    return sweeps


def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(period).mean().values
    atr = np.nan_to_num(atr, nan=np.mean(tr[1:]) if len(tr) > 1 else 1.0)
    return atr


# --------------------------------------------------------------------------- #
# Bias and killzones
# --------------------------------------------------------------------------- #
def htf_bias(df_htf: pd.DataFrame) -> str:
    """
    Higher-timeframe bias using structure break + EMA.
    Returns "bullish", "bearish", or "neutral".
    """
    if len(df_htf) < 50:
        return "neutral"
    ema20 = df_htf["close"].ewm(span=20).mean().values
    ema50 = df_htf["close"].ewm(span=50).mean().values
    # Structure: higher highs / higher lows = bullish
    swing_highs, swing_lows = _fractal_swings(df_htf, lookback=3)
    last_high = swing_highs[-1].price if swing_highs else df_htf["high"].iloc[-1]
    last_low = swing_lows[-1].price if swing_lows else df_htf["low"].iloc[-1]
    prev_high = swing_highs[-2].price if len(swing_highs) > 1 else last_high
    prev_low = swing_lows[-2].price if len(swing_lows) > 1 else last_low

    bullish_structure = last_high > prev_high and last_low > prev_low
    bearish_structure = last_high < prev_high and last_low < prev_low
    ema_bullish = ema20[-1] > ema50[-1] and df_htf["close"].iloc[-1] > ema20[-1]
    ema_bearish = ema20[-1] < ema50[-1] and df_htf["close"].iloc[-1] < ema20[-1]

    if (bullish_structure or ema_bullish) and not bearish_structure:
        return "bullish"
    if (bearish_structure or ema_bearish) and not bullish_structure:
        return "bearish"
    return "neutral"


def htf_bias_series(df_htf: pd.DataFrame, lookback: int = 3,
                    shift: int = 1) -> pd.Series:
    """
    Per-bar HTF bias with NO lookahead. A fractal swing is only 'known'
    `lookback` bars after it forms, and the series is shifted so a still-forming
    HTF candle is never used. Returns a Series indexed like df_htf.
    """
    n = len(df_htf)
    if n < 50:
        return pd.Series(["neutral"] * n, index=df_htf.index)
    import bisect
    close = df_htf["close"].values
    ema20 = pd.Series(close).ewm(span=20).mean().values
    ema50 = pd.Series(close).ewm(span=50).mean().values
    sh, sl = _fractal_swings(df_htf, lookback)
    hkeys = [s.index + lookback for s in sh]   # first bar where swing is known
    lkeys = [s.index + lookback for s in sl]
    out = []
    for i in range(n):
        nh = bisect.bisect_right(hkeys, i)
        nl = bisect.bisect_right(lkeys, i)
        ema_bull = ema20[i] > ema50[i] and close[i] > ema20[i]
        ema_bear = ema20[i] < ema50[i] and close[i] < ema20[i]
        sbull = sbear = False
        if nh >= 2 and nl >= 2:
            lh, ph = sh[nh - 1].price, sh[nh - 2].price
            ll, pl = sl[nl - 1].price, sl[nl - 2].price
            sbull = lh > ph and ll > pl
            sbear = lh < ph and ll < pl
        b = "neutral"
        if (sbull or ema_bull) and not sbear:
            b = "bullish"
        elif (sbear or ema_bear) and not sbull:
            b = "bearish"
        out.append(b)
    s = pd.Series(out, index=df_htf.index)
    if shift:
        s = s.shift(shift).fillna("neutral")
    return s


def in_killzone(dt) -> bool:
    """
    Killzones in New York time:
      London Killzone: 02:00 - 05:00 NY (07:00 - 10:00 UTC)
      NY Killzone:     07:00 - 10:00 NY (12:00 - 15:00 UTC)
    dt must be timezone-aware or we assume UTC.
    """
    try:
        utc = dt.tz_convert("UTC") if dt.tzinfo is not None else dt
        hour = utc.hour
        return (7 <= hour < 10) or (12 <= hour < 15)
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# OTE logic
# --------------------------------------------------------------------------- #
def ote_zones(swing_low_price: float, swing_high_price: float,
              direction: str) -> Tuple[float, float, float]:
    """
    Return (fib_062, fib_0705, fib_079) for the given swing.
    For a long (bullish OTE), fib is drawn low->high, retracement measured
    from high down:  price = high - level * (high - low)
    For a short, drawn high->low, retracement measured from low up.
    """
    rng = swing_high_price - swing_low_price
    if direction == "long":
        return (swing_high_price - 0.62 * rng,
                swing_high_price - 0.705 * rng,
                swing_high_price - 0.79 * rng)
    else:
        return (swing_low_price + 0.62 * rng,
                swing_low_price + 0.705 * rng,
                swing_low_price + 0.79 * rng)


def price_in_ote(price: float, zones: Tuple[float, float, float],
                 direction: str) -> bool:
    """OTE zone spans 0.62 to 0.79."""
    lo, mid, hi = zones
    zone_low = min(lo, hi)
    zone_high = max(lo, hi)
    return zone_low <= price <= zone_high


# --------------------------------------------------------------------------- #
# Main strategy runner
# --------------------------------------------------------------------------- #
def run_strategy(df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                 rr_ratio: float = 2.0,
                 atr_mult_stop: float = 1.5,
                 min_confluence: int = 2) -> List[Trade]:
    """
    Run the OTE+SMC strategy.

    df_ltf : lower-timeframe data (e.g. M5 / H1) where entries are taken
    df_htf : higher-timeframe data (e.g. H4 / D1) for bias
    rr_ratio: reward-to-risk target multiple
    atr_mult_stop: stop loss distance in ATR units
    min_confluence: number of PD arrays required inside OTE to take the trade
    """
    trades: List[Trade] = []
    if len(df_ltf) < 60 or len(df_htf) < 60:
        return trades

    atr = _atr(df_ltf, 14)
    swing_highs, swing_lows = _fractal_swings(df_ltf, lookback=3)
    fvgs = find_fvgs(df_ltf)
    obs = find_order_blocks(df_ltf, threshold=1.0)
    sweeps = find_liquidity_sweeps(df_ltf, swing_highs, swing_lows)

    h = df_ltf["high"].values
    l = df_ltf["low"].values
    c = df_ltf["close"].values
    o = df_ltf["open"].values
    dates = df_ltf.index

    # Per-bar HTF bias aligned to LTF bars (no lookahead). Align timezones
    # first: Yahoo returns daily bars tz-naive and intraday bars tz-aware.
    bs = htf_bias_series(df_htf)
    b_idx, l_idx = bs.index, df_ltf.index
    if (b_idx.tz is None) != (l_idx.tz is None):
        if b_idx.tz is not None:
            b_idx = b_idx.tz_convert("UTC").tz_localize(None)
        if l_idx.tz is not None:
            l_idx = l_idx.tz_convert("UTC").tz_localize(None)
    bias_ltf = pd.Series(bs.values, index=b_idx).reindex(
        l_idx, method="ffill").fillna("neutral")

    # Build quick lookup maps for confluence at each bar
    fvg_by_idx = {f.index: f for f in fvgs}
    ob_by_idx = {ob.index: ob for ob in obs}
    sweep_by_idx = {s.index: s for s in sweeps}

    in_trade: Optional[Trade] = None

    for i in range(60, len(df_ltf)):
        # Manage open trade
        if in_trade is not None:
            if in_trade.direction == "long":
                if l[i] <= in_trade.stop:
                    in_trade.outcome = "loss"
                    in_trade.exit_price = in_trade.stop
                elif h[i] >= in_trade.target:
                    in_trade.outcome = "win"
                    in_trade.exit_price = in_trade.target
            else:
                if h[i] >= in_trade.stop:
                    in_trade.outcome = "loss"
                    in_trade.exit_price = in_trade.stop
                elif l[i] <= in_trade.target:
                    in_trade.outcome = "win"
                    in_trade.exit_price = in_trade.target
            if in_trade.outcome != "open":
                in_trade.exit_index = i
                in_trade.pnl_pct = ((in_trade.exit_price - in_trade.entry_price) /
                                    in_trade.entry_price) * 100
                if in_trade.direction == "short":
                    in_trade.pnl_pct *= -1
                trades.append(in_trade)
                in_trade = None
            continue

        # Only look for entries inside killzones
        if not in_killzone(dates[i]):
            continue

        # Determine direction from HTF bias (per-bar, no lookahead)
        b = bias_ltf.iloc[i]
        direction = None
        if b == "bullish":
            direction = "long"
        elif b == "bearish":
            direction = "short"
        else:
            continue

        # Find the most recent completed swing to draw the fib
        if direction == "long":
            # need a recent swing low -> swing high leg
            recent_lows = [s for s in swing_lows if s.index < i]
            recent_highs = [s for s in swing_highs if s.index < i]
            if not recent_lows or not recent_highs:
                continue
            sw_low = recent_lows[-1]
            sw_high = recent_highs[-1]
            if sw_low.index >= sw_high.index or sw_high.price <= sw_low.price:
                continue
            zones = ote_zones(sw_low.price, sw_high.price, "long")
        else:
            recent_highs = [s for s in swing_highs if s.index < i]
            recent_lows = [s for s in swing_lows if s.index < i]
            if not recent_highs or not recent_lows:
                continue
            sw_high = recent_highs[-1]
            sw_low = recent_lows[-1]
            if sw_high.index >= sw_low.index or sw_high.price <= sw_low.price:
                continue
            zones = ote_zones(sw_high.price, sw_low.price, "short")

        # Current price must be inside OTE zone
        cur_price = c[i]
        if not price_in_ote(cur_price, zones, direction):
            continue

        # Count confluence (PD arrays inside the OTE zone + liquidity sweep)
        confluence: List[str] = []
        # FVG in zone
        for f in fvgs:
            if f.index < i and (f.bullish == (direction == "long")):
                fvg_mid = (f.top + f.bottom) / 2
                if price_in_ote(fvg_mid, zones, direction):
                    confluence.append("FVG")
                    break
        # Order block in zone
        for ob in obs:
            if ob.index < i and (ob.bullish == (direction == "long")):
                ob_mid = (ob.top + ob.bottom) / 2
                if price_in_ote(ob_mid, zones, direction):
                    confluence.append("OB")
                    break
        # Liquidity sweep
        for s in sweeps:
            if s.index < i:
                if (direction == "long" and s.kind == "SSL") or \
                   (direction == "short" and s.kind == "BSL"):
                    confluence.append(f"Sweep({s.kind})")
                    break

        if len(confluence) < min_confluence:
            continue

        # Entry + structurally-correct stop + target
        #   SL sits just beyond the swing that defined the OTE leg (invalidation
        #   level), capped at 3*ATR so a wide swing can't create an oversized risk.
        entry = cur_price
        a = atr[i]
        if a <= 0:
            continue
        if direction == "long":
            raw_stop = sw_low.price - 0.1 * a        # below the leg origin
            risk = min(entry - raw_stop, 3 * a)
            if risk <= 0:
                continue
            stop = entry - risk
            target = entry + risk * rr_ratio
        else:
            raw_stop = sw_high.price + 0.1 * a        # above the leg origin
            risk = min(raw_stop - entry, 3 * a)
            if risk <= 0:
                continue
            stop = entry + risk
            target = entry - risk * rr_ratio

        in_trade = Trade(
            entry_index=i,
            entry_price=entry,
            direction=direction,
            stop=stop,
            target=target,
            risk_reward=rr_ratio,
            confluence=confluence,
        )

    # Close any still-open trade at last price
    if in_trade is not None:
        last = len(df_ltf) - 1
        in_trade.exit_index = last
        in_trade.exit_price = c[last]
        in_trade.outcome = "open"
        in_trade.pnl_pct = ((c[last] - in_trade.entry_price) /
                            in_trade.entry_price) * 100
        if in_trade.direction == "short":
            in_trade.pnl_pct *= -1
        trades.append(in_trade)

    return trades
