"""
TradingView-style chart renderer for OTE+SMC signals.
Draws real candlesticks + Entry / SL / TP levels and saves a PNG.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


BG = "#131722"       # TradingView dark background
GRID = "#1e222d"     # faint grid
AXIS = "#2a2e39"     # axis lines
TEXT = "#d1d4dc"     # light text
UP = "#26a69a"       # bullish candle
DOWN = "#ef5350"     # bearish candle
ENTRY_C = "#2962ff"  # entry line
TP_C = "#26a69a"     # take profit
SL_C = "#ef5350"     # stop loss


def render_chart(df: pd.DataFrame, signal: dict, out_path: str, bars: int = 90):
    """df: OHLC dataframe (open/high/low/close). signal: dict with entry/stop/target."""
    d = df.iloc[-bars:].copy()
    n = len(d)
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # --- Candles ---
    for i, (idx, row) in enumerate(d.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = UP if c >= o else DOWN
        # wick
        ax.plot([i, i], [l, h], color=color, linewidth=1.0, zorder=2)
        # body
        body_h = abs(c - o) or (h - l) * 0.001
        ax.add_patch(Rectangle((i - 0.35, min(o, c)), 0.7, body_h,
                               facecolor=color, edgecolor=color, zorder=3))

    entry, stop, target = signal["entry"], signal["stop"], signal["target"]
    is_long = signal["direction"] == "long"

    # --- Shaded risk / reward regions ---
    ax.axhspan(min(entry, target), max(entry, target), color=TP_C, alpha=0.08, zorder=1)
    ax.axhspan(min(entry, stop), max(entry, stop), color=SL_C, alpha=0.08, zorder=1)

    # --- Level lines ---
    ax.axhline(entry, color=ENTRY_C, lw=1.4, ls="--", zorder=4)
    ax.axhline(target, color=TP_C, lw=1.4, ls="-", zorder=4)
    ax.axhline(stop, color=SL_C, lw=1.4, ls="-", zorder=4)

    # --- Right-side labels ---
    x_lab = n - 0.5
    ax.text(x_lab, entry, f"  Entry {entry:.5g}", color=ENTRY_C,
            va="center", ha="left", fontsize=9, fontweight="bold")
    if signal.get("type") == "EXIT":
        ex = signal.get("exit", entry)
        state = signal.get("outcome", "").upper()
        ex_color = UP if state == "WIN" else DOWN
        ax.axhline(ex, color=ex_color, lw=1.6, ls=":", zorder=5)
        ax.text(x_lab, ex, f"  Exit {ex:.5g} ({state})", color=ex_color,
                va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax.text(x_lab, target, f"  TP {target:.5g}  ({signal.get('rr', 2)}R)",
                color=TP_C, va="center", ha="left", fontsize=9, fontweight="bold")
        ax.text(x_lab, stop, f"  SL {stop:.5g}", color=SL_C,
                va="center", ha="left", fontsize=9, fontweight="bold")

    # --- Entry marker ---
    ax.axvline(n - 1, color=ENTRY_C, lw=0.8, ls=":", alpha=0.6, zorder=1)
    marker = "^" if is_long else "v"
    mcolor = UP if is_long else DOWN
    ymark = min(stop, entry) if is_long else max(stop, entry)
    ax.scatter([n - 1], [ymark], marker=marker, s=160, color=mcolor, zorder=5)

    # --- Styling to mimic TradingView ---
    ax.tick_params(colors=TEXT, labelsize=8)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_xlim(-1, n + 6)

    direction = signal["direction"].upper()
    title = (f"{signal['symbol']}  •  {signal.get('timeframe','')}  •  {direction}"
             f"   |   OTE+SMC setup")
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", loc="left", pad=10)

    conf = "+".join(signal.get("confluence", []))
    sub = f"Entry {entry:.5g}   SL {stop:.5g}   TP {target:.5g}" + \
          (f"   |   {conf}" if conf else "")
    ax.text(0, 1.015, sub, transform=ax.transAxes, color="#787b86",
            fontsize=9, ha="left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path
