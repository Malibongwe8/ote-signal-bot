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
    tp1 = signal.get("tp1")
    is_long = signal["direction"] == "long"

    # --- Shaded risk / reward regions ---
    ax.axhspan(min(entry, target), max(entry, target), color=TP_C, alpha=0.08, zorder=1)
    ax.axhspan(min(entry, stop), max(entry, stop), color=SL_C, alpha=0.08, zorder=1)

    # --- Level lines (Entry, TP1, TP2, SL) ---
    ax.axhline(entry, color=ENTRY_C, lw=1.4, ls="--", zorder=4)
    ax.axhline(target, color=TP_C, lw=1.4, ls="-", zorder=4)              # TP2
    if tp1 is not None:
        ax.axhline(tp1, color=TP_C, lw=1.0, ls="-.", alpha=0.8, zorder=4)  # TP1
    ax.axhline(stop, color=SL_C, lw=1.4, ls="-", zorder=4)

    # --- Right-side labels ---
    x_lab = n - 0.5
    ax.text(x_lab, entry, f"  Entry {entry:.5g}", color=ENTRY_C,
            va="center", ha="left", fontsize=9, fontweight="bold")
    if signal.get("type") in ("EXIT", "TP1"):
        ex = signal.get("exit", entry)
        state = (signal.get("outcome") or ("TP1" if signal.get("type") == "TP1" else "")).upper()
        ex_color = UP if state in ("WIN", "TP1") else DOWN
        ax.axhline(ex, color=ex_color, lw=1.6, ls=":", zorder=5)
        ax.text(x_lab, ex, f"  Exit {ex:.5g} ({state})", color=ex_color,
                va="center", ha="left", fontsize=9, fontweight="bold")
    else:
        ax.text(x_lab, target, f"  TP2 {target:.5g}  ({signal.get('rr', 2)}R)",
                color=TP_C, va="center", ha="left", fontsize=9, fontweight="bold")
        if tp1 is not None:
            ax.text(x_lab, tp1, f"  TP1 {tp1:.5g}", color=TP_C, alpha=0.85,
                    va="center", ha="left", fontsize=8)
        ax.text(x_lab, stop, f"  SL {stop:.5g}", color=SL_C,
                va="center", ha="left", fontsize=9, fontweight="bold")

    # --- Swing high/low the OTE is acting on + the OTE zone ---
    SW = "#f5a623"          # swing marker colour
    sh_p = signal.get("swing_high")
    sl_p = signal.get("swing_low")
    zt = signal.get("ote_top")
    zb = signal.get("ote_bottom")
    if zt is not None and zb is not None:
        ax.axhspan(min(zt, zb), max(zt, zb), color="#2962ff", alpha=0.12, zorder=1)
        ax.text(1, (zt + zb) / 2, " OTE 0.62-0.79", color="#9aa0aa",
                fontsize=8, va="center", ha="left", zorder=6)
    if sh_p is not None:
        ax.axhline(sh_p, color=SW, lw=1.0, ls=":", alpha=0.9, zorder=3)
        ax.text(x_lab, sh_p, f"  Swing High {sh_p:.5g}", color=SW,
                va="center", ha="left", fontsize=8)
    if sl_p is not None:
        ax.axhline(sl_p, color=SW, lw=1.0, ls=":", alpha=0.9, zorder=3)
        ax.text(x_lab, sl_p, f"  Swing Low {sl_p:.5g}", color=SW,
                va="center", ha="left", fontsize=8)
    # Markers placed exactly on the swing bars (when in view)
    off_hi = signal.get("swing_high_offset")
    off_lo = signal.get("swing_low_offset")
    if sh_p is not None and isinstance(off_hi, int) and 0 <= off_hi <= n - 1:
        ax.scatter([(n - 1) - off_hi], [sh_p], marker="o", s=55,
                   facecolors="none", edgecolors=SW, linewidths=1.4, zorder=7)
    if sl_p is not None and isinstance(off_lo, int) and 0 <= off_lo <= n - 1:
        ax.scatter([(n - 1) - off_lo], [sl_p], marker="o", s=55,
                   facecolors="none", edgecolors=SW, linewidths=1.4, zorder=7)

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
    tp1s = f"   TP1 {tp1:.5g}" if tp1 is not None else ""
    sub = f"Entry {entry:.5g}   SL {stop:.5g}{tp1s}   TP2 {target:.5g}" + \
          (f"   |   {conf}" if conf else "")
    ax.text(0, 1.015, sub, transform=ax.transAxes, color="#787b86",
            fontsize=9, ha="left")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_path
