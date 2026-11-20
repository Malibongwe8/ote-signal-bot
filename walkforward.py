#!/usr/bin/env python3
"""Honest walk-forward pattern study (research only).

Learns a small logistic-regression model on candlestick / ICT / context features
from PAST trades, then tests it on genuinely UNSEEN future data using an
expanding-window walk-forward. It reports REAL out-of-sample numbers and does
NOT tune itself to look good. If there is no edge, it will say so.

This is deliberately NOT wired into the live scanner.

Usage:
  python walkforward.py --symbol GBPUSD=X \
      --setups "5m:1d:60d,15m:1d:60d,30m:1d:60d,1h:1d:730d"
"""
import argparse
import sys

import numpy as np
import pandas as pd

from ote_strategy import (run_strategy, _atr, _fractal_swings,
                          find_liquidity_sweeps, candle_confirms, ict_confirms,
                          in_killzone)
from backtest import fetch_data


def _features(df, i, direction, atr, ema20):
    """Features known at the close of bar i (no lookahead)."""
    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    a = atr[i] if atr[i] > 0 else 1e-9
    base = float(np.mean(atr[max(0, i - 100):i])) if i > 0 else a
    base = base if base > 0 else a

    sh, sl = _fractal_swings(df.iloc[:i + 1], lookback=2)
    sweeps = find_liquidity_sweeps(df.iloc[:i + 1], sh, sl)
    cand = candle_confirms(df, i, direction)
    ict = ict_confirms(df, i, direction, sweeps, atr)
    slope = (ema20[i] - ema20[max(0, i - 5)]) / a if i >= 5 else 0.0
    return np.array([
        1.0 if direction == "long" else 0.0,   # side
        a / base,                              # volatility regime
        (c[i] - o[i]) / a,                     # entry candle body
        (h[i] - l[i]) / a,                     # entry candle range
        (c[i] - l[i]) / a,                     # close location in range
        1.0 if cand == "Engulfing" else 0.0,
        1.0 if cand == "Pin" else 0.0,
        1.0 if ict == "Displacement" else 0.0,
        1.0 if ict == "Sweep" else 0.0,
        1.0 if in_killzone(df.index[i]) else 0.0,
        slope,
    ], dtype=float)


def _dataset(df_ltf, df_htf, rr):
    trades = [t for t in run_strategy(df_ltf, df_htf, rr_ratio=rr,
                                      min_confluence=2)
              if t.exit_index is not None and t.outcome != "open"]
    if not trades:
        return None
    atr = _atr(df_ltf, 14)
    ema20 = pd.Series(df_ltf["close"].values).ewm(span=20).mean().values
    X, y, R, tix = [], [], [], []
    for t in trades:
        i = t.entry_index
        if i < 5 or i >= len(df_ltf):
            continue
        X.append(_features(df_ltf, i, t.direction, atr, ema20))
        y.append(1 if t.r_multiple > 0 else 0)
        R.append(t.r_multiple)
        tix.append(i)
    if len(y) < 20:
        return None
    return np.array(X), np.array(y), np.array(R), np.array(tix)


def fit_logreg(X, y, l2=1.0, lr=0.15, epochs=800):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - y
        w -= lr * (Xs.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return mu, sd, w, b


def predict(model, X):
    mu, sd, w, b = model
    return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) +
                 0.5 * np.mean(pos[:, None] == neg[None, :]))


def walk_forward(X, y, R, folds=5):
    n = len(y)
    if n < 40:
        return None
    po = np.full(n, np.nan)
    for f in range(1, folds):
        tr_end = int(n * f / folds)
        te_end = int(n * (f + 1) / folds)
        if tr_end < 20 or te_end <= tr_end:
            continue
        tr, te = slice(0, tr_end), slice(tr_end, te_end)
        if len(np.unique(y[tr])) < 2:
            continue
        model = fit_logreg(X[tr], y[tr])
        po[te] = predict(model, X[te])
    mask = ~np.isnan(po)
    return po, mask


def report(name, X, y, R):
    res = walk_forward(X, y, R)
    if res is None:
        print(f"{name:14} : too few trades to walk-forward (n={len(y)})")
        return None
    po, mask = res
    yo, Ro, pno = y[mask], R[mask], po[mask]
    if len(yo) < 10:
        print(f"{name:14} : too few OOS trades (n={len(yo)})")
        return None
    base = yo.mean()
    acc = float(np.mean((pno >= 0.5) == yo))
    a = auc(yo, pno)
    all_r = Ro.mean()
    print(f"{name:14} : OOS n={len(yo):3d} | base win {base*100:4.1f}% | "
          f"acc@0.5 {acc*100:4.1f}% | AUC {a:.3f} | take-all {all_r:+.3f}R")
    best = None
    for th in (0.45, 0.5, 0.55, 0.6, 0.65):
        sel = pno >= th
        if sel.sum() >= 5:
            print(f"{'':14}   p>={th:.2f}: take {int(sel.sum()):3d} | "
                  f"win {yo[sel].mean()*100:4.1f}% | {Ro[sel].mean():+.3f}R")
            if best is None or Ro[sel].mean() > best[1]:
                best = (th, Ro[sel].mean(), int(sel.sum()))
    return base, a, all_r, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GBPUSD=X")
    ap.add_argument("--setups", default="5m:1d:60d,15m:1d:60d,30m:1d:60d,1h:1d:730d")
    ap.add_argument("--rr", type=float, default=2.0)
    args = ap.parse_args()

    print(f"Walk-forward study — {args.symbol} (expanding window, 5 folds)")
    print("Out-of-sample only. No tuning. 'take-all' = every setup taken.\n")
    rows = []
    for spec in args.setups.split(","):
        ltf, htf, period = spec.split(":")
        try:
            df_ltf, df_htf = fetch_data(args.symbol, ltf, htf, period)
        except Exception as e:
            print(f"{ltf+'/'+htf:14} : fetch failed ({e})")
            continue
        d = _dataset(df_ltf, df_htf, args.rr)
        if d is None:
            print(f"{ltf+'/'+htf:14} : not enough setups in {period}")
            continue
        X, y, R, _ = d
        r = report(f"{ltf}/{htf}", X, y, R)
        if r:
            rows.append((f"{ltf}/{htf}", len(y), r))
    print("\n--- summary ---")
    for name, n, (base, a, all_r, best) in rows:
        print(f"{name:10} setups={n:3d} AUC={a:.3f} take-all={all_r:+.3f}R "
              f"best-filter={best[2] if best else 0} trades @ "
              f"{best[1]:+.3f}R" if best else
              f"{name:10} setups={n:3d} AUC={a:.3f} take-all={all_r:+.3f}R")
    print("\nHonest reading: an AUC near 0.50 means the model cannot tell winners "
          "from losers. A high win% on a tiny sample is noise.")


if __name__ == "__main__":
    main()
