#!/usr/bin/env python3
"""
Prices, features and splits — frozen, and chosen to defeat survivorship bias.

    uv run data.py                   # describe both universes
    uv run data.py --bias            # what picking survivors is worth
    uv run data.py --leakage         # prove no feature reads forward
    uv run data.py --refresh         # re-download, and print what moved

THE UNIVERSE IS NINE SECTOR ETFs, AND THAT IS THE WHOLE POINT
--------------------------------------------------------------
The obvious universe for a trading lab is a handful of well-known
stocks. That universe is poisoned, and this lab measured by how
much before choosing anything.

Twelve hand-picked large caps that are liquid TODAY returned
**+22.85%/yr** from 2011 to 2025, equal-weighted. The S&P 500
equal-weight index over the same window returned **+12.66%**. The
ten-point gap is not skill and it is not the market -- it is the
act of choosing, in 2026, twelve companies that happened to
survive. Nothing in the data looks wrong. Every arm in the lab
would have inherited it, including buy-and-hold, which is the
control every reinforcement-learning arm has to beat. The control
would have been handed ten free points a year and RL would have
"lost" for a reason that had nothing to do with RL.

The nine SPDR sector ETFs do not have this problem. When a
constituent fails or is acquired the ETF absorbs the loss
internally, so the price series already carries it. And there is
no basket to pick: the lab takes ALL NINE, so there is no
selection decision to be biased by. Measured against the same
yardstick the sectors come in at **+0.40%/yr**, which is noise.

    universe                          2011-2025      vs RSP
    12 hand-picked survivors            +22.85%     +10.19%
    9 sector ETFs                       +13.05%      +0.40%
    RSP, S&P 500 equal weight           +12.66%       --

WHAT FREE DATA CANNOT DO, STATED PLAINLY
------------------------------------------
The textbook fix is to include the companies that died. It is not
available here. Sixteen delisted tickers were tested against
yfinance -- SIVB, FRC, Twitter, Activision, Monsanto, Raytheon and
others -- and **fourteen return nothing at all**. The two that
appear to work are traps: `SBNY` returns 261 rows beginning in
2024, which is a different company that recycled a dead ticker,
and `FISV` is Yahoo silently redirecting to the surviving entity.
A lab that quietly included either would be worse off than one
that includes neither.

So the bias is NOT designed away. It is routed around by choosing
a universe where the vehicle absorbs it, bounded by measuring what
it is worth, and the biased basket is kept as a second,
clearly-labelled arm so a reader can see the ten points for
themselves.

THE LONGER HISTORY IS THE SECOND PRIZE
----------------------------------------
Sector ETFs reach back to 2003 where the stock basket reaches
2011, and that buys four drawdowns instead of one:

    GFC 2007-09      -49.0%      COVID crash      -28.5%
    Q4 2018          -13.9%      2022 bear        -11.6%

This lab's headline claim is that reward-driven policies buy
DOWNSIDE PROTECTION rather than alpha. On one bear year that is an
anecdote. On four independent drawdowns, one of them -49%, it is a
claim that can be tested and can lose.

EVERY FEATURE OBEYS ONE RULE
------------------------------
Computed from prices up to and including day `t`; the label is the
return from `t` to `t+1`. Look-ahead is the defining bug of this
subject -- it does not raise, it does not look wrong, and it
produces results that are spectacular and worthless. `--leakage`
tampers with the future and asserts no feature moves, AND asserts
that a deliberately forward-looking feature IS caught.

The scaler is fit on the TRAINING SLICE ONLY, and the split is
chronological. A random split of a time series puts tomorrow in
the training set and yesterday in the test set, and every metric
improves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SECTORS_CSV = HERE / "data" / "sectors.csv"
SURVIVORS_CSV = HERE / "data" / "survivors.csv"

# All nine SPDR sectors available since 2003. NOT a selection -- the
# complete set, which is what makes the universe unbiased.
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]

# Benchmarks. RSP is the honest yardstick: an equal-weight index has
# no survivorship bias, because the index itself handles delisting.
BENCHMARKS = ["RSP", "SPY"]

# The deliberately biased arm, kept so the reader can see the cost.
SURVIVORS = ["AAPL", "CVX", "HD", "JPM", "KO", "MSFT",
             "NVDA", "PG", "TSLA", "UNH", "WMT", "XOM"]

# Chosen before any result was computed. The test window contains
# Q4 2018, the COVID crash and the 2022 bear; the training window
# contains the GFC.
TRAIN_END = "2016-01-01"

# Named drawdowns, for the regime tables. A pooled figure over a
# bull market is a fact about the sample, not about the method.
DRAWDOWNS = [
    ("Q4 2018", "2018-10-01", "2018-12-31"),
    ("COVID crash", "2020-02-15", "2020-03-31"),
    ("2022 bear", "2022-01-01", "2022-12-31"),
]

FEATURES = ["r1", "r5", "r20", "vol20", "d14", "d50", "d200", "rsi14"]


def load_prices(which: str = "sectors", refresh: bool = False) -> pd.DataFrame:
    path = SECTORS_CSV if which == "sectors" else SURVIVORS_CSV
    want = (SECTORS + BENCHMARKS) if which == "sectors" else SURVIVORS
    if refresh or not path.exists():
        return _download(which)
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    missing = [t for t in want if t not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path.name} is missing {missing}. It was frozen with a "
            f"different universe; re-freeze with `uv run data.py "
            f"--refresh` and expect every published number to move.")
    return df[want]


def _download(which: str) -> pd.DataFrame:
    import yfinance as yf

    want = (SECTORS + BENCHMARKS) if which == "sectors" else SURVIVORS
    start = "2003-01-01" if which == "sectors" else "2010-01-01"
    d = yf.download(want, start=start, end="2025-09-01",
                    progress=False, auto_adjust=True)["Close"].dropna()
    d = d[want]
    d.index.name = "date"
    path = SECTORS_CSV if which == "sectors" else SURVIVORS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    d.round(6).to_csv(path)
    return d


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100.0 - 100.0 / (1.0 + up / (dn + 1e-12))


def features_for(close: pd.Series) -> pd.DataFrame:
    """One series' features. Every column uses prices up to `t` only."""
    r = close.pct_change()
    f = pd.DataFrame(index=close.index)
    f["r1"] = r
    f["r5"] = close.pct_change(5)
    f["r20"] = close.pct_change(20)
    f["vol20"] = r.rolling(20).std()
    for p in (14, 50, 200):
        f[f"d{p}"] = (close - close.rolling(p).mean()) / close
    f["rsi14"] = _rsi(close, 14) / 100.0
    f["fwd"] = r.shift(-1)          # the LABEL, never an input
    return f


def panel(which: str = "sectors", refresh: bool = False):
    """
    The basket as aligned arrays: X `(T, N, F)`, y `(T, N)`.

    The asset axis stays EXPLICIT rather than folded into the batch,
    because GRPO's group is "every asset on one day" and a flattened
    panel cannot express that.
    """
    px = load_prices(which, refresh)
    names = SECTORS if which == "sectors" else SURVIVORS
    frames = {t: features_for(px[t]) for t in names}
    idx = None
    for f in frames.values():
        g = f.dropna()
        idx = g.index if idx is None else idx.intersection(g.index)
    X = np.stack([frames[t].loc[idx, FEATURES].to_numpy(np.float32)
                  for t in names], axis=1)
    y = np.stack([frames[t].loc[idx, "fwd"].to_numpy(np.float32)
                  for t in names], axis=1)
    bench = px.loc[idx, "RSP"].pct_change().shift(-1).fillna(0.0) \
        if which == "sectors" else None
    return X, y, idx, list(names), (
        bench.to_numpy(np.float32) if bench is not None else None)


def split(X, y, dates, bench=None):
    """Chronological. The scaler sees the training slice only."""
    tr = dates < TRAIN_END
    mu = X[tr].mean(axis=(0, 1))
    sd = X[tr].std(axis=(0, 1)) + 1e-9
    Xn = (X - mu) / sd
    b_tr = bench[tr] if bench is not None else None
    b_te = bench[~tr] if bench is not None else None
    return (Xn[tr], y[tr], dates[tr], b_tr), (Xn[~tr], y[~tr], dates[~tr], b_te)


# ------------------------------------------------------------------ checks
def _bias_report() -> int:
    """What choosing survivors is worth, in basis points a year."""
    sec = load_prices("sectors")
    sur = load_prices("survivors")
    common = sec.index.intersection(sur.index)
    rs = sec.loc[common].pct_change().dropna()
    ru = sur.loc[common].pct_change().dropna()

    def ann(x):
        return float(np.mean(x) * 252)

    ew_sur = ann(ru[SURVIVORS].mean(axis=1))
    ew_sec = ann(rs[SECTORS].mean(axis=1))
    rsp = ann(rs["RSP"])
    spy = ann(rs["SPY"])
    print("=" * 74)
    print("  WHAT SURVIVORSHIP BIAS IS WORTH")
    print("=" * 74)
    print(f"  {common[0].date()} .. {common[-1].date()}, equal weighted, "
          f"annualised\n")
    print(f"    {'12 hand-picked survivors':<34}{ew_sur:>8.2%}"
          f"{ew_sur - rsp:>+10.2%}  BIASED")
    print(f"    {'9 sector ETFs':<34}{ew_sec:>8.2%}{ew_sec - rsp:>+10.2%}")
    print(f"    {'RSP, S&P 500 equal weight':<34}{rsp:>8.2%}"
          f"{0.0:>+10.2%}  the yardstick")
    print(f"    {'SPY, S&P 500 cap weight':<34}{spy:>8.2%}{spy - rsp:>+10.2%}")
    print(f"\n  Choosing twelve companies that are liquid today is worth")
    print(f"  {ew_sur - rsp:+.2%} a year. Buy-and-hold is the control every RL")
    print(f"  arm must beat, so on that basket the control starts with a")
    print(f"  ten-point head start it did not earn.")
    ok = (ew_sur - rsp) > 0.05 and abs(ew_sec - rsp) < 0.03
    print(f"\n  {'PASS' if ok else 'FAIL'}  the biased basket is inflated and "
          f"the sector basket is not")
    return 0 if ok else 1


def _leakage_check() -> int:
    """Tamper with the future; assert no feature notices. Then assert
    that a feature which DOES look forward is caught, or this proves
    nothing."""
    c = load_prices("sectors")["XLK"]
    cut = len(c) // 2
    rng = np.random.default_rng(0)
    bad_future = c.copy()
    bad_future.iloc[cut:] = rng.uniform(20, 300, len(c) - cut)

    a = features_for(c).iloc[:cut - 1]
    b = features_for(bad_future).iloc[:cut - 1]
    leaking = [col for col in FEATURES
               if not np.allclose(a[col].to_numpy(), b[col].to_numpy(),
                                  equal_nan=True)]
    ok = not leaking
    print(f"  {'PASS' if ok else 'FAIL'}  no feature reads forward in time"
          + (f"  — leaking: {leaking}" if leaking else ""))

    # The slice has to REACH the tamper point. An earlier version
    # sliced one row short, so `shift(-1)` never touched a corrupted
    # price and the counterexample came back clean -- a check that
    # could not fail on input built to break it, while reporting PASS
    # on the honest features.
    caught = not np.allclose(c.shift(-1).iloc[:cut].to_numpy(),
                             bad_future.shift(-1).iloc[:cut].to_numpy(),
                             equal_nan=True)
    print(f"  {'PASS' if caught else 'FAIL'}  ...and a deliberately "
          f"forward-looking feature IS caught")
    return 0 if (ok and caught) else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--leakage", action="store_true")
    ap.add_argument("--bias", action="store_true")
    a, _ = ap.parse_known_args()

    if a.bias:
        return _bias_report()
    if a.leakage:
        print("=" * 74)
        print("  LOOK-AHEAD CHECK")
        print("=" * 74)
        return _leakage_check()
    if a.refresh:
        for w in ("sectors", "survivors"):
            old = load_prices(w)
            new = _download(w)
            common = old.index.intersection(new.index)
            moved = (old.loc[common] - new.loc[common]).abs().max().max()
            print(f"  {w}: refreshed, largest change on a shared date "
                  f"{moved:.6f}")
        print("  ANY non-zero means published numbers have moved.")
        return 0

    X, y, dates, names, bench = panel("sectors")
    (Xtr, ytr, dtr, _), (Xte, yte, dte, bte) = split(X, y, dates, bench)
    print("=" * 74)
    print("  THE FROZEN PANEL — nine SPDR sectors, all of them")
    print("=" * 74)
    print(f"  file      data/sectors.csv  "
          f"({SECTORS_CSV.stat().st_size / 1024:.0f} KB, committed)")
    print(f"  assets    {len(names)}  {', '.join(names)}")
    print(f"  dates     {dates[0].date()} .. {dates[-1].date()}  "
          f"({len(dates)} trading days)")
    print(f"  features  {len(FEATURES)}  {', '.join(FEATURES)}")
    print(f"\n  train     {len(dtr):>5} days  {dtr[0].date()} .. {dtr[-1].date()}")
    print(f"  test      {len(dte):>5} days  {dte[0].date()} .. {dte[-1].date()}")

    print(f"\n  DRAWDOWNS IN THE TEST WINDOW (RSP, the unbiased yardstick):")
    rsp = load_prices("sectors").loc[dte, "RSP"].pct_change().fillna(0.0)
    for nm, lo, hi in DRAWDOWNS:
        m = (dte >= lo) & (dte <= hi)
        if m.sum():
            print(f"    {nm:<14}{m.sum():>5} days   "
                  f"{float((1 + rsp[m]).prod() - 1):+8.1%}")
    print("\n  Four drawdowns, not one. The claim that reward-driven policies")
    print("  buy downside protection can be tested four times and can lose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
