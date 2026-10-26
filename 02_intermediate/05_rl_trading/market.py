#!/usr/bin/env python3
"""
The decision problem: positions, costs, and what a day of trading is worth.

    uv run market.py                 # self-checks, no training, no network
    uv run market.py --cost-curve    # where any edge dies

WHY THIS IS NOT A REGRESSION
------------------------------
`02_intermediate/02_rnn_stock_data` predicts a number and is scored
by how close it is. That is a complete description of FORECASTING
and it is not a description of TRADING, because three things a
squared error cannot express decide whether a forecast is worth
anything:

    the position is STATE      being long and going long are
                               different acts; one is free and one
                               costs the spread
    acting costs money         a forecast with a tiny edge is
                               correct and unprofitable
    the objective is a PATH    two policies with identical mean
                               return and different drawdowns are
                               not equally good

All three are trivial to write as a reward and impossible to write
as a loss on tomorrow's price. That is the entire argument for
pointing reinforcement learning at this, and it is what this
module makes concrete.

THE ARITHMETIC, IN FULL
-------------------------
For asset i on day t, with position w in [-1, +1] and next-day
return r:

    pnl[i,t] = w[i,t] * r[i,t]  -  cost * |w[i,t] - w[i,t-1]|

`w` is a FRACTION of capital, so +1 is fully long, -1 fully short,
0 flat. The cost term charges the CHANGE, which is what makes
holding free and churning expensive -- and it is the only reason a
policy ever learns to sit still.

COST IS A PARAMETER, NOT A CONSTANT
-------------------------------------
Five basis points per side is a retail-ish round number and it is
not a law of nature. `--cost-curve` sweeps it, because a result
that holds at 0 bps and dies at 3 is not a result about the
method, and reporting the 0 bps number alone is the most common
way this subject lies to itself.
"""

from __future__ import annotations

import argparse

import numpy as np

# Five basis points per side. Charged on the CHANGE in position, so
# a round trip from flat to long and back costs 10 bps in total.
DEFAULT_COST = 0.0005

TRADING_DAYS = 252


def pnl(positions: np.ndarray, returns: np.ndarray,
        cost: float = DEFAULT_COST) -> np.ndarray:
    """
    Daily profit and loss per asset, after costs.

    `positions` and `returns` are both `(T, N)`. Position at row t
    is what you HOLD into day t's return, so it must have been
    decided with information available at t-1 -- the caller's job,
    checked by `assert_causal`.

    The previous position on the first row is taken to be zero: you
    start flat and pay to get in. Starting fully invested for free
    is a small, silent gift that flatters every always-long arm.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    returns = np.atleast_2d(np.asarray(returns, dtype=float))
    if positions.shape != returns.shape:
        raise ValueError(
            f"positions {positions.shape} and returns {returns.shape} "
            f"must match; a mismatch here silently misaligns the book "
            f"against the market by one day or one asset.")
    prev = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
    turnover = np.abs(positions - prev)
    return positions * returns - cost * turnover


def book(positions: np.ndarray, returns: np.ndarray,
         cost: float = DEFAULT_COST) -> np.ndarray:
    """Portfolio P&L per day: the mean across assets, equally weighted."""
    return pnl(positions, returns, cost).mean(axis=1)


def stats(daily: np.ndarray, rf: float = 0.0) -> dict:
    """
    Annualised return, Sharpe, worst drawdown, and the hit rate.

    Drawdown is computed on the COMPOUNDED equity curve rather than
    on the cumulative sum, because a sequence of small losses after
    a large gain is not the same risk as the same losses before it,
    and the sum cannot tell them apart.
    """
    daily = np.asarray(daily, dtype=float)
    if daily.size == 0:
        return {"ann": 0.0, "sharpe": 0.0, "maxdd": 0.0, "hit": 0.0}
    eq = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(eq)
    return {
        "ann": float(np.mean(daily) * TRADING_DAYS),
        "sharpe": float((np.mean(daily) - rf / TRADING_DAYS)
                        / (np.std(daily) + 1e-12) * np.sqrt(TRADING_DAYS)),
        "maxdd": float(np.min(eq / peak) - 1.0),
        "hit": float(np.mean(daily > 0.0)),
        "total": float(eq[-1] - 1.0),
    }


def turnover(positions: np.ndarray) -> float:
    """Mean absolute change in position per asset per day."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    prev = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
    return float(np.mean(np.abs(positions - prev)))


def assert_causal(fn, X: np.ndarray) -> None:
    """
    A policy must not react to information it cannot have.

    Corrupts the future of the feature panel and asserts the
    positions produced for the EARLIER part do not move. A policy
    that reads forward is the one bug in this subject that produces
    beautiful results and cannot be seen in any metric -- so it is
    checked by construction rather than by reading the code.
    """
    rng = np.random.default_rng(0)
    cut = len(X) // 2
    tampered = X.copy()
    tampered[cut:] = rng.normal(size=tampered[cut:].shape).astype(X.dtype)
    # EVERY row before the cut is compared. An earlier version held
    # back the last 40 rows to dodge boundary effects, and that made
    # the check blind to any peek shorter than 40 days -- including
    # the one-day peek that is by far the most likely bug. It passed
    # a policy built to break it.
    a = np.asarray(fn(X))[:cut]
    b = np.asarray(fn(tampered))[:cut]
    if not np.allclose(a, b, atol=1e-6):
        bad = int(np.argmax(np.abs(a - b).max(axis=1) > 1e-6))
        raise RuntimeError(
            f"the policy reacts to data it cannot have seen: positions "
            f"diverge from row {bad} of {cut}, {np.abs(a - b).max():.3e} "
            f"max difference, when only rows >= {cut} were changed.")


def _checks() -> int:
    """Properties of the arithmetic, asserted and watched failing."""
    bad = 0

    def ck(name, ok, detail=""):
        nonlocal bad
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            bad += 1

    rng = np.random.default_rng(0)
    T, N = 500, 9
    r = rng.normal(0.0004, 0.012, (T, N))

    print("  The arithmetic")
    flat = np.zeros((T, N))
    ck("a flat book earns and costs exactly nothing",
       abs(book(flat, r).sum()) < 1e-12)

    long_ = np.ones((T, N))
    gross = float(book(long_, r, cost=0.0).sum())
    net = float(book(long_, r).sum())
    ck("always-long pays the spread ONCE, to get in",
       abs((gross - net) - DEFAULT_COST) < 1e-9,
       f"{(gross - net) * 1e4:.2f} bps total, not per day")

    churn = np.tile(np.array([[1.0], [-1.0]]), (T // 2, N))
    paid = float(book(churn, r, 0.0).sum() - book(churn, r).sum())
    # 1 to get in, then |+1 - -1| = 2 on each of the remaining days.
    expected = (1.0 + 2.0 * (T - 1)) * DEFAULT_COST
    ck("a book that flips every day pays EXACTLY the expected fee",
       abs(paid - expected) < 1e-9,
       f"{paid:.4f} paid against {expected:.4f} predicted — "
       f"{paid / abs(book(churn, r, 0.0).sum() + 1e-12):.0%} of its "
       f"gross return")
    # If the churn test passed with cost=0 too, it would be proving
    # nothing about the cost model.
    ck("...and costs nothing when the cost is zero",
       abs(book(churn, r, cost=0.0).sum()
           - float((churn * r).mean(axis=1).sum())) < 1e-12)

    print("\n  Position is state, not a signal")
    hold = np.ones((T, N))
    rebuy = np.tile(np.array([[1.0], [0.0]]), (T // 2, N))
    ck("holding is free and re-entering is not",
       turnover(hold) < turnover(rebuy) / 100,
       f"turnover {turnover(hold):.5f} vs {turnover(rebuy):.3f} — the "
       f"distinction a squared error on tomorrow's price cannot make")

    print("\n  The path matters, not just the mean")
    # SHUFFLE one return series. Every ordering has the SAME mean by
    # construction, so mean return cannot distinguish them -- and the
    # worst drawdown varies hugely across orderings. An earlier
    # version compared "up then down" against "down then up", which
    # is symmetric and gave -95.1% both ways: a demonstration that
    # demonstrated nothing while printing PASS.
    base = rng.normal(0.0005, 0.02, 1000)
    shuffles = [rng.permutation(base) for _ in range(400)]
    means = np.array([stats(s_)["ann"] for s_ in shuffles])
    dds = np.array([stats(s_)["maxdd"] for s_ in shuffles])
    ck("every ordering has the same annualised return",
       float(means.max() - means.min()) < 1e-9,
       f"{means[0]:+.1%} for all 400 shuffles")
    ck("...and the worst drawdown ranges widely across them",
       float(dds.max() - dds.min()) > 0.10,
       f"{dds.min():.1%} to {dds.max():.1%} — a spread of "
       f"{dds.max() - dds.min():.1%} that the mean cannot see")

    print("\n  Shape errors are caught, not broadcast")
    try:
        pnl(np.ones((T, N)), r[:, :3])
        ck("a mismatched book is REJECTED", False)
    except ValueError:
        ck("a mismatched book is REJECTED", True,
           "numpy would happily broadcast and misalign the book")

    print("\n  Causality")
    try:
        assert_causal(lambda X: X[:, :, 0], rng.normal(size=(T, N, 4)))
        ck("an honest policy passes the causality check", True)
    except RuntimeError as e:
        ck("an honest policy passes the causality check", False, str(e)[:60])
    try:
        assert_causal(lambda X: np.roll(X[:, :, 0], -1, axis=0),
                      rng.normal(size=(T, N, 4)))
        ck("...and a policy that peeks ONE day ahead is CAUGHT", False,
           "the check passed input built to break it")
    except RuntimeError:
        ck("...and a policy that peeks ONE day ahead is CAUGHT", True,
           "one day is the realistic bug; a check that only catches "
           "long peeks catches nothing")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-curve", action="store_true")
    a, _ = ap.parse_known_args()

    print("=" * 74)
    print(f"  THE MARKET — cost {DEFAULT_COST * 1e4:.0f} bps per side, "
          f"charged on the CHANGE in position")
    print("=" * 74)
    bad = _checks()

    if a.cost_curve:
        import data as D

        X, y, dates, names, _ = D.panel("sectors")
        (_, _, _, _), (_, yte, dte, _) = D.split(X, y, dates)
        rng = np.random.default_rng(0)
        print("\n" + "=" * 74)
        print("  WHERE AN EDGE DIES — the same books at different costs")
        print("=" * 74)
        books = {
            "always long": np.ones_like(yte),
            "coin flip, daily": np.sign(rng.normal(size=yte.shape)),
            "perfect foresight": np.sign(yte),
        }
        print(f"  {'book':<22}" + "".join(f"{c:>11}" for c in
                                          ("0 bps", "1 bps", "5 bps",
                                           "10 bps", "25 bps")))
        for nm, pos in books.items():
            row = "".join(
                f"{stats(book(pos, yte, c))['ann']:>10.1%} "
                for c in (0.0, 0.0001, 0.0005, 0.0010, 0.0025))
            print(f"  {nm:<22}{row}")
        print("\n  Perfect foresight survives any cost; a coin flip is")
        print("  destroyed by the smallest. A method that only wins at")
        print("  0 bps has not been shown to win.")

    print("\n" + "=" * 74)
    print(f"  {'FAILED' if bad else 'all checks passed'}")
    print("=" * 74)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
