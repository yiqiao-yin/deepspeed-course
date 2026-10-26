#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "pandas>=2.0", "torch==2.11.0"]
# ///
"""
02_intermediate/05_rl_trading — the three ways this lab could lie.

Trading research has a small number of mistakes that produce
spectacular results, raise nothing, and look completely normal in
every chart. This lab's claims are only worth the checks that rule
them out, so each gets one here.

  1. LOOK-AHEAD. A feature or a policy that reads one day forward
     turns noise into a perfect strategy. It is the defining bug of
     the subject. Checked by TAMPERING with the future and asserting
     nothing earlier moves -- and by asserting that a deliberately
     peeking policy IS caught, because a causality check that has
     never rejected anything is decoration.

  2. COSTS NOT CHARGED. Every edge in this area survives at 0 bps
     and most die by 5. The cost model is asserted to charge the
     CHANGE in position, to make holding free, and to destroy a
     book that churns.

  3. THE COMPARISON IS RIGGED. A result against no baseline, or
     against a strawman, or on a survivorship-biased universe, is
     not a result. The published universe is asserted to be
     unbiased against an index that cannot be, and the biased one
     is asserted to still be inflated -- so the lab's own
     demonstration of the bias cannot silently stop working.

Properties, not shapes. No GPU, no network, no training.

    uv run tests/test_rl_trading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "02_intermediate" / "05_rl_trading"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    sys.path.insert(0, str(LAB))
    import data as D
    import market as M

    print("=" * 78)
    print("  02_intermediate/05_rl_trading")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. Look-ahead.
    # ------------------------------------------------------------------
    print("\n  Nothing may read forward in time")
    px = D.load_prices("sectors")
    c = px["XLK"]
    cut = len(c) // 2
    rng = np.random.default_rng(0)
    tampered = c.copy()
    tampered.iloc[cut:] = rng.uniform(20, 300, len(c) - cut)
    a = D.features_for(c).iloc[:cut - 1]
    b = D.features_for(tampered).iloc[:cut - 1]
    leaking = [col for col in D.FEATURES
               if not np.allclose(a[col].to_numpy(), b[col].to_numpy(),
                                  equal_nan=True)]
    check("no feature moves when only the FUTURE is corrupted",
          not leaking, f"leaking: {leaking}" if leaking else
          f"all {len(D.FEATURES)} features unchanged")
    check("...and a feature that DOES peek is caught",
          not np.allclose(c.shift(-1).iloc[:cut].to_numpy(),
                          tampered.shift(-1).iloc[:cut].to_numpy(),
                          equal_nan=True),
          "the slice must REACH the tamper point — an earlier version "
          "stopped one row short and its counterexample came back clean")

    X = rng.normal(size=(400, 9, 4)).astype(np.float32)
    check("assert_causal passes an honest policy",
          not _raises(lambda: M.assert_causal(lambda Z: Z[:, :, 0], X)))
    check("...and REJECTS one that peeks a single day",
          _raises(lambda: M.assert_causal(
              lambda Z: np.roll(Z[:, :, 0], -1, axis=0), X)),
          "one day is the realistic bug; a check that only catches long "
          "peeks catches nothing")

    print("\n  The split is chronological and the scaler sees training only")
    Xp, yp, dates, names, _ = D.panel("sectors")
    (Xtr, ytr, dtr, _), (Xte, yte, dte, _) = D.split(Xp, yp, dates)
    check("every training date precedes every test date",
          dtr.max() < dte.min(), f"{dtr.max().date()} < {dte.min().date()}")
    check("the training slice is standardised, the test slice is NOT",
          abs(float(Xtr.mean())) < 0.02 and abs(float(Xte.mean())) > 1e-3,
          f"train mean {Xtr.mean():+.4f}, test mean {Xte.mean():+.4f} — "
          f"a test mean of exactly zero would mean the scaler saw it")

    # ------------------------------------------------------------------
    # 2. Costs.
    # ------------------------------------------------------------------
    print("\n  Costs are charged on the CHANGE, and they bite")
    T, N = 400, 9
    r = rng.normal(0.0004, 0.012, (T, N))
    check("a flat book earns and costs nothing",
          abs(M.book(np.zeros((T, N)), r).sum()) < 1e-12)
    hold = np.ones((T, N))
    paid = float(M.book(hold, r, 0.0).sum() - M.book(hold, r).sum())
    check("holding pays the spread exactly ONCE",
          abs(paid - M.DEFAULT_COST) < 1e-9,
          f"{paid * 1e4:.2f} bps in total, not per day")
    churn = np.tile(np.array([[1.0], [-1.0]]), (T // 2, N))
    paid = float(M.book(churn, r, 0.0).sum() - M.book(churn, r).sum())
    expect = (1.0 + 2.0 * (T - 1)) * M.DEFAULT_COST
    check("flipping daily pays exactly the predicted fee",
          abs(paid - expect) < 1e-9, f"{paid:.4f} against {expect:.4f}")
    check("...and the SAME book is free when the cost is zero",
          abs(float(M.book(churn, r, 0.0).sum()
                    - (churn * r).mean(axis=1).sum())) < 1e-12,
          "or the test above would prove nothing about the cost model")
    check("a mismatched book is REJECTED, not broadcast",
          _raises(lambda: M.pnl(np.ones((T, N)), r[:, :3])))

    print("\n  Mean return cannot see risk; the drawdown can")
    base = rng.normal(0.0005, 0.02, 800)
    sh = [rng.permutation(base) for _ in range(200)]
    means = np.array([M.stats(s)["ann"] for s in sh])
    dds = np.array([M.stats(s)["maxdd"] for s in sh])
    check("every ordering of one return series has the same mean",
          float(means.max() - means.min()) < 1e-9, f"{means[0]:+.1%}")
    check("...and wildly different worst drawdowns",
          float(dds.max() - dds.min()) > 0.10,
          f"{dds.min():.1%} to {dds.max():.1%} — the spread a squared "
          f"error on tomorrow's price cannot express")

    # ------------------------------------------------------------------
    # 3. The comparison.
    # ------------------------------------------------------------------
    print("\n  The universe is unbiased, and the biased one still is")
    sec = D.load_prices("sectors")
    sur = D.load_prices("survivors")
    common = sec.index.intersection(sur.index)
    rs = sec.loc[common].pct_change().dropna()
    ru = sur.loc[common].pct_change().dropna()
    rsp = float(rs["RSP"].mean() * 252)
    ew_sec = float(rs[D.SECTORS].mean(axis=1).mean() * 252)
    ew_sur = float(ru[D.SURVIVORS].mean(axis=1).mean() * 252)
    check("the published sector universe tracks an unbiased index",
          abs(ew_sec - rsp) < 0.03,
          f"{ew_sec:.2%} against RSP's {rsp:.2%} — a gap of "
          f"{ew_sec - rsp:+.2%}")
    check("...and the hand-picked survivors are still inflated",
          ew_sur - rsp > 0.05,
          f"{ew_sur:.2%}, a gap of {ew_sur - rsp:+.2%}/yr — if this ever "
          f"stops holding, the lab's demonstration of survivorship bias "
          f"has quietly broken")
    check("the sector universe is ALL of them, not a chosen subset",
          len(D.SECTORS) == 9 and "RSP" not in D.SECTORS,
          "no selection decision means no selection bias")

    print("\n  Every claim has something to beat")
    import baselines as BL

    for fn, nm in ((BL.buyhold, "buyhold"), (BL.flat, "flat")):
        pos = fn(yte)
        check(f"`{nm}` is a real book of the right shape",
              pos.shape == yte.shape)
    bh = M.stats(M.book(BL.buyhold(yte), yte))
    check("buy-and-hold is positive over this window",
          bh["ann"] > 0.05,
          f"{bh['ann']:+.1%}/yr, drawdown {bh['maxdd']:.1%} — the control "
          f"every learned arm must beat")
    coin = M.stats(M.book(BL.random_book(yte), yte))
    check("a daily coin flip is destroyed by costs",
          coin["ann"] < -0.05,
          f"{coin['ann']:+.1%}/yr — if this were near zero the cost model "
          f"would not be biting and no result here would mean anything")

    # ------------------------------------------------------------------
    # 4. GRPO's group baseline is what it claims to be.
    # ------------------------------------------------------------------
    print("\n  GRPO's advantage is an excess return over the basket")
    import torch

    import train_rl as TR

    R = torch.tensor(rng.normal(0.001, 0.01, (50, 9)), dtype=torch.float32)
    adv, ret = TR.advantages("grpo", None, None, R, 0.95, 0.95)
    check("the group advantage sums to zero on every day",
          float(adv.sum(dim=1).abs().max()) < 1e-4,
          "subtracting the basket mean IS taking an excess return")
    check("...and GRPO builds no critic target at all",
          ret is None, "no value function, no bootstrapping")
    net = TR.build(5, 16, critic=False)
    check("a GRPO policy carries no value head",
          net.v_body is None and net.v_head is None,
          f"{net.n_params():,} params")
    net2 = TR.build(5, 16, critic=True)
    check("...and a PPO policy does, at roughly twice the size",
          net2.n_params() > 1.8 * net.n_params(),
          f"{net2.n_params():,} against {net.n_params():,}")

    print("\n" + "=" * 78)
    print(f"  {'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
