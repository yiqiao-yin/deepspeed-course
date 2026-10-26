#!/usr/bin/env python3
"""
Score every arm against the controls, by regime, against the falsifiers.

    uv run evaluate.py
    uv run evaluate.py --json results.json

THE TWO FALSIFIERS, WRITTEN BEFORE THE ARMS WERE TRAINED
----------------------------------------------------------
    A. RISK.  The reward-driven arms must cut the worst drawdown by
       at least a third against buy-and-hold while keeping at least
       70% of its return. If they do not, optimising P&L buys
       nothing a loss function could not, and the lab's central
       claim is wrong here.

    B. SELECTION.  GRPO's group baseline is the equal-weight market,
       so its advantage is an excess return and the policy should be
       learning which sector beats the others. The DOLLAR-NEUTRAL
       book tests that directly: strip out the net market exposure
       and only the relative call remains. It must be positive, or
       the group baseline did not learn selection and any gain is
       market timing wearing a long-short costume.

Both are printed as HOLDS or FAILS whichever way they fall. Topic
3's prefill lab wrote its falsifier first, got FAILS on the first
run, and documented a regime it would otherwise have missed.

WHY A BOOTSTRAP AND NOT A t-TEST
----------------------------------
Daily returns are fat-tailed, serially dependent and heteroskedastic,
so the standard error of a Sharpe ratio from a t-test is optimistic
in a way that matters -- it is routine to find "significant" edges
in pure noise. A stationary block bootstrap resamples CONTIGUOUS
blocks, which preserves the autocorrelation that makes financial
series hard, and it is the cheapest honest answer available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import baselines as B
import data as D
import market as M

HERE = Path(__file__).parent

DD_CUT = 1 / 3      # falsifier A: cut drawdown by at least this much
KEEP = 0.70         # falsifier A: keep at least this share of the return
BLOCK = 21          # bootstrap block, one trading month


def block_bootstrap(daily: np.ndarray, n: int = 2000,
                    block: int = BLOCK, seed: int = 0) -> np.ndarray:
    """
    Stationary block bootstrap of the annualised mean.

    Resamples contiguous blocks so serial dependence survives. An
    iid bootstrap -- or a t-test -- would treat 2,428 days as 2,428
    independent observations, which they are not, and would report
    an interval far too tight.
    """
    rng = np.random.default_rng(seed)
    T = len(daily)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, T, nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        out[i] = daily[idx].mean() * M.TRADING_DAYS
    return out


def load_runs(root: Path) -> dict:
    """Pool the seeds of each arm, keeping the per-seed spread."""
    arms: dict[str, list] = {}
    for d in sorted(root.glob("*/")):
        f = d / "summary.json"
        p = d / "positions.npy"
        if not (f.exists() and p.exists()):
            continue
        s = json.loads(f.read_text())
        base = d.name.rsplit("_s", 1)[0]
        arms.setdefault(base, []).append((s, np.load(p)))
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--json", default="")
    ap.add_argument("--cost", type=float, default=M.DEFAULT_COST)
    a, _ = ap.parse_known_args()

    base, yte, dte, fc = B.all_baselines(cost=a.cost)
    arms = load_runs(HERE / a.runs)
    if not arms:
        raise SystemExit("no runs found — run ./run_arms.sh first")

    rows: dict[str, dict] = {}
    for k, v in base.items():
        rows[k] = {"daily": v["daily"], "turnover": v["turnover"],
                   "seeds": [], "params": 0}
    for k, runs in arms.items():
        # MEDIAN across seeds, not mean: one diverged seed should not
        # drag the published figure, and the spread is printed anyway.
        dailies = [M.book(p, yte, a.cost) for _s, p in runs]
        anns = [M.stats(d)["ann"] for d in dailies]
        pick = int(np.argsort(anns)[len(anns) // 2])
        rows[k] = {"daily": dailies[pick], "all": dailies,
                   "turnover": float(np.median([s["turnover"] for s, _ in runs])),
                   "seeds": anns, "params": runs[0][0]["params"]}

    order = ["buyhold", "rsp_index", "ppo", "grpo", "grpo_neutral",
             "forecast_band", "forecast_rule", "flat", "random"]
    order = [o for o in order if o in rows]

    print("=" * 94)
    print(f"  THE ARMS — nine SPDR sectors, {dte[0].date()} .. "
          f"{dte[-1].date()}, {a.cost * 1e4:.0f} bps/side")
    print("=" * 94)
    print(f"  {'arm':<16}{'ann':>8}{'95% CI':>18}{'sharpe':>8}"
          f"{'maxDD':>9}{'turnover':>10}{'params':>9}")
    for k in order:
        r = rows[k]
        s = M.stats(r["daily"])
        ci = np.percentile(block_bootstrap(r["daily"]), [2.5, 97.5])
        pr = f"{r['params']:,}" if r["params"] else "—"
        print(f"  {k:<16}{s['ann']:>7.1%}"
              f"{f'[{ci[0]:+.1%}, {ci[1]:+.1%}]':>18}{s['sharpe']:>8.2f}"
              f"{s['maxdd']:>9.1%}{r['turnover']:>10.3f}{pr:>9}")

    print(f"\n  per seed, because a median hides the spread:")
    for k in order:
        if rows[k]["seeds"]:
            print(f"    {k:<16}"
                  + "  ".join(f"{v:+.1%}" for v in sorted(rows[k]["seeds"])))

    # --- by regime ----------------------------------------------------
    print(f"\n" + "=" * 94)
    print("  BY REGIME — CUMULATIVE over each drawdown, annualised for the rest")
    print("=" * 94)
    masks = {n: (dte >= lo) & (dte <= hi) for n, lo, hi in D.DRAWDOWNS}
    rest = ~np.any(list(masks.values()), axis=0)
    print(f"  {'arm':<16}"
          + "".join(f"{n:>15}" for n in masks) + f"{'rest, ann':>12}")
    print(f"  {'(days)':<16}"
          + "".join(f"{int(m.sum()):>15}" for m in masks.values())
          + f"{int(rest.sum()):>12}")
    for k in order:
        d = rows[k]["daily"]
        print(f"  {k:<16}"
              + "".join(f"{M.stats(d[m])['total']:>14.1%} "
                        for m in masks.values())
              + f"{M.stats(d[rest])['ann']:>11.1%}")

    # --- the falsifiers ------------------------------------------------
    bh = M.stats(rows["buyhold"]["daily"])
    print(f"\n" + "=" * 94)
    print("  THE FALSIFIERS")
    print("=" * 94)
    verdicts = {}

    print(f"\n  A. RISK — cut the drawdown by a third, keep 70% of the return")
    print(f"     buy-and-hold: {bh['ann']:+.1%}/yr, worst drawdown "
          f"{bh['maxdd']:.1%}")
    for k in ("ppo", "grpo"):
        if k not in rows:
            continue
        s = M.stats(rows[k]["daily"])
        cut = 1 - s["maxdd"] / bh["maxdd"]
        keep = s["ann"] / bh["ann"]
        ok = cut >= DD_CUT and keep >= KEEP
        verdicts[f"A_{k}"] = bool(ok)
        print(f"     {k:<14} drawdown {s['maxdd']:>7.1%} "
              f"({cut:>5.0%} smaller)   return {keep:>5.0%} of it   "
              f"{'HOLDS' if ok else 'FAILS'}")

    print(f"\n  B. SELECTION — the dollar-neutral book must be positive")
    if "grpo_neutral" in rows:
        s = M.stats(rows["grpo_neutral"]["daily"])
        ci = np.percentile(block_bootstrap(rows["grpo_neutral"]["daily"]),
                           [2.5, 97.5])
        ok = s["ann"] > 0 and ci[0] > 0
        verdicts["B"] = bool(ok)
        print(f"     grpo_neutral   {s['ann']:+.1%}/yr, 95% CI "
              f"[{ci[0]:+.1%}, {ci[1]:+.1%}]   "
              f"{'HOLDS' if ok else 'FAILS'}")
        if not ok:
            print(f"     The group baseline did NOT learn to pick sectors.")
            print(f"     Strip the market exposure out and nothing is left,")
            print(f"     so the gain above is market timing rather than")
            print(f"     cross-sectional selection. Reported as measured.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"arms": {k: {**M.stats(v["daily"]),
                          "turnover": v["turnover"], "seeds": v["seeds"]}
                      for k, v in rows.items()},
             "verdicts": verdicts,
             "window": [str(dte[0].date()), str(dte[-1].date())],
             "cost": a.cost}, indent=1))
        print(f"\n  wrote {a.json}")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
