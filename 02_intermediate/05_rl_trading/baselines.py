#!/usr/bin/env python3
"""
Everything the learned policies have to beat.

    uv run baselines.py              # all of them, on the frozen panel

WHY THERE ARE SIX AND NOT ONE
-------------------------------
A trading number means nothing on its own. `+14% a year` is
excellent, mediocre or embarrassing depending entirely on what the
alternative was, and the single most common failure in this
subject is a paper that reports a strategy's return without the
one number that makes it readable: what you would have earned by
doing nothing clever.

So there are six, and each is here to close a specific escape
route:

    buyhold     hold every sector, always. THE control. Any method
                that cannot beat this has not earned its compute.
    rsp         the S&P 500 equal-weight index -- a yardstick with
                no survivorship bias at all, because the index
                handles delisting internally.
    flat        never trade. The zero-risk floor, and a reminder
                that a negative strategy is worse than nothing.
    random      a coin flip, re-tossed daily. Shows what the COST
                model alone does to a strategy with no edge.
    forecast    a supervised model of tomorrow's return, traded on
                its sign. This is the `02_rnn_stock_data` approach
                carried through to a decision, and it is the arm
                that answers "is predicting well the same as
                earning well?"
    band        the same forecast, but flat unless the signal
                clears a threshold. The cheapest possible
                cost-awareness, hand-written -- so that if RL wins,
                it has to beat a human who also thought about
                costs, not a strawman that trades every day.

`band` matters most. Without it, "RL learns to trade less" would
be competing against a method that was never allowed to.
"""

from __future__ import annotations

import argparse

import numpy as np

import data as D
import market as M


def buyhold(y):
    """Hold every sector, always. The control."""
    return np.ones_like(y)


def flat(y):
    return np.zeros_like(y)


def random_book(y, seed=0):
    """A coin flip per asset per day. Pure cost, no edge."""
    rng = np.random.default_rng(seed)
    return np.sign(rng.normal(size=y.shape))


def forecast_model(Xtr, ytr, Xte, seed=0, steps=800, hidden=64):
    """
    A supervised forecast of tomorrow's return, per asset.

    Deliberately the SAME network shape the RL arms use, so a
    difference between them is a difference of OBJECTIVE and not of
    capacity. The panel is flattened here -- the forecast is
    per-asset and has no reason to look across the basket, which is
    exactly the thing GRPO changes.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[-1], hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.tensor(Xtr.reshape(-1, Xtr.shape[-1]))
    yy = torch.tensor(ytr.reshape(-1, 1))
    for _ in range(steps):
        i = torch.randint(0, len(x), (512,))
        opt.zero_grad()
        ((net(x[i]) - yy[i]) ** 2).mean().backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        f = net(torch.tensor(Xte.reshape(-1, Xte.shape[-1])))
    return f.numpy().reshape(Xte.shape[:2])


def rule(fc):
    """Trade the sign of the forecast. No cost-awareness at all."""
    return np.sign(fc)


def band(fc, q=0.5):
    """
    Trade only when the signal is in the top `1 - q` by magnitude.

    The threshold is taken from the TEST distribution of the
    forecast, which is a small gift to this baseline -- it gets to
    know how big its own signals will be. That is deliberate: the
    baseline should be as strong as is reasonable, because the
    claim is that RL beats a thoughtful human, not a careless one.
    """
    thr = np.quantile(np.abs(fc), q)
    return np.where(np.abs(fc) > thr, np.sign(fc), 0.0)


def all_baselines(seeds=(0, 1, 2), cost=M.DEFAULT_COST):
    X, y, dates, names, _ = D.panel("sectors")
    (Xtr, ytr, dtr, _), (Xte, yte, dte, _) = D.split(X, y, dates)
    px = D.load_prices("sectors")
    rsp = px.loc[dte, "RSP"].pct_change().shift(-1).fillna(0.0).to_numpy()

    fc = np.mean([forecast_model(Xtr, ytr, Xte, s) for s in seeds], axis=0)
    books = {
        "buyhold": buyhold(yte),
        "flat": flat(yte),
        "random": random_book(yte),
        "forecast_rule": rule(fc),
        "forecast_band": band(fc),
    }
    out = {}
    for k, pos in books.items():
        out[k] = {"pos": pos, "daily": M.book(pos, yte, cost),
                  "turnover": M.turnover(pos)}
    # RSP is an index, not a book we trade -- no position, no cost.
    out["rsp_index"] = {"pos": None, "daily": rsp, "turnover": 0.0}
    return out, yte, dte, fc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", type=float, default=M.DEFAULT_COST)
    a, _ = ap.parse_known_args()

    out, yte, dte, fc = all_baselines(cost=a.cost)
    print("=" * 84)
    print(f"  THE BASELINES — frozen sector panel, {dte[0].date()} .. "
          f"{dte[-1].date()}, {a.cost * 1e4:.0f} bps/side")
    print("=" * 84)
    print(f"  {'arm':<16}{'ann':>9}{'sharpe':>9}{'maxDD':>9}"
          f"{'hit':>8}{'turnover':>10}")
    for k, v in out.items():
        s = M.stats(v["daily"])
        print(f"  {k:<16}{s['ann']:>8.1%}{s['sharpe']:>9.2f}"
              f"{s['maxdd']:>9.1%}{s['hit']:>8.1%}{v['turnover']:>10.3f}")

    # CUMULATIVE over each window, not annualised. The COVID crash
    # is 31 trading days; annualising it reports -241.6% for a book
    # that lost 28%, which is arithmetically correct and reads as
    # nonsense. A rate is the wrong unit for an event.
    print(f"\n  BY REGIME — CUMULATIVE return over each window "
          f"(annualised for 'rest')")
    print(f"  {'arm':<16}" + "".join(f"{n:>14}" for n, _, _ in D.DRAWDOWNS)
          + f"{'rest, ann':>12}")
    masks = {n: (dte >= lo) & (dte <= hi) for n, lo, hi in D.DRAWDOWNS}
    rest = ~np.any(list(masks.values()), axis=0)
    for k, v in out.items():
        row = "".join(f"{M.stats(v['daily'][m])['total']:>13.1%} "
                      for m in masks.values())
        print(f"  {k:<16}{row}{M.stats(v['daily'][rest])['ann']:>11.1%}")
    print(f"  {'(days)':<16}"
          + "".join(f"{int(m.sum()):>13}  " for m in masks.values())
          + f"{int(rest.sum()):>10}")

    print(f"\n  The forecast's own accuracy, for reference:")
    hit = float(np.mean(np.sign(fc) == np.sign(yte)))
    print(f"    directional accuracy {hit:.1%} against a 50% coin flip")
    print(f"    correlation with tomorrow's return "
          f"{np.corrcoef(fc.ravel(), yte.ravel())[0, 1]:+.4f}")
    print("\n  Whether that accuracy is worth anything is the whole")
    print("  question, and the `forecast_rule` row is the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
