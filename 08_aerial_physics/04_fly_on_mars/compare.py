#!/usr/bin/env python3
"""
Score the learned policy against the guidance law, PAIRED.

    uv run compare.py runs/mars_res_s0 --world mars --episodes 96

WHY PAIRED, AND WHY McNEMAR
----------------------------
Both controllers fly the SAME held-out courses from the same seeds,
so the right question is not "are these two rates different" but
"on the courses where they disagree, which one wins". Comparing two
independent binomial intervals throws that pairing away: at 96
episodes it cannot separate 79% from 73% even when one controller
wins nearly every disagreement, because most of the variance is the
course draw rather than the controller.

McNemar's test uses only the discordant pairs -- courses one flew
and the other did not -- which is exactly the information the
pairing buys. The same test settled the perception question in
`07_physical_ai/10_perception_pays`.

The exact binomial form is used rather than the chi-square
approximation: the discordant count here is small, and the
approximation is poor below about 25.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def exact_mcnemar(b: int, c: int) -> float:
    """
    Two-sided exact McNemar on the discordant pairs.

    Under the null the b + c disagreements split 50/50, so this is a
    binomial sign test on them.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def run_pair(run_dir: str, world: str, episodes: int, seed0: int,
             **env_kw) -> dict:
    import torch

    import classical
    from evaluate import load_policy
    from flight_env import FlightEnv

    net, norm = load_policy(run_dir)

    # Two envs, identical construction, so the course a seed produces
    # is the same for both arms. The policy needs the residual
    # interface and the guidance law needs valves.
    learned = FlightEnv("dense", world=world, action="residual", **env_kw)
    sched = FlightEnv("dense", world=world, action="valves", **env_kw)

    rows = []
    for ep in range(episodes):
        sd = seed0 + ep
        o = learned.reset(sd)
        while True:
            with torch.no_grad():
                x = torch.as_tensor(norm(o), dtype=torch.float32).unsqueeze(0)
                a = net.distribution(x).mean.squeeze(0).numpy()
            o, _r, d, _ = learned.step(np.clip(a, 0.0, 1.0))
            if d:
                break

        sched.reset(sd)
        ctl = classical.FlightController(sched)
        while True:
            _o, _r, d, _ = sched.step(ctl.act())
            if d:
                break

        assert np.allclose(np.array(learned.gates), np.array(sched.gates)), (
            "the two arms flew DIFFERENT courses from the same seed — "
            "the pairing is void and so is the test")
        rows.append(dict(seed=sd,
                         learned=bool(learned.success()),
                         sched=bool(sched.success()),
                         learned_gas=learned.gas_used,
                         sched_gas=sched.gas_used,
                         learned_outcome=learned.outcome,
                         sched_outcome=sched.outcome))
    return {"rows": rows}


def report(rows: list[dict]) -> dict:
    n = len(rows)
    lk = sum(r["learned"] for r in rows)
    sk = sum(r["sched"] for r in rows)
    b = sum(1 for r in rows if r["learned"] and not r["sched"])
    c = sum(1 for r in rows if r["sched"] and not r["learned"])
    p = exact_mcnemar(b, c)
    lg = float(np.mean([r["learned_gas"] for r in rows]))
    sg = float(np.mean([r["sched_gas"] for r in rows]))

    print("=" * 74)
    print(f"  PAIRED COMPARISON — {n} held-out courses, same seeds")
    print("=" * 74)
    for name, k in (("learned (residual PPO)", lk), ("scheduled guidance", sk)):
        lo, hi = wilson(k, n)
        print(f"  {name:<24}{k:>4}/{n}  {k / n:>5.0%}  [{lo:.2f}, {hi:.2f}]")
    print()
    print(f"  courses only the LEARNED arm flew     {b}")
    print(f"  courses only the SCHEDULED arm flew   {c}")
    print(f"  exact McNemar p = {p:.4f}"
          f"   {'significant' if p < 0.05 else 'NOT significant'} at 0.05")
    print()
    # PROPELLANT, ON THE COURSES BOTH ARMS FLEW.
    #
    # Averaging gas over every episode is a trap: a controller that
    # crashes at second two spends almost nothing, so failing more
    # often looks like efficiency. Only courses BOTH arms completed
    # are comparable, and on those the comparison is paired, so it
    # is a signed-rank test rather than two means.
    both = [r for r in rows if r["learned"] and r["sched"]]
    print(f"  propellant, ALL episodes      learned {lg:6.1f} kg   "
          f"scheduled {sg:6.1f} kg   ({lg / sg - 1:+.0%})")
    if len(both) >= 8:
        bl = np.array([r["learned_gas"] for r in both])
        bs = np.array([r["sched_gas"] for r in both])
        from scipy.stats import wilcoxon
        stat, pg = wilcoxon(bl, bs)
        print(f"  propellant, BOTH flew ({len(both):>2})    learned "
              f"{bl.mean():6.1f} kg   scheduled {bs.mean():6.1f} kg   "
              f"({bl.mean() / bs.mean() - 1:+.0%})")
        print(f"  paired Wilcoxon p = {pg:.2e}"
              f"   {'significant' if pg < 0.05 else 'NOT significant'}"
              f" at 0.05   (n = {len(both)} shared courses)")
        out_gas = {"n_both": len(both), "learned": float(bl.mean()),
                   "sched": float(bs.mean()), "p": float(pg)}
    else:
        out_gas = {}
        print("  too few shared completions for a paired gas test")
    print()
    print("  The pairing is what makes this readable: both arms fly the")
    print("  same courses, so the discordant pairs isolate the controller")
    print("  from the draw.")
    return {"n": n, "learned": lk, "sched": sk, "b": b, "c": c,
            "p": p, "learned_gas": lg, "sched_gas": sg, "gas": out_gas}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--world", default="mars", choices=("earth", "mars"))
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--seed0", type=int, default=900_000)
    ap.add_argument("--json", default="")
    a, _ = ap.parse_known_args()

    import sys
    sys.path.insert(0, str(HERE))
    d = run_pair(a.run, a.world, a.episodes, a.seed0)
    out = report(d["rows"])
    if a.json:
        Path(a.json).write_text(json.dumps({**out, "rows": d["rows"]}, indent=1))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
