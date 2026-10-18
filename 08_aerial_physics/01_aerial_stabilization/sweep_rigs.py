#!/usr/bin/env python3
"""
Does thruster COUNT matter, or does PLACEMENT? Train one policy per rig.

    uv run sweep_rigs.py --audit                 # free: the LP, no training
    uv run sweep_rigs.py --baseline              # cheap: classical, no training
    uv run sweep_rigs.py --steps 4000000         # the experiment
    uv run sweep_rigs.py --rigs reveal,waisted

THE QUESTION
------------
Topic 7 spent ten labs discovering that the machine, not the
algorithm, was the constraint -- and then that it was the machine's
PROPORTIONS rather than its part count. A short wide biped beat every
multi-legged body at the same actuator count.

This asks the same question of thrusters, where it can be answered
much more sharply, because a thruster rig has a closed-form
description of what it can do: the 6 x N control effectiveness
matrix. So the hypothesis is checkable before any training, and then
checkable again after.

THE CONTROL THAT MAKES IT AN EXPERIMENT
----------------------------------------
Every rig is built at EQUAL INSTALLED THRUST. A 20-nozzle rig has
nozzles half the size of a 10-nozzle rig's. Without that, "more
nozzles is better" would just be "more thrust is better", which
needs no experiment and would be true of any vehicle.

THE FALSIFIER, WRITTEN BEFORE THE RUN
--------------------------------------
If COUNT is what matters, clean-landing rate is monotone in nozzle
count and `waisted` (6 nozzles at the waist) beats `minimal` (6 at
the corners) or ties it. If PLACEMENT is what matters, the ordering
breaks and the rank-deficient rig collapses regardless of count.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
ORDER = ["minimal", "waisted", "sixlift", "narrow", "reveal", "wide", "dense"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rigs", default=",".join(ORDER))
    ap.add_argument("--steps", type=int, default=4_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=96)
    ap.add_argument("--audit", action="store_true",
                    help="the effectiveness matrix only — free, no training")
    ap.add_argument("--baseline", action="store_true",
                    help="the classical controller per rig — no training")
    ap.add_argument("--json", default="runs/sweep.json")
    a, _ = ap.parse_known_args()

    sys.path.insert(0, str(HERE))
    import thrusters as TH

    rigs = [r.strip() for r in a.rigs.split(",") if r.strip()]

    if a.audit:
        print("=" * 78)
        print("  WHAT EACH RIG CAN DO — closed form, before a single step")
        print("=" * 78)
        print(f"  {'rig':<10} {'N':>3} {'rank':>5} {'pure roll':>11} "
              f"{'pure pitch':>11} {'purity':>7} {'360° in':>8}")
        for nm in rigs:
            d = TH.audit(nm, TH.rig(nm))
            t = d["min_roll_time_s"]
            print(f"  {nm:<10} {d['n_thrusters']:>3} {d['rank']:>4}/6 "
                  f"{d['roll_pure_Nm']:>10,.0f}N "
                  f"{d['pitch_pure_Nm']:>10,.0f}N {d['purity']:>6.0%} "
                  f"{(f'{t:.2f}s' if math.isfinite(t) else 'never'):>8}")
        print("\n  A rank of 5 means one wrench direction cannot be produced")
        print("  at ANY command. No policy can learn around that — it is a")
        print("  property of where the nozzles are bolted.")
        return 0

    if a.baseline:
        from classical import run as run_pd

        print("=" * 78)
        print(f"  CLASSICAL BASELINE PER RIG — {a.episodes} held-out throws")
        print("=" * 78)
        rows = []
        for nm in rigs:
            d = TH.audit(nm, TH.rig(nm))
            b = run_pd(nm, "rolling", False, a.episodes, seed0=900_000)
            lo, hi = wilson(b["landed_ok"], a.episodes)
            rows.append({**b, "n": d["n_thrusters"], "rank": d["rank"],
                         "ci": [lo, hi]})
            print(f"  {nm:<10} N={d['n_thrusters']:>2} rank {d['rank']}/6  "
                  f"landed {b['rate']:>5.0%} [{lo:.0%},{hi:.0%}]  "
                  f"sink {b['sink']:.2f} m/s  gas {b['gas']:.0f} kg",
                  flush=True)
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rows, indent=2))
        return 0

    # The real sweep. One training run per rig, SEQUENTIALLY -- the
    # worker pool already owns every core, so running two trainings
    # at once halves both rather than finishing sooner.
    print("=" * 78)
    print(f"  RIG SWEEP — {len(rigs)} rigs x {a.steps:,} steps")
    print("=" * 78)
    results = []
    for i, nm in enumerate(rigs):
        t0 = time.time()
        print(f"\n  [{i + 1}/{len(rigs)}] training {nm} ...", flush=True)
        name = f"r_{nm}"
        p = subprocess.run(
            [sys.executable, str(HERE / "train_aerial.py"),
             "--rig", nm, "--name", name,
             "--total-steps", str(a.steps),
             "--n-envs", str(a.n_envs),
             "--eval-episodes", str(a.episodes)],
            cwd=HERE, capture_output=True, text=True)
        run = HERE / "runs" / name
        if p.returncode != 0 or not (run / "summary.json").exists():
            # A failed training is a DATA POINT and must look like
            # one. Topic 7 had a sweep report "FAILED training" for
            # bodies whose argument parser simply rejected the name.
            print(f"      FAILED rc={p.returncode}")
            print("      " + "\n      ".join(p.stderr.strip().splitlines()[-6:]))
            results.append({"rig": nm, "failed": True,
                            "rc": p.returncode,
                            "stderr": p.stderr[-2000:]})
            continue
        s = json.loads((run / "summary.json").read_text())
        ev = s.get("eval", {})
        results.append({"rig": nm, "failed": False,
                        "n": s["act_dim"],
                        "rank": s["rig_audit"]["rank"],
                        "roll_pure": s["rig_audit"]["roll_pure_Nm"],
                        "purity": s["rig_audit"]["purity"],
                        "rate": ev.get("rate", float("nan")),
                        "ci": ev.get("ci", [0, 0]),
                        "sink": ev.get("sink", float("nan")),
                        "gas": ev.get("gas", float("nan")),
                        "minutes": (time.time() - t0) / 60})
        r = results[-1]
        print(f"      {nm}: landed {r['rate']:.0%} "
              f"[{r['ci'][0]:.0%},{r['ci'][1]:.0%}]  "
              f"sink {r['sink']:.2f}  gas {r['gas']:.0f} kg  "
              f"({r['minutes']:.1f} min)", flush=True)
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(results, indent=2))

    good = [r for r in results if not r["failed"]]
    good.sort(key=lambda r: -r["rate"])
    print("\n" + "=" * 78)
    print(f"  {'rig':<10} {'N':>3} {'rank':>5} {'pure roll':>11} "
          f"{'landed':>8} {'95% CI':>14} {'sink':>8} {'gas':>7}")
    print("  " + "-" * 74)
    for r in good:
        ci = f"[{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]"
        print(f"  {r['rig']:<10} {r['n']:>3} {r['rank']:>4}/6 "
              f"{r['roll_pure']:>10,.0f}N {r['rate']:>7.0%} "
              f"{ci:>14} {r['sink']:>6.2f}m/s {r['gas']:>5.0f}kg")

    if good:
        by_count = sorted(good, key=lambda r: r["n"])
        monotone = all(by_count[i]["rate"] <= by_count[i + 1]["rate"] + 1e-9
                       for i in range(len(by_count) - 1))
        print(f"\n  monotone in nozzle count? {'YES' if monotone else 'NO'}")
        print("  That was the falsifier, written before the run: if count")
        print("  is what matters, this is YES.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
