#!/usr/bin/env python3
"""
How good must perception be before it pays? Finally measurable.

    uv run perception_sweep.py --check        # the setup, no robot
    uv run perception_sweep.py --seeds 16     # the sweep
    uv run perception_sweep.py --arm oracle

THE QUESTION, AND WHY IT TOOK THREE LABS TO ASK IT
---------------------------------------------------
Lab 7 built a model that reads terrain, scored it at 52.9%, and could
not say whether that was useful. Lab 8 tried to find out and could
not: its robot stalled before reaching anything worth deciding about,
so every router -- perfect or blind -- produced the same zero.

Two things changed. Lab 9 found a body that travels (short, wide,
two legs) and an integrator that is 3.9x faster and better suited to
contact. And the task here is sized to the ROBOT rather than to
ambition:

    route band   wp gap   waypoints reached   blockers   stalled
      25-50 m     4.0 m        11/33  (33%)       4.5       75%
      10-16 m     2.5 m         6/14  (43%)       4.5       50%
       6-10 m     2.0 m         4/11  (36%)       4.5       75%

The 10-16 m band is the one where a robot completes legs AND still
meets 4.5 blockers per surface. That is lab 5's lesson for the third
time: size the task to the measured machine. Lab 5 asked for 24 m
from a robot that walked 6 and read the resulting zeros as a terrain
problem.

THE ARMS
--------
The router decides, per landform, OVER or AROUND. Accuracy is the
swept variable:

    oracle      always right -- the ceiling
    p90..p50    correct with probability p, symmetric noise
    biased      lab 7's REAL failure mode: 72% on blockers, 40% on
                open ground. A single accuracy number cannot express
                that, and the traverse may care which way errors lean
    blind       no perception; every landform believed crossable,
                which is what lab 5's robot actually did

WHAT EACH ERROR COSTS
---------------------
    wrong AROUND -> the planner detours round passable ground.
                    Costs DISTANCE.
    wrong OVER   -> the planner routes through a scarp. The robot
                    walks into it and stalls. Costs the RUN.

Those are lab 7's two confusion cells, priced differently for the
first time. An accuracy figure prices them identically.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
LAB8 = HERE.parent / "08_long_traverse"
for p in (LAB5, LAB8):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Sized to the measured robot. See the docstring table.
ROUTE_BAND = (10.0, 16.0)
WAYPOINT_GAP = 2.5


def belief_sensitive_task(seed: int, band, min_gap: float = 1.5,
                          tries: int = 240):
    """
    A task where what the router BELIEVES changes the route.

    THE CHECK THAT CAUGHT THIS. On a randomly chosen A-B pair, the
    oracle, blind and paranoid beliefs all planned the identical
    10.5 m route: the pair simply did not pass near a blocker, so
    blocking every landform changed nothing. A sweep over such tasks
    measures router accuracy against a route nobody can affect, and
    every arm scores the same by construction.

    This is lab 4 and lab 5's lesson one level up. There it was the
    ratio of route to arena -- at 13% the robot walked past the
    obstacles and the task contained no choice while every arrival
    number still looked reasonable. Here it is the same failure
    wearing a different hat: the task must be SELECTED for
    containing a decision, not assumed to contain one.

    So a pair is kept only if believing everything impassable costs
    at least `min_gap` metres against believing nothing is. That
    gap IS the decision; without it there is nothing for perception
    to be right or wrong about.

    `tries` is 240 rather than 40 because the binding constraint is
    finding two STANDABLE points in the route band at all, not the
    gap threshold -- measured over 20 seeds, 40 tries found a task
    on 13 and 240 found one on all 20, with the same mean gap. A
    lower `min_gap` does not help; more attempts do.
    """
    import traverse as T

    import mars

    s = mars.generate(seed, scale="traverse")
    ok = mars.traversable(s, T.STEP_MAX)
    rng = np.random.default_rng(seed)
    e = s.extent
    n_lf = len(s.landforms)
    free = T.belief_mask(s, [False] * n_lf)
    paranoid_mask = T.belief_mask(s, [True] * n_lf)
    for _ in range(tries):
        a = (rng.uniform(-e / 2 + 3, e / 2 - 3),
             rng.uniform(-e / 2 + 3, e / 2 - 3))
        b = (rng.uniform(-e / 2 + 3, e / 2 - 3),
             rng.uniform(-e / 2 + 3, e / 2 - 3))
        if not (mars.standable(s, *a) and mars.standable(s, *b)):
            continue
        _p, L = T.plan(s, a, b, ok)
        if not (math.isfinite(L) and band[0] <= L <= band[1]):
            continue
        _p, l_free = T.plan(s, a, b, free)
        _p, l_par = T.plan(s, a, b, paranoid_mask)
        if not math.isfinite(l_free):
            continue
        # Unreachable-when-paranoid is the strongest possible
        # sensitivity: a wrong AROUND would wall the robot in.
        gap = (float("inf") if not math.isfinite(l_par)
               else l_par - l_free)
        if gap >= min_gap:
            return s, a, b, L, gap
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def arms():
    import traverse as T

    return {
        "oracle": T.Router("oracle"),
        "p90": T.NoisyRouter("p=0.90", 0.90),
        "p80": T.NoisyRouter("p=0.80", 0.80),
        "p70": T.NoisyRouter("p=0.70", 0.70),
        "p60": T.NoisyRouter("p=0.60", 0.60),
        "biased": T.BiasedRouter("biased(lab7)"),
        "blind": T.BlindRouter("blind"),
    }


def run_arm(name, router, body_run, seeds, max_steps):
    import torch  # noqa: F401  (loaded via traverse)

    import traverse as T
    from mars_env import NavWorld
    from robot import make_layout

    T.WAYPOINT_SPACING = WAYPOINT_GAP
    T.WAYPOINT_RADIUS = max(0.8, WAYPOINT_GAP * 0.4)

    meta = json.loads((LAB5 / "runs" / body_run / "summary.json").read_text())
    layout = make_layout(**meta["body_spec"]) if meta.get("body_spec") else None
    net, norm, _m = T.load_policy(body_run)

    reached = total = 0
    arrived = stalled = 0
    dists, ivs, ratios, n = [], [], [], 0
    for sd in seeds:
        task = belief_sensitive_task(sd, ROUTE_BAND)
        if task is None:
            continue
        s, A, B, _L, _gap = task
        s.start, s.goal = A, B
        env = NavWorld(mode=meta["mode"], flat=False, seed=0,
                       relief_scale=meta.get("relief_scale", 1.0),
                       upright_w=meta.get("upright_w", 0.0),
                       integrator=meta.get("integrator"), layout=layout)
        T.place(env, s, A)
        r = T.run_traverse(env, net, norm, s, A, B, router,
                           np.random.default_rng(50_000 + sd),
                           max_steps=max_steps)
        n += 1
        reached += r.wp_reached
        total += r.wp_total
        arrived += int(r.arrived)
        stalled += int(r.stalled)
        dists.append(r.distance)
        ivs.append(r.interventions)
        # How much longer the BELIEVED route is than the true optimum.
        # A router that cries wolf pays here, in metres.
        if math.isfinite(r.planned_len) and math.isfinite(r.oracle_len) \
                and r.oracle_len > 0.1:
            ratios.append(r.planned_len / r.oracle_len)
    if not n:
        return None
    lo, hi = wilson(reached, max(total, 1))
    return {"arm": name, "n": n,
            "wp_reached": reached, "wp_total": total,
            "wp_rate": reached / max(total, 1), "wp_ci": [lo, hi],
            "arrived": arrived / n, "stalled": stalled / n,
            "distance_m": float(np.mean(dists)),
            "interventions": float(np.mean(ivs)),
            "detour": float(np.mean(ratios)) if ratios else float("nan")}


def check() -> int:
    """Is the task actually decidable? No robot involved."""
    import traverse as T

    fails = []

    def ck(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("=" * 74)
    print("  LAB 10 — is the task decidable before a robot is involved?")
    print("=" * 74)

    blk, lens, gaps = [], [], []
    for sd in range(4, 20):
        t = belief_sensitive_task(sd, ROUTE_BAND)
        if t is None:
            continue
        s, _A, _B, L, gap = t
        blk.append(sum(T.truth_blocked(s)))
        lens.append(L)
        gaps.append(gap)
    ck("tasks exist in the band", len(lens) >= 8,
       f"{len(lens)} of 16 seeds solvable")
    ck("routes are the length we sized for",
       ROUTE_BAND[0] <= np.mean(lens) <= ROUTE_BAND[1],
       f"mean {np.mean(lens):.1f} m")
    ck("every surface has something to decide about",
       min(blk) >= 1, f"blockers per surface: min {min(blk)}, "
                      f"mean {np.mean(blk):.1f}")
    finite = [g for g in gaps if math.isfinite(g)]
    ck("belief changes the route on EVERY selected task",
       min(gaps) >= 1.5,
       f"paranoid-vs-blind gap: min {min(gaps):.1f} m, "
       f"{len(gaps)-len(finite)} tasks where paranoia walls it in")

    # A sweep is meaningless if the arms plan the SAME route.
    t = belief_sensitive_task(4, ROUTE_BAND)
    s, A, B, _L, _g = t
    truth = T.truth_blocked(s)
    import mars
    _p, oracle_len = T.plan(s, A, B, T.belief_mask(s, truth))
    _p, blind_len = T.plan(s, A, B, T.belief_mask(s, [False] * len(truth)))
    _p, paranoid = T.plan(s, A, B, T.belief_mask(s, [True] * len(truth)))
    ck("beliefs produce DIFFERENT routes",
       abs(oracle_len - paranoid) > 0.5 or not math.isfinite(paranoid),
       f"oracle {oracle_len:.1f} m, blind {blind_len:.1f} m, "
       f"paranoid {paranoid:.1f} m")

    print("=" * 74)
    if fails:
        print(f"  {len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("  the task is decidable; the sweep can mean something")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="bs_m2",
                    help="lab 9's best traveller, course-trained (bodies "
                         "trained ON this preset measured WORSE)")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=7000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default="runs/sweep.json")
    a, _ = ap.parse_known_args()

    if a.check:
        return check()

    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    todo = arms() if a.arm is None else {a.arm: arms()[a.arm]}

    print("=" * 80)
    print(f"  PERCEPTION SWEEP — body {a.body}, {len(seeds)} seeds, "
          f"routes {ROUTE_BAND[0]:.0f}-{ROUTE_BAND[1]:.0f} m")
    print("=" * 80)
    print(f"  {'arm':<10} {'waypoints':>18} {'arrived':>8} {'stalled':>8} "
          f"{'dist':>7} {'interv':>7} {'detour':>7}")

    rows = []
    for name, router in todo.items():
        r = run_arm(name, router, a.body, seeds, a.max_steps)
        if r is None:
            print(f"  {name:<10}  no solvable tasks")
            continue
        rows.append(r)
        wp = (f"{r['wp_reached']}/{r['wp_total']} "
              f"[{r['wp_ci'][0]:.0%},{r['wp_ci'][1]:.0%}]")
        print(f"  {name:<10} {wp:>18} {r['arrived']:>7.0%} "
              f"{r['stalled']:>7.0%} {r['distance_m']:>6.1f}m "
              f"{r['interventions']:>7.1f} {r['detour']:>6.2f}x", flush=True)

    if len(rows) >= 2:
        best = max(rows, key=lambda r: r["wp_rate"])
        blind = next((r for r in rows if r["arm"] == "blind"), None)
        print()
        if blind:
            gap = best["wp_rate"] - blind["wp_rate"]
            overlap = best["wp_ci"][0] <= blind["wp_ci"][1]
            print(f"  best arm `{best['arm']}` {best['wp_rate']:.0%} vs "
                  f"blind {blind['wp_rate']:.0%}  (+{gap:.0%})")
            if overlap:
                print("  INTERVALS OVERLAP: perception does not measurably "
                      "pay at this budget.\n  Reporting the gap as a result "
                      "would be reading noise.")
            else:
                print("  Intervals are separated: perception pays, and the "
                      "sweep says how much\n  accuracy is needed to get it.")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
