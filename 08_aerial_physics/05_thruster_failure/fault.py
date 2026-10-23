#!/usr/bin/env python3
"""
What a broken thruster does to a vehicle, before anything is trained.

    uv run fault.py                      # the survivability table
    uv run fault.py --rig reveal         # one rig in detail
    uv run fault.py --simultaneous       # how many failures can `dense` take?

THE POINT OF THIS FILE
----------------------
Lab 1 established that full rank is not reachability: nozzles push and
never pull, so the achievable wrench set is a convex POLYTOPE rather
than a subspace, and a rank-6 rig can still be unable to push in some
direction. A failure shrinks that polytope.

So "can this vehicle survive losing nozzle k" is a question about
geometry, answerable by a linear program in milliseconds, and it is
worth answering BEFORE paying for a training run that would otherwise
discover it. The same `--verify-arch` reflex as the LLM labs, pointed
at failure modes.

What the LP answers is whether the remaining nozzles can hold a hover
at all. That is a NECESSARY condition, not a sufficient one: a failure
can be geometrically survivable and still unrecoverable in flight,
because the transient tumble has to be caught before the ground
arrives. Measuring that gap is what the lab is for. Nothing in this
file knows about dynamics, time, or the ground.

TWO FAILURE MODES, AND THEY ARE NOT SYMMETRIC
----------------------------------------------
  stuck shut   the nozzle stops responding and produces nothing.
               The vehicle loses authority.
  stuck open   the nozzle latches wide open and cannot be closed.
               The vehicle loses authority, gains a force and torque
               it did not ask for, AND burns propellant continuously
               until the tank is dry.

Stuck open is strictly worse everywhere this file has been run, and on
some rigs the asymmetry is total: `sixlift` on Mars survives losing
ANY one of its eight nozzles and is killed by ANY one of them
sticking on. A design review that only asked "what if one fails?"
would have passed that vehicle.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import thrusters as TH

HERE = Path(__file__).parent

MODES = ("shut", "open")


def hover_wrench(world) -> np.ndarray:
    """
    The body wrench that holds the vehicle up, in design-point units.

    Divided by the world's thrust scale because the effectiveness
    matrix `B` is built at the design point and the environment
    scales commands by the ratio at runtime. Forgetting that factor
    is how `audit()` once reported Mars lift-to-weight 8.6% low.
    """
    return np.array([0.0, 0.0,
                     TH.wet_mass() * world.g / TH.thrust_scale(world),
                     0.0, 0.0, 0.0])


def can_hover(B: np.ndarray, want: np.ndarray,
              dead: tuple[int, ...] = (), latched: tuple[int, ...] = ()
              ) -> tuple[bool, float]:
    """
    Is there a valve setting that produces `want` exactly?

    `dead` nozzles are clamped shut, `latched` nozzles are clamped wide
    open. Returns (feasible, total valve opening) -- the second value
    is a propellant-rate proxy, so a failure that is survivable but
    expensive shows up as a larger number rather than as a pass.

    Minimising the sum of openings picks the cheapest feasible
    solution, which matters: a latched nozzle can often be cancelled
    in several ways and the vehicle would use the cheapest.
    """
    from scipy.optimize import linprog

    n = B.shape[1]
    lo = np.zeros(n)
    hi = np.ones(n)
    for k in dead:
        hi[k] = 0.0
    for k in latched:
        lo[k] = hi[k] = 1.0
    r = linprog(np.ones(n), A_eq=B, b_eq=want,
                bounds=list(zip(lo, hi)), method="highs")
    return (r.status == 0), (float(r.fun) if r.status == 0 else float("nan"))


def max_lift(B: np.ndarray, world, dead: tuple[int, ...] = (),
             latched: tuple[int, ...] = ()) -> float:
    """
    The largest net UPWARD force the rig can still make, torque-free,
    as a multiple of weight.

    A graded alternative to `can_hover`, and a better-posed question
    for a lander. Hover feasibility asks whether the vehicle can hold
    station forever; landing never requires that. What landing needs
    is enough upward authority to arrest a descent before the ground,
    which is a quantity, not a yes or no.

    Torque is constrained to zero because a vehicle spending its
    authority on spin is not spending it on slowing down.
    """
    from scipy.optimize import linprog

    n = B.shape[1]
    lo = np.zeros(n)
    hi = np.ones(n)
    for k in dead:
        hi[k] = 0.0
    for k in latched:
        lo[k] = hi[k] = 1.0
    # maximise Fz  ->  minimise -Fz, with Fx, Fy and all torque at 0
    A_eq = B[[0, 1, 3, 4, 5], :]
    r = linprog(-B[2], A_eq=A_eq, b_eq=np.zeros(5),
                bounds=list(zip(lo, hi)), method="highs")
    if r.status != 0:
        return 0.0
    weight = TH.wet_mass() * world.g / TH.thrust_scale(world)
    return float(-r.fun) / weight


def survey(rig_name: str, world_name: str) -> dict:
    """Every single-nozzle failure of both modes, for one rig and world."""
    world = TH.WORLDS[world_name]
    rig = TH.rig(rig_name)
    B = TH.effectiveness(rig)
    n = B.shape[1]
    want = hover_wrench(world)

    healthy, base = can_hover(B, want)
    out = {"rig": rig_name, "world": world_name, "n": n,
           "healthy": healthy, "base_valve": base,
           "shut_fatal": [], "open_fatal": [], "open_cost": {}}
    if not healthy:
        return out

    for k in range(n):
        ok, _ = can_hover(B, want, dead=(k,))
        if not ok:
            out["shut_fatal"].append(k)
        ok2, g2 = can_hover(B, want, latched=(k,))
        if not ok2:
            out["open_fatal"].append(k)
        else:
            out["open_cost"][k] = g2 / base
    # The graded predictor, per nozzle and mode.
    out["lift_shut"] = {k: max_lift(B, world, dead=(k,)) for k in range(n)}
    out["lift_open"] = {k: max_lift(B, world, latched=(k,)) for k in range(n)}
    out["lift_healthy"] = max_lift(B, world)
    return out


def simultaneous(rig_name: str, world_name: str, trials: int = 300,
                 upto: int = 8, seed: int = 0) -> list[tuple[int, float]]:
    """
    How many nozzles can go at once before hover is unreachable?

    Random draws rather than every combination: at 20 nozzles the
    exhaustive set for 6 failures is 38,760 linear programs, and the
    answer wanted here is a rate, not a list.
    """
    world = TH.WORLDS[world_name]
    B = TH.effectiveness(TH.rig(rig_name))
    n = B.shape[1]
    want = hover_wrench(world)
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, min(upto, n - 1) + 1):
        fatal = 0
        for _ in range(trials):
            ks = tuple(rng.choice(n, size=m, replace=False))
            if not can_hover(B, want, dead=ks)[0]:
                fatal += 1
        rows.append((m, fatal / trials))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rig", default="")
    ap.add_argument("--simultaneous", action="store_true")
    a, _ = ap.parse_known_args()

    rigs = [a.rig] if a.rig else ["minimal", "sixlift", "reveal",
                                  "fore10", "dense"]

    print("=" * 78)
    print("  WHICH FAILURES CAN THIS VEHICLE SURVIVE? (geometry only)")
    print("=" * 78)
    print("  A failure is 'fatal' here when the remaining nozzles cannot hold")
    print("  a hover at any valve setting. Necessary, not sufficient: the")
    print("  vehicle still has to CATCH the tumble before the ground.\n")
    print(f"  {'rig':<9}{'n':>3}{'world':>7}{'stuck SHUT':>14}{'stuck OPEN':>14}"
          f"{'open costs':>13}")
    for rg in rigs:
        for wn in ("earth", "mars"):
            d = survey(rg, wn)
            if not d["healthy"]:
                print(f"  {rg:<9}{d['n']:>3}{wn:>7}   cannot hover even healthy")
                continue
            cost = d["open_cost"].values()
            cs = (f"{min(cost):.1f}-{max(cost):.1f}x" if cost else "—")
            shut = f"{len(d['shut_fatal'])}/{d['n']}"
            opn = f"{len(d['open_fatal'])}/{d['n']}"
            print(f"  {rg:<9}{d['n']:>3}{wn:>7}{shut:>14}{opn:>14}{cs:>13}")

    print("\n  'open costs' is the extra valve opening needed to hold hover")
    print("  against a latched nozzle, as a multiple of the healthy command.")
    print("  It is a propellant-rate proxy: a survivable failure can still")
    print("  empty the tank before the vehicle reaches the ground.")

    if a.simultaneous:
        print("\n" + "=" * 78)
        print("  HOW MANY AT ONCE? (dense, mars, 300 random draws each)")
        print("=" * 78)
        for m, rate in simultaneous("dense", "mars"):
            print(f"    {m} dead nozzles: {rate:>5.0%} of draws cannot hold hover")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
