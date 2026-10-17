#!/usr/bin/env python3
"""
Search the space of bodies instead of guessing five of them.

    uv run search_bodies.py --plan            # what it would run, free
    uv run search_bodies.py --stage count     # leg count x posture
    uv run search_bodies.py --stage refine --around quadruped

WHY A SEARCH
------------
Lab 8 compared five bodies somebody typed out by hand and found a
4.2x spread in distance walked. That says the body matters; it does
not say the five were well chosen. Quadruped won on distance and
spider8 on stability, which means the interesting shapes are
probably BETWEEN them and nobody has looked.

`robot.make_layout` turns a body into parameters -- leg count,
posture, limb length, fore-aft spread, lateral width -- so the space
can be swept rather than enumerated by hand. Every named body in
labs 5-8 is reproducible from it (`tests/test_bodyspec.py` asserts
this exactly, leg for leg).

TWO STAGES, AND THE ORDER MATTERS
----------------------------------
  stage `count`   leg count x posture, everything else fixed. This
                  is where lab 8's effect lived, so it is measured
                  first and cheaply.
  stage `refine`  limb length and stance width around the winner.
                  Running this first would tune the proportions of
                  a shape that was never the right shape.

WHAT IS HELD FIXED, AND WHY THAT IS THE WHOLE EXPERIMENT
---------------------------------------------------------
Terrain, task, seed, training budget, observation mode, integrator.
Only the body changes. A search that also varied the budget would
report "bigger bodies need more steps" as if it were a fact about
bodies.

EVERY RUN USES `implicitfast`, which is 3.93x faster than the
shipped RK4 and is what MuJoCo recommends for contact-rich models
(RK4 is for energy-conserving systems and can inject contact
vibration). It also CHANGES measured behaviour -- the same policy
walks 7.9 m instead of 5.9 m -- so these numbers are internally
comparable and are NOT comparable with labs 5-8. The re-baselined
`if_*` runs exist so the comparison stays inside this lab.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))

# Fixed across every body in the search. Changing any of these
# mid-search would make the results a story about the budget.
HELD = dict(total_steps=1_500_000, workers=16, n_envs=64,
            eval_every=15, seed=0, integrator="implicitfast")


def candidates(stage: str, around: str | None) -> list[dict]:
    """The bodies to train, as parameter dicts."""
    import robot

    if stage == "count":
        out = []
        for n in (2, 4, 6, 8, 10):
            out.append(dict(name=f"m{n}", n_legs=n, posture="mammal",
                            leg_scale=0.58, spread=0.26, width=0.17))
        for n in (3, 4, 6, 8, 10):
            out.append(dict(name=f"s{n}", n_legs=n, posture="sprawl"))
        # A mammal body needs an even leg count; the generator raises
        # rather than building a lopsided robot, so drop those here
        # instead of discovering it mid-sweep.
        return [c for c in out
                if not (c["posture"] == "mammal" and c["n_legs"] % 2)]

    if stage == "refine":
        base = {"quadruped": dict(n_legs=4, posture="mammal"),
                "hexapod": dict(n_legs=6, posture="mammal"),
                "spider8": dict(n_legs=8, posture="sprawl")}.get(around)
        if base is None:
            raise SystemExit(f"--around must name a body, got {around!r}")
        if base["posture"] == "sprawl":
            raise SystemExit(
                "a sprawl's proportions come from COXA/FEMUR/TIBIA, which "
                "this stage does not vary. Refine a mammal body.")
        out = []
        for scale, width in itertools.product((0.45, 0.58, 0.72),
                                              (0.13, 0.17, 0.22)):
            out.append(dict(name=f"{around[0]}{base['n_legs']}"
                                 f"_l{int(scale*100)}_w{int(width*100)}",
                            leg_scale=scale, width=width, spread=0.26,
                            **base))
        return out

    raise SystemExit(f"unknown stage {stage!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="count", choices=("count", "refine"))
    ap.add_argument("--around", default="quadruped")
    ap.add_argument("--plan", action="store_true",
                    help="list the bodies and the cost, train nothing")
    ap.add_argument("--prefix", default="bs_")
    ap.add_argument("--json", default=None)
    a, _ = ap.parse_known_args()

    import robot

    cands = candidates(a.stage, a.around)
    print("=" * 76)
    print(f"  BODY SEARCH — stage `{a.stage}`, {len(cands)} bodies, "
          f"only the body changes")
    print("=" * 76)
    print(f"  held fixed: " + ", ".join(f"{k}={v}" for k, v in HELD.items()))
    print()
    print(f"  {'name':<16} {'legs':>5} {'posture':<8} {'actuators':>10} "
          f"{'status':>10}")

    rows = []
    for c in cands:
        kw = {k: v for k, v in c.items() if k != "name"}
        layout = robot.make_layout(**kw)
        per = 3 if layout["kind"] == "sprawl" else 4
        nu = len(layout["legs"]) * per + 1
        run = f"{a.prefix}{c['name']}"
        done = (LAB5 / "runs" / run / "summary.json").exists()
        rows.append((c, run, nu, done))
        print(f"  {c['name']:<16} {c['n_legs']:>5} {c['posture']:<8} "
              f"{nu:>10} {'trained' if done else 'to train':>10}")

    todo = [r for r in rows if not r[3]]
    # ~9 min per body under implicitfast, measured on the re-baseline.
    print(f"\n  {len(todo)} to train, roughly {len(todo) * 9} min")
    if a.plan:
        print("\n  --plan: nothing trained.")
        return 0

    t0 = time.time()
    for i, (c, run, nu, done) in enumerate(rows, 1):
        if done:
            print(f"\n  [{i}/{len(rows)}] {run} already trained, skipping")
            continue
        kw = {k: v for k, v in c.items() if k != "name"}
        spec = json.dumps(kw)
        print(f"\n  [{i}/{len(rows)}] training {run}  ({nu} actuators)",
              flush=True)
        cmd = [sys.executable, str(LAB5 / "train_mars.py"),
               "--body-spec", spec, "--name", run, "--quiet",
               "--total-steps", str(HELD["total_steps"]),
               "--workers", str(HELD["workers"]),
               "--n-envs", str(HELD["n_envs"]),
               "--eval-every", str(HELD["eval_every"]),
               "--seed", str(HELD["seed"]),
               "--integrator", HELD["integrator"]]
        r = subprocess.run(cmd, cwd=str(LAB5), capture_output=True, text=True)
        if r.returncode:
            # Loud. A sweep that swallows a failed body reports a
            # shorter table as if it were complete -- lab 8 did
            # exactly that and called two bodies "FAILED training"
            # when argparse had rejected them.
            print(f"      FAILED rc={r.returncode}: "
                  f"{(r.stderr or '').strip().splitlines()[-1:] or ['?']}")
            continue
        s = json.loads((LAB5 / "runs" / run / "summary.json").read_text())
        f = s["final"]
        print(f"      falls {f['fell']:.1%}  to_goal {f['to_goal']:.2f}m  "
              f"{s['wall_seconds']:.0f}s")

    print(f"\n  stage complete in {(time.time()-t0)/60:.0f} min")
    print(f"  measure with:  uv run rank_bodies.py --prefix {a.prefix}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"stage": a.stage, "held": HELD,
             "bodies": [c for c, _r, _n, _d in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
