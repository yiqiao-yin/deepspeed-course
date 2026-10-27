#!/usr/bin/env python3
"""
Physics is 55-67% of a training step. Make it cheaper, honestly.

    uv run speed_physics.py                 # the options, measured
    uv run speed_physics.py --body spider8

WHY PHYSICS AND NOT THE LEARNER
--------------------------------
Lab 6's corrected Amdahl figures put physics at 49% of a biped
training step and 56% of a spider8 one, and a direct profile of the
loop agrees (67% physics, 32% PPO, everything else under 1%). An
earlier version of that number said 21% and sent this lab after the
PPO update, which would have been most of a day spent on a third of
the cost.

Lab 9 needs to train MANY bodies, so a physics speedup multiplies
across the whole search. Two candidates, and the first is large:

  1. THE INTEGRATOR. The model uses `RK4`, which evaluates the
     dynamics FOUR times per step. MuJoCo's `implicitfast` costs
     about one and is the documented default for exactly this kind
     of contact-rich model.

  2. SELF-COLLISION. Every geom carries contype=1/conaffinity=1, so
     an eight-legged body checks its own legs against each other
     every step. With 33 geoms that is a lot of pairs for an event
     that rarely matters.

A SPEEDUP THAT CHANGES THE PHYSICS IS NOT A SPEEDUP
----------------------------------------------------
It is a different experiment. So this measures two things for every
option, and the second is the one with teeth:

  * steps/s -- is it faster?
  * trajectory divergence from RK4 over a fixed open-loop rollout --
    is it still the same world?

Labs 5-8's published numbers were all measured under RK4. If an
option is adopted it must either agree closely enough that trained
behaviour transfers, or the lab must say plainly that its numbers
are not comparable to the earlier ones. `vecenv.py` had to prove
bit-identity before anyone trusted its worker pool; this is the same
bar, relaxed only because a different integrator cannot be
bit-identical by construction.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LAB5 = HERE.parent / "05_mars_terrain"
if str(LAB5) not in sys.path:
    sys.path.insert(0, str(LAB5))


def build(body: str, integrator: str = "RK4",
          self_collide: bool = True):
    """The lab 5 course with one physics option overridden."""
    import tempfile

    import mujoco

    import mars
    import robot

    task = mars.make_task(0, route_band=(6.0, 11.0), min_detour=1.25,
                          forced=True, scale="course")
    if task is None:
        raise SystemExit("no solvable task")
    h = task.heights - task.heights.min()
    tmp = Path(tempfile.mkdtemp())
    robot.hfield_png(h, tmp / "t.png")
    xml = robot.model_xml("t.png", task.start, task.goal,
                          relief=max(float(h.max()), 1e-3),
                          rocks=task.rocks, morphology=body)
    xml = xml.replace('integrator="RK4"', f'integrator="{integrator}"')
    if not self_collide:
        # Robot limb geoms stop colliding with each other while still
        # colliding with the world. Terrain and rocks keep group 1;
        # limbs move to group 2 and accept only group 1.
        xml = xml.replace('<geom type="capsule" size="0.045"',
                          '<geom contype="2" conaffinity="1" '
                          'type="capsule" size="0.045"')
    return mujoco.MjModel.from_xml_string(
        xml, {"t.png": (tmp / "t.png").read_bytes()})


def rate(model, steps: int = 400) -> float:
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    for _ in range(30):
        mujoco.mj_step(model, d)
    t0 = time.time()
    for _ in range(steps):
        mujoco.mj_step(model, d)
    return steps / (time.time() - t0)


def divergence(a, b, steps: int = 400) -> float:
    """
    How far apart do two models drift on an IDENTICAL open-loop
    rollout? Returned as max |qpos| difference in metres/radians.

    Same actions, same start. Any difference is the physics option.
    """
    import mujoco

    da, db = mujoco.MjData(a), mujoco.MjData(b)
    mujoco.mj_resetData(a, da)
    mujoco.mj_resetData(b, db)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(steps):
        u = rng.uniform(-0.3, 0.3, a.nu)
        da.ctrl[:] = u
        db.ctrl[:] = u
        mujoco.mj_step(a, da)
        mujoco.mj_step(b, db)
        worst = max(worst, float(np.abs(da.qpos - db.qpos).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="spider8")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--json", default="runs/speed.json")
    a, _ = ap.parse_known_args()

    base = build(a.body, "RK4", True)
    base_rate = rate(base, a.steps)

    print("=" * 76)
    print(f"  PHYSICS OPTIONS — {a.body}, nq={base.nq} nu={base.nu} "
          f"ngeom={base.ngeom}")
    print("=" * 76)
    print(f"  {'option':<34} {'steps/s':>9} {'speedup':>8} "
          f"{'divergence from RK4':>21}")
    print(f"  {'RK4 + self-collision (shipped)':<34} {base_rate:>9,.0f} "
          f"{1.0:>7.2f}x {'-- (baseline)':>21}")

    out = {"body": a.body, "baseline_steps_per_s": base_rate, "options": {}}
    trials = [
        ("implicitfast", "implicitfast", True),
        ("Euler", "Euler", True),
        ("RK4, no self-collision", "RK4", False),
        ("implicitfast, no self-collision", "implicitfast", False),
    ]
    for label, integ, selfc in trials:
        m = build(a.body, integ, selfc)
        r = rate(m, a.steps)
        d = divergence(base, m, a.steps)
        out["options"][label] = {"steps_per_s": r, "speedup": r / base_rate,
                                 "divergence_m": d}
        print(f"  {label:<34} {r:>9,.0f} {r/base_rate:>7.2f}x "
              f"{d:>20.3f}m")

    print("\n  Divergence is max |qpos| difference over "
          f"{a.steps} identical open-loop steps.")
    print("  Small (< ~0.05) means a policy trained under one should "
          "transfer to the\n  other. Large means it is a DIFFERENT WORLD "
          "and the lab's numbers would\n  not be comparable with labs 5-8.")

    best = max(out["options"], key=lambda k: out["options"][k]["speedup"])
    b = out["options"][best]
    print(f"\n  fastest: {best} at {b['speedup']:.2f}x, "
          f"divergence {b['divergence_m']:.3f} m")
    if b["divergence_m"] > 0.05:
        print("  -> NOT adoptable silently: it changes the physics. Either "
              "retrain\n     everything under it and say so, or keep RK4.")
    else:
        print("  -> adoptable: same world, cheaper.")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
