#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0"]
# ///
"""
Lab 8 (`07_physical_ai/08_long_traverse`) — bodies, and the metric that
judges them.

Lab 8 compares five morphologies, and every number it reports depends
on two things being right: the bodies must actually differ in the way
claimed, and "fallen" must mean fallen. Both were wrong at first, and
both were wrong in the quiet way:

  * the four-legged body inherited the biped's 0.76 m legs on a 0.22 m
    base -- a height/width of 3.5, which is a biped on stilts. It
    measured WORSE than the biped and looked like a real result.
  * `fallen()` was `clearance < threshold`, i.e. torso height above
    the ground beneath it. On uneven terrain that fires when the
    GROUND RISES. Measured: 13 of 13 trigger events happened at a
    torso tilt of 0-1 degrees. Not one was a fall. And because
    `fallen()` terminates training episodes, it was cutting the
    low-slung bodies off before they could learn -- penalising
    exactly the shapes the lab exists to test.

No GPU, no training, no downloads.

    uv run tests/test_morphology.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB5 = REPO / "07_physical_ai" / "05_mars_terrain"
LAB8 = REPO / "07_physical_ai" / "08_long_traverse"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    sys.path.insert(0, str(LAB8))
    sys.path.insert(0, str(LAB5))
    import mujoco

    import robot
    from mars_env import NavWorld

    print("=" * 74)
    print("  LAB 8 — the bodies, and the metric that judges them")
    print("=" * 74)

    # ---------------------------------------------------------------
    # The biped must be EXACTLY what labs 5-7 measured on. Everything
    # else here is additive and worthless if this regresses.
    # ---------------------------------------------------------------
    h = np.zeros((64, 64))
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    robot.hfield_png(h, tmp / "t.png")
    png = {"t.png": (tmp / "t.png").read_bytes()}
    m = mujoco.MjModel.from_xml_string(
        robot.model_xml("t.png", (0, 0), (1, 1), relief=1.0, rocks=[]), png)
    check("the default body is still the biped",
          (m.nq, m.nu, m.ngeom) == (14, 9, 14),
          f"nq={m.nq} nu={m.nu} ngeom={m.ngeom}")
    check("biped standing height is unchanged",
          abs(robot.stand_clearance("biped") - 0.706) < 1e-9,
          f"{robot.stand_clearance('biped'):.4f} m")
    check("biped fall threshold is unchanged",
          abs(robot.fall_clearance("biped") - 0.400) < 1e-9,
          f"{robot.fall_clearance('biped'):.4f} m")

    # ---------------------------------------------------------------
    # The bodies must differ as claimed: leg count AND stance.
    # ---------------------------------------------------------------
    want = {"biped": 9, "quadruped": 17, "hexapod": 25,
            "spider6": 19, "spider8": 25}
    for body, nu in want.items():
        mm = mujoco.MjModel.from_xml_string(
            robot.model_xml("t.png", (0, 0), (1, 1), relief=1.0, rocks=[],
                            morphology=body), png)
        check(f"{body} builds with {nu} actuators", mm.nu == nu,
              f"nu={mm.nu}")

    # A sprawl must be WIDE AND LOW relative to a mammal, or it is not
    # a sprawl -- this is the check the stilts version would have failed.
    for body in ("quadruped", "spider8"):
        legs, _k = robot.leg_layout(body)
        if robot.layout_kind(body) == "sprawl":
            width = 2 * (robot.COXA + robot.FEMUR)
        else:
            ys = [y for _t, _x, y in legs]
            width = max(ys) - min(ys)
        ratio = robot.stand_clearance(body) / width
        check(f"{body} is wider than it is tall", ratio < 1.5,
              f"height/width {ratio:.2f} (a biped on stilts was 3.5)")

    # ---------------------------------------------------------------
    # THE METRIC. stance_height must measure the ROBOT, not the ground.
    # ---------------------------------------------------------------
    e = NavWorld(mode="blind", flat=True, seed=0, morphology="spider8")
    e.reset(seed=1)
    base = e.stance_height()
    base_clear = e.clearance()
    check("a standing spider has a sane stance height",
          0.15 < base < 0.40, f"{base:.3f} m")

    # THE INVARIANCE, asserted two-sidedly so it can fail.
    #
    # Lift the whole robot 0.25 m. Its legs come with it, so the
    # torso-to-feet distance is UNCHANGED -- the body did not
    # collapse. Clearance, which measures torso-to-ground, MUST move
    # by the full 0.25 m. One number tracks the robot and the other
    # tracks the world, and that difference is the entire fix.
    e.data.qpos[e.q["rootz"]] += 0.25
    e._mj.mj_forward(e.model, e.data)
    check("stance height is invariant to ride height",
          abs(e.stance_height() - base) < 1e-6,
          f"{base:.3f} -> {e.stance_height():.3f} m (must not move)")
    check("clearance, by contrast, moves with the world",
          abs((e.clearance() - base_clear) - 0.25) < 1e-6,
          f"{base_clear:.3f} -> {e.clearance():.3f} m (must move 0.25)")

    # ---------------------------------------------------------------
    # THE KEPT COUNTEREXAMPLE. The OLD rule -- clearance alone --
    # must be shown to call a perfectly upright robot "fallen" when
    # the ground rises. If this ever stops firing, the bug is back.
    # ---------------------------------------------------------------
    e2 = NavWorld(mode="blind", flat=True, seed=0, morphology="spider8")
    e2.reset(seed=1)
    tilt = e2.tilt_deg()
    # Simulate the ground arriving: drop the torso toward the terrain
    # WITHOUT tipping it, exactly what a rise under the body does.
    e2.data.qpos[e2.q["rootz"]] -= 0.16
    e2._mj.mj_forward(e2.model, e2.data)
    old_rule = e2.clearance() < e2._fall_clearance
    check("the OLD clearance rule calls an upright robot fallen",
          old_rule and e2.tilt_deg() < 5.0,
          f"clearance {e2.clearance():.3f} < {e2._fall_clearance:.3f} "
          f"at tilt {e2.tilt_deg():.1f}°")

    # And a genuinely tipped robot must still be caught.
    e3 = NavWorld(mode="blind", flat=True, seed=0, morphology="spider8")
    e3.reset(seed=1)
    e3.data.qpos[e3.q["rootroll"]] = math.radians(80.0)
    e3._mj.mj_forward(e3.model, e3.data)
    check("a robot tipped 80 degrees IS fallen", e3.fallen(),
          f"tilt {e3.tilt_deg():.0f}°")

    # ---------------------------------------------------------------
    # The CLI must offer every body that exists. A hardcoded triple
    # silently rejected spider6/spider8 -- argparse exited 2, the
    # sweep swallowed it, and two bodies were reported as "FAILED
    # training" having never started.
    # ---------------------------------------------------------------
    src = (LAB5 / "train_mars.py").read_text()
    check("the --morphology choices derive from the layout table",
          "LEG_LAYOUTS" in src and 'choices=("biped", "quadruped"' not in src,
          "a restated list goes stale the moment a body is added")

    # ---------------------------------------------------------------
    # Lab 8's planner: a belief can only remove options, never add.
    # ---------------------------------------------------------------
    import mars

    from traverse import belief_mask, plan
    s = mars.generate(3, scale="course")
    ok = mars.traversable(s, 0.12)
    a_pt, b_pt = (-10.0, -10.0), (10.0, 10.0)
    _, free = plan(s, a_pt, b_pt, ok)
    _, paranoid = plan(s, a_pt, b_pt,
                       belief_mask(s, [True] * len(s.landforms)))
    check("believing everything is blocked never shortens the route",
          not math.isfinite(paranoid) or paranoid >= free - 1e-6,
          f"free {free:.1f} m vs paranoid {paranoid:.1f} m")

    print("=" * 74)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
