#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "mujoco>=3.2", "pillow>=10.0"]
# ///
"""
The navigation task must have a decision in it, and the robot must stand.

`07_physical_ai/04_terrain_navigation` asks whether a robot can tell a
climbable ridge from an impassable one and route accordingly. Four
things have to be true before that question means anything, and every
one of them was FALSE in some version of this lab:

  1. THE MAP MUST BE SOLVABLE. An unsolvable arena scores every policy
     at zero and reads as a hard task rather than a broken one. This
     repository has shipped that twice.

  2. CLIMBING MUST PAY. With round mounds, refusing to climb cost a
     median of 0.00 m over 109 maps -- the "learn to climb" half of the
     lab was decoration. Ridge LENGTH is what fixes it, and the test
     pins the property rather than the parameter.

  3. THE TWO OBSTACLE CLASSES MUST DIFFER. A wall that the traversal
     rule treats as climbable is not a wall.

  4. THE ROBOT MUST BE ABLE TO STAND. The first version collapsed in 27
     steps with any action, and 1M training steps on FLAT ground
     reached 0% arrival. Worse, the fall detector was set at 0.55,
     inside the healthy oscillation band (0.49-0.84), so episodes ended
     on a standing robot's startup wobble.

Shape assertions would pass on all four failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LAB = REPO / "07_physical_ai" / "04_terrain_navigation"

_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def test_maps(world) -> None:
    """Solvable, and with a real detour to make."""
    solvable = detours = 0
    ratios = []
    for s in range(60):
        m = world.generate(s)
        p, length = world.solve(m)
        if not p:
            continue
        solvable += 1
        r = length / max(world.straight_line(m), 1e-6)
        ratios.append(r)
        detours += r > 1.15
    check("most maps are solvable", solvable >= 55, f"{solvable}/60")
    check("most maps need a real detour", detours / solvable > 0.7,
          f"{detours}/{solvable} over 15% longer than the straight line")
    check("the detour is substantial", np.median(ratios) > 1.2,
          f"median route/straight = {np.median(ratios):.2f}")


def test_climbing_pays(world) -> None:
    """
    Refusing to climb must COST something.

    This is the property the ridge geometry exists to produce, and it
    is asserted instead of the geometry so that a future edit to the
    lengths is caught by its consequence rather than by a magic number.
    """
    gains = []
    for s in range(60):
        m = world.generate(s)
        p, L = world.solve(m)
        if not p:
            continue
        blocked = world.Map(heights=m.heights.copy(), start=m.start,
                            goal=m.goal, seed=s)
        blocked.heights[blocked.heights > 0.02] = 9.9
        pn, Ln = world.solve(blocked)
        if pn:
            gains.append(Ln - L)
    g = np.array(gains)
    check("refusing to climb costs real distance", np.median(g) > 1.0,
          f"median {np.median(g):+.2f} m, mean {g.mean():+.2f} m")
    check("and the choice is usually still AVAILABLE",
          len(gains) / 60 > 0.8,
          f"{len(gains)}/60 maps have both a climbing and a going-round route")


def test_two_classes_differ(world) -> None:
    """A wall must be impassable and a step must not be."""
    step_ok = wall_blocked = total_s = total_w = 0
    for s in range(30):
        m = world.generate(s)
        ok = world.traversable(m.heights)
        for o in m.obstacles:
            i = int((o.cx + world.ARENA / 2) / world.CELL)
            j = int((o.cy + world.ARENA / 2) / world.CELL)
            if o.kind == "step":
                total_s += 1
                step_ok += bool(ok[j, i])
            else:
                total_w += 1
                # A wall's own interior may be flat on top; what must be
                # blocked is its EDGE, which is where a robot would try
                # to step up. Sample across the ridge.
                band = ok[max(0, j-12):j+12, max(0, i-12):i+12]
                wall_blocked += bool((~band).any())
    check("climbable ridges are traversable",
          step_ok / max(total_s, 1) > 0.9, f"{step_ok}/{total_s}")
    check("impassable ridges block something",
          wall_blocked / max(total_w, 1) > 0.9, f"{wall_blocked}/{total_w}")
    check("the threshold sits between the two classes",
          world.STEP_MAX < world.WALL_MIN,
          f"STEP_MAX {world.STEP_MAX} < WALL_MIN {world.WALL_MIN}")


def test_observation_is_derived(NavWorld) -> None:
    """
    obs_dim must come from the model, not from a typed constant.

    Lab 3 shipped this quantity as a literal, it went stale, and three
    training runs died on a broadcast error while the arms that never
    touched it sailed through.
    """
    src = (LAB / "nav_env.py").read_text()
    check("PROPRIO is not a hand-typed literal",
          "PROPRIO = 1" not in src and "self.model.nq" in src)
    for mode in ("blind", "privileged"):
        env = NavWorld(mode=mode, flat=True, seed=0)
        obs, _ = env.reset(seed=1)
        check(f"declared obs_dim matches reality ({mode})",
              len(obs) == env.obs_dim, f"{len(obs)} vs {env.obs_dim}")
    env = NavWorld(mode="privileged", flat=True, seed=0)
    blind = NavWorld(mode="blind", flat=True, seed=0)
    check("privileged really does carry more than blind",
          env.obs_dim > blind.obs_dim,
          f"{env.obs_dim} vs {blind.obs_dim}")


def test_robot_stands(NavWorld) -> None:
    """
    The fall detector must fire on a collapse and stay silent on a
    standing robot -- BOTH directions, measured.

    A detector set inside the healthy oscillation band is worse than no
    detector: it ends episodes on the one behaviour the policy got
    right, and the training log is indistinguishable from "the robot
    cannot stand".
    """
    env = NavWorld(mode="blind", flat=True, seed=0)
    out = {}
    for name, a in (("crouch", -np.ones(7)), ("stand", np.zeros(7)),
                    ("straight", np.ones(7))):
        env.reset(seed=1)
        lo, fired = 9.0, False
        for _ in range(250):
            env.step(a.copy())
            lo = min(lo, env.clearance())
            fired |= env.fallen()
        out[name] = (lo, fired)

    check("a standing robot is NOT flagged as fallen", not out["stand"][1],
          f"min clearance {out['stand'][0]:.2f}, "
          f"threshold {env.FALL_CLEARANCE}")
    check("a collapsed robot IS flagged", out["crouch"][1],
          f"min clearance {out['crouch'][0]:.2f}")
    check("the threshold sits between the two",
          out["crouch"][0] < env.FALL_CLEARANCE < out["stand"][0],
          f"{out['crouch'][0]:.2f} < {env.FALL_CLEARANCE} < {out['stand'][0]:.2f}")
    check("zero action holds the standing pose", out["stand"][0] > 0.4,
          "position actuators centred on STAND; torque control collapsed "
          "in 27 steps and sent a 1M-step run to 0%")


def test_goal_is_relative(NavWorld) -> None:
    """
    The goal must be given in the ROBOT's frame.

    A world-frame goal invites memorising positions in a fixed arena.
    Rotating the robot in place must change the goal features while the
    distance stays put.
    """
    import mujoco

    env = NavWorld(mode="blind", flat=True, seed=0)
    env.reset(seed=2)
    d0 = env.to_goal()
    f0 = env._goal_feats().copy()
    env.data.qpos[3] += 1.2                      # turn on the spot
    mujoco.mj_forward(env.model, env.data)
    f1 = env._goal_feats()
    check("turning changes the goal BEARING",
          abs(f1[1] - f0[1]) > 0.1 or abs(f1[2] - f0[2]) > 0.1,
          f"sin/cos {f0[1:].round(2)} -> {f1[1:].round(2)}")
    check("turning does not change the goal RANGE",
          abs(env.to_goal() - d0) < 1e-6)


def main() -> int:
    sys.path.insert(0, str(LAB))
    import world
    from nav_env import NavWorld

    print("=" * 74)
    print("  The task must have a decision in it, and the robot must stand")
    print("=" * 74)
    test_maps(world)
    test_climbing_pays(world)
    test_two_classes_differ(world)
    test_observation_is_derived(NavWorld)
    test_robot_stands(NavWorld)
    test_goal_is_relative(NavWorld)

    print()
    if _fails:
        print(f"  {len(_fails)} FAILED: {', '.join(_fails)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
