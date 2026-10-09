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


def test_frames_agree(NavWorld) -> None:
    """
    `pos()` must equal where MuJoCo actually puts the robot.

    It did not. `rootx`/`rooty` are SLIDE joints, and the torso body
    was declared at pos="(start)" in the XML, so the world position
    was start + qpos -- while reset() also wrote start into qpos. The
    robot was spawned at TWICE its start coordinates and every
    distance, every terrain sample and the arrival test ran in a frame
    shifted by the start offset: to_goal() read 10.44 m where the true
    distance was 14.78 m.

    Eighteen property checks passed throughout, because not one of
    them compared the lab's own idea of position against MuJoCo's.
    A reader spotted it by noticing that the track in an animation did
    not line up with the robot.
    """
    import mujoco
    import numpy as np

    env = NavWorld(mode="privileged", flat=False, seed=0)
    worst = 0.0
    for s in range(6):
        env.reset(seed=400 + s)
        mujoco.mj_forward(env.model, env.data)
        tid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        worst = max(worst, float(np.linalg.norm(
            env.pos() - env.data.xpos[tid][:2])))
        for _ in range(40):                     # and after moving, too
            env.step(np.zeros(env.act_dim))
        mujoco.mj_forward(env.model, env.data)
        worst = max(worst, float(np.linalg.norm(
            env.pos() - env.data.xpos[tid][:2])))
    check("pos() matches MuJoCo's own torso position", worst < 1e-6,
          f"worst disagreement {worst:.2e} m over 6 resets, before and "
          f"after motion")

    # And the goal the distance is measured against must be where the
    # goalpost is actually drawn -- otherwise "arrived" means reaching
    # somewhere the reader cannot see.
    env.reset(seed=404)
    mujoco.mj_forward(env.model, env.data)
    g = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_pole")
    err = float(np.linalg.norm(
        np.array(env.map.goal) - env.data.geom_xpos[g][:2]))
    check("the goalpost is drawn AT the goal", err < 1e-6,
          f"{err:.2e} m")


def test_efficiency_denominator(world) -> None:
    """
    The optimum must actually BE the optimum.

    Path efficiency was scored against `solve()`, a 4-connected
    breadth-first search. A 4-connected path cannot move diagonally, so
    it staircases: a straight diagonal of length L comes back as
    L*sqrt(2). Three evaluation episodes therefore scored 133%, 136% and
    144% efficiency -- a policy beating the optimum, which is not a
    thing that can happen.

    And it was invisible, because `evaluate.py` wrapped the ratio in
    `min(..., 1.0)`. Every impossible value was silently rewritten to
    "exactly 100%": the clamp took the one signal that the denominator
    was broken and converted it into a plausible number. An impossible
    measurement is EVIDENCE.

    The counterexample below is kept permanently, per
    CLAUDE.md's rule that a check you have not watched reject bad input
    is not a check.
    """
    import numpy as np

    # The counterexample, in its purest form: flat ground, A and B on a
    # perfect diagonal. The true distance is the hypotenuse; a
    # 4-connected walk must take every step as a rook.
    flat = world.Map(heights=np.zeros((world.N, world.N)),
                     start=(-4.0, -4.0), goal=(4.0, 4.0), seed=0)
    _, bfs = world.solve(flat)
    geo = world.geodesic(flat)
    true = world.straight_line(flat)
    check("on open flat ground the geodesic IS the straight line",
          abs(geo - true) / true < 0.02,
          f"geodesic {geo:.2f} m vs straight {true:.2f} m")
    check("the 4-connected oracle overstates it by ~sqrt(2)",
          1.35 < bfs / true < 1.45,
          f"solve() {bfs:.2f} m = {bfs/true:.3f}x the true distance "
          f"(sqrt(2) = 1.414) -- this is the bug, pinned")

    # And across real maps it must never be LONGER than the 4-connected
    # path, since every 4-connected path is also an 8-connected one.
    worse = 0
    ratios = []
    for s in range(40):
        m = world.generate(s)
        p, L = world.solve(m)
        if not p:
            continue
        g = world.geodesic(m)
        if g <= 0:
            continue
        ratios.append(L / g)
        worse += g > L + 1e-9
    check("the geodesic is never longer than the 4-connected path",
          worse == 0, f"{worse} maps where it was longer")
    check("and it is strictly shorter on real maps",
          float(np.median(ratios)) > 1.05,
          f"solve()/geodesic median {np.median(ratios):.3f}x, "
          f"max {max(ratios):.3f}x over {len(ratios)} maps")

    # The clamp must stay gone. It is the reason the bug survived.
    src = (LAB / "evaluate.py").read_text()
    check("evaluate.py does not clamp efficiency to 1.0",
          "min(info[\"route\"] / tg, 1.0)" not in src
          and "1.0)" not in src.split("effs.append")[1].split("\n")[0],
          "an impossible efficiency must be visible, not rewritten")
    check("and it scores against the geodesic, not the BFS route",
          "env.geodesic_len / tg" in src)


def test_run_records_its_task() -> None:
    """
    A run must record the TASK it was trained on, and the renderer must
    read it rather than inventing one.

    `goal_range` caps how far B may be placed. It is part of the task,
    not a training detail -- and it was not written to `summary.json`,
    so nothing downstream could recover it. `render.py` therefore built
    `NavWorld(mode=..., flat=False, seed=0)` with no range at all and
    filmed goals at the map's own endpoints: a harder task than any
    policy here was trained on.

    The symptom was a reader's question. The animations showed the
    robot stranded 14 m from B while the published table said 42%
    arrival, and the two were irreconcilable because they were
    measurements of different worlds. A figure and a number that
    disagree are not a rendering quirk.
    """
    import json

    src = (LAB / "train_nav.py").read_text()
    check("train_nav.py records goal_range in summary.json",
          '"goal_range": a.goal_range' in src)

    rsrc = (LAB / "render.py").read_text()
    check("render.py reads goal_range from the run's own summary",
          'goal_range=meta.get("goal_range")' in rsrc,
          "not hardcoded, and above all not omitted")
    check("render.py does not build NavWorld without a goal range",
          'NavWorld(mode=mode, flat=False, seed=0)' not in rsrc)

    runs = sorted((LAB / "runs").glob("nav*_*_s*/summary.json"))
    missing = [p.parent.name for p in runs
               if "goal_range" not in json.loads(p.read_text())]
    # Only the CURRENT sweep is required to carry it; the superseded
    # broken-frame runs are kept as a record and are not re-scored.
    cur = [m for m in missing if m.startswith("nav5_")]
    check("every corrected-frame run records its goal range",
          not cur, f"{len(cur)} missing" if cur else
          f"{len([p for p in runs if p.parent.name.startswith('nav5_')])} runs")


def test_no_orphan_assets() -> None:
    """
    Every image the pages show must be produced by a shipped command.

    `nav-route-b.gif` was referenced by the tutorial page while nothing
    in `render.py` generated it -- rendered once by hand, then
    orphaned. When the lab was re-rendered after the coordinate-frame
    fix, that one file silently survived from the broken world and sat
    on the page beside eight corrected clips, indistinguishable from
    them.

    An orphan cannot be regenerated, so it cannot be corrected, so it
    is a claim nobody can check. This asks the question the other way
    round from the usual "does the file exist": it asks whether the
    repository can still PRODUCE it.
    """
    import re

    pages = [LAB / "README.md",
             REPO / "docusaurus-docs" / "docs" / "tutorials" / "physical"
             / "terrain-navigation.md"]
    gen = ((LAB / "render.py").read_text()
           + (LAB / "make_figures.py").read_text())

    referenced: set[str] = set()
    for p in pages:
        if p.exists():
            for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", p.read_text()):
                name = m.rsplit("/", 1)[-1]
                if name.startswith("nav-"):
                    referenced.add(name)

    orphans = sorted(n for n in referenced if n not in gen)
    check("every nav-* image on the pages is generated by a script",
          not orphans,
          f"orphaned: {', '.join(orphans)}" if orphans
          else f"{len(referenced)} images, all reproducible")


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
    test_frames_agree(NavWorld)
    test_efficiency_denominator(world)
    test_run_records_its_task()
    test_no_orphan_assets()

    print()
    if _fails:
        print(f"  {len(_fails)} FAILED: {', '.join(_fails)}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
