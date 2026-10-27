#!/usr/bin/env python3
"""
Four robots and one staircase, generated from two switches.

THE EXPERIMENT
--------------
Lab 1 built a one-legged hopper with its torso locked upright and said, in
passing, that freeing the torso turns the task into the bipedal-balance
problem. This lab tests that claim against the obvious alternative
explanation -- that difficulty comes from having more joints to coordinate.

Two switches, crossed:

    legs    1 or 2      3 actuators or 6
    torso   locked      cannot rotate; falling over is impossible
            free        can rotate; the robot must balance

Everyone assumes the limb count is the hard part. The measured prediction
from lab 1 is that it is not even close -- a free torso falls over on its
own in under two seconds, and no amount of leg coordination helps until
that is solved.

THE STAIRCASE
-------------
Three rising steps instead of one box: 0.08, 0.16, 0.24 m. A single
obstacle can be cleared with one lucky lunge. A staircase cannot -- the
robot has to do the same thing three times, from three different starting
heights, which is a much poorer fit for a memorised trajectory.

The rise is per-episode randomised the same way lab 1 randomised its
single box, so the policy sees a range rather than one geometry.

ON DUPLICATING LAB 1's CODE
---------------------------
`ppo.py` here is lab 1's, copied rather than imported. That is the
repository's rule and it is deliberate: a reader must be able to open one
folder and run it without the other twenty-three existing. The cost is
real duplication; the benefit is that this folder has no hidden
dependencies on a sibling that may change underneath it.
"""

from __future__ import annotations

import numpy as np

# -- the staircase ----------------------------------------------------------
STAIR_X0 = 1.10           # where the first riser starts
STAIR_DEPTH = 0.40        # tread depth, per step
N_STAIRS = 3
# Per-step rise, randomised each episode. MEASURED, not chosen: lab 1
# established that this robot rests stably on a 0.17 m step and topples off
# 0.20 m, so a three-step climb must stay well under that per step.
RISE_MIN = 0.04
RISE_MAX = 0.10
RISE_DEFAULT = 0.08

STAIR_END_X = STAIR_X0 + N_STAIRS * STAIR_DEPTH     # 2.30


def stair_tops(rise: float) -> list[tuple[float, float, float]]:
    """(x_start, x_end, top_z) for each step. One source of truth."""
    out = []
    for i in range(N_STAIRS):
        x0 = STAIR_X0 + i * STAIR_DEPTH
        out.append((x0, x0 + STAIR_DEPTH, (i + 1) * rise))
    return out


def build_xml(legs: int = 2, locked_torso: bool = False,
              rise: float = RISE_DEFAULT) -> str:
    """
    Emit the MuJoCo model for one cell of the 2x2.

    The two switches change exactly two things and nothing else, which is
    what makes the comparison a comparison: `locked_torso` omits the
    `rooty` hinge, and `legs` emits one chain or two. Masses, gear ratios,
    link lengths, friction and the staircase are identical across all four.
    """
    if legs not in (1, 2):
        raise ValueError(f"legs must be 1 or 2, got {legs}")

    pitch = "" if locked_torso else (
        '<joint name="rooty" type="hinge" axis="0 1 0" limited="false" '
        'armature="0" damping="0"/>')

    def leg(tag: str, y: float) -> str:
        return f"""
      <body name="thigh{tag}" pos="0 {y} 0">
        <joint name="hip{tag}" type="hinge" axis="0 -1 0" range="-150 20"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.045"/>
        <body name="shin{tag}" pos="0 0 -0.40">
          <joint name="knee{tag}" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.04"/>
          <body name="foot{tag}" pos="0 0 -0.36">
            <joint name="ankle{tag}" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot{tag}_geom" type="capsule"
                  fromto="-0.08 0 0 0.14 0 0" size="0.045"/>
          </body>
        </body>
      </body>"""

    # A single leg sits on the centreline; a pair straddles it. Same total
    # actuation per leg either way, so a two-legged robot really does have
    # twice the muscle -- which is part of what the experiment measures.
    tags = [("L", -0.07), ("R", 0.07)] if legs == 2 else [("C", 0.0)]
    bodies = "".join(leg(t, y) for t, y in tags)
    motors = "".join(
        f'<motor joint="hip{t}" ctrlrange="-1 1" gear="120" ctrllimited="true"/>'
        f'<motor joint="knee{t}" ctrlrange="-1 1" gear="120" ctrllimited="true"/>'
        f'<motor joint="ankle{t}" ctrlrange="-1 1" gear="70" ctrllimited="true"/>'
        for t, _ in tags)

    steps = "".join(
        f'<geom name="stair{i}" type="box" '
        f'pos="{(x0 + x1) / 2:.4f} 0 {top / 2:.4f}" '
        f'size="{STAIR_DEPTH / 2:.4f} 0.6 {top / 2:.4f}" '
        f'rgba="{0.88 - 0.05 * i:.2f} {0.58 - 0.05 * i:.2f} 0.25 1"/>'
        for i, (x0, x1, top) in enumerate(stair_tops(rise)))

    return f"""
<mujoco model="biped_stairs">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.002"/>
  <!-- Visual only; MuJoCo does not read these in the dynamics. -->
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" friction="0.9 0.1 0.1"
          rgba="0.42 0.62 0.82 1" solimp="0.95 0.95 0.01" solref="0.01 1"/>
  </default>
  <worldbody>
    <light pos="0 -1.5 4" dir="0 0.3 -1" diffuse="0.85 0.85 0.85"
           specular="0.2 0.2 0.2" castshadow="true"/>
    <light pos="3 2 3" dir="-0.5 -0.4 -1" diffuse="0.45 0.5 0.6"
           castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 0" size="40 2 0.1"
          rgba="0.16 0.20 0.25 1"/>
    {steps}
    <body name="torso" pos="0 0 1.0">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false"
             armature="0" damping="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false"
             armature="0" damping="0"/>
      {pitch}
      <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.35"
            size="0.06"/>{bodies}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>
"""


def obs_dim(legs: int, locked_torso: bool) -> int:
    """
    How many numbers the policy sees, derived rather than hardcoded.

    qpos is [x, z, (pitch), 3 per leg]; qvel matches. The observation drops
    absolute x -- locomotion is translation invariant and feeding a world
    coordinate invites memorisation -- and adds four task terms: distance
    to the staircase, the per-step rise, how many steps are already behind
    the robot, and the height of the next tread.
    """
    nq = 2 + (0 if locked_torso else 1) + 3 * legs
    return (nq - 1) + nq + 4


def act_dim(legs: int) -> int:
    return 3 * legs


CELLS = [
    ("1leg_locked", 1, True), ("1leg_free", 1, False),
    ("2leg_locked", 2, True), ("2leg_free", 2, False),
]


def main() -> None:
    """Show all four cells build, and how big each one is."""
    import mujoco

    print("=" * 74)
    print("  The 2x2: legs x torso, on an identical staircase")
    print("=" * 74)
    print(f"  staircase: {N_STAIRS} steps of {STAIR_DEPTH} m depth, rise "
          f"~U({RISE_MIN}, {RISE_MAX}) m, ending at x={STAIR_END_X:.2f}")
    print()
    print(f"  {'cell':<14}{'nq':>4}{'nv':>4}{'actuators':>11}"
          f"{'obs':>6}   passive survival")
    for name, legs, locked in CELLS:
        m = mujoco.MjModel.from_xml_string(build_xml(legs, locked))
        d = mujoco.MjData(m)
        fell = None
        for t in range(3000):
            mujoco.mj_step(m, d)
            if 1.0 + d.qpos[1] < 0.55 or (
                    not locked and abs(d.qpos[2]) > 1.2):
                fell = t * 0.002
                break
        verdict = f"fell at {fell:.1f}s" if fell else "upright at 6.0s"
        print(f"  {name:<14}{m.nq:>4}{m.nv:>4}{m.nu:>11}"
              f"{obs_dim(legs, locked):>6}   {verdict}")
    print()
    print("  Passive survival is the prediction: a free torso falls over on")
    print("  its own. Leg count does not change that, which is the whole")
    print("  question this lab measures.")


if __name__ == "__main__":
    main()
