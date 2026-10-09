#!/usr/bin/env python3
"""
The robot, and the one simplification that makes this lab possible.

    uv run robot.py            # build it, print the joints, drop it on a map

WHAT CHANGED FROM LAB 3
-----------------------
Lab 3's biped was PLANAR. Its base had two joints -- `rootx` and
`rootz` -- so it could move forward and bob up and down, and nothing
else. It could not step sideways and it could not turn. That is fine
for a corridor and useless for navigation.

Here the base gains two more:

    rootx, rooty   translation in the plane
    rootyaw        rotation about vertical -- the robot can TURN
    rootz          vertical

THE SIMPLIFICATION, STATED PLAINLY
----------------------------------
**Torso pitch and roll stay locked.** The robot cannot tip forward or
sideways; it can only sink.

That is not realism and it is not hidden. Lab 1 measured what happens
without it: free the torso and you have the bipedal-balance problem,
"a famously hard benchmark and not what the lab is teaching" -- a
1M-step run produced something that dived forward and fell at one
second. Lab 2 then measured the same switch properly and found that
with two legs, freeing the torso costs a lot of return and buys no
extra climbing.

So the lock is inherited from two labs' worth of evidence rather than
chosen for convenience. What it buys is that the hard part of THIS lab
is navigation -- telling a climbable ridge from an impassable one and
routing accordingly -- instead of balance, which has already been
studied to death elsewhere and would consume the entire compute budget.

What it costs is stated on the page: this robot cannot fall over, so
nothing here measures recovery, and the gaits it finds are not ones a
real biped could use unmodified.
"""

from __future__ import annotations

import numpy as np

from world import ARENA, CELL, MAX_H, N

# Link geometry, carried over from lab 3 unchanged so the two robots are
# comparable and the gait findings there still apply.
THIGH, SHIN = 0.40, 0.36
FOOT_LEN = 0.22
HIP_Y = 0.07                       # lateral offset of each leg
START_Z = 1.00


def model_xml(hfield_file: str, start: tuple[float, float],
              goal: tuple[float, float]) -> str:
    """
    One MuJoCo model: a height-field floor and a two-legged robot on it.

    The height field is loaded from a PNG written by `hfield_png()`.
    MuJoCo scales it so white = `MAX_H` metres, which is why the arena
    and the elevation ceiling are shared constants rather than numbers
    typed in two places.
    """
    sx, sy = start
    gx, gy = goal
    return f"""
<mujoco model="terrain_nav">
  <compiler angle="degree"/>
  <option integrator="RK4" timestep="0.002"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <!-- MuJoCo lights a default HEADLIGHT from the camera. Combined
         with the scene lights it blew the height field out to near
         white and desaturated the route markers until they were
         unreadable. Dimmed to a faint ambient fill. -->
    <headlight ambient="0.22 0.23 0.26" diffuse="0.18 0.18 0.20"
               specular="0 0 0"/>
    <map shadowclip="8"/>
  </visual>
  <asset>
    <hfield name="terrain" file="{hfield_file}"
            size="{ARENA/2} {ARENA/2} {MAX_H} 0.1"/>
    <material name="ground" rgba="0.28 0.32 0.39 1"/>
    <material name="shell"  rgba="0.86 0.86 0.84 1"/>
    <material name="accent" rgba="0.72 0.20 0.17 1"/>
    <material name="limb"   rgba="0.42 0.62 0.82 1"/>
    <material name="flag"   rgba="0.36 0.78 0.52 1"/>
  </asset>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom condim="3" contype="1" conaffinity="1" friction="0.9 0.1 0.1"
          material="limb" solimp="0.95 0.95 0.01" solref="0.01 1"/>
  </default>
  <worldbody>
    <light pos="0 0 14" dir="0 0 -1" directional="true"
           diffuse="0.68 0.70 0.76" specular="0.04 0.04 0.04"
           castshadow="true"/>
    <light pos="-7 -7 9" dir="0.5 0.5 -1" directional="true"
           diffuse="0.18 0.20 0.26" specular="0 0 0" castshadow="false"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"
          material="ground"/>

    <!-- The goal, drawn so an animation can show what the robot is
         aiming at. It does not collide; a target you can trip over is
         a different experiment. -->
    <body name="goalpost" pos="{gx} {gy} 0">
      <geom name="goal_pole" type="cylinder" size="0.05 0.9" pos="0 0 0.9"
            material="flag" contype="0" conaffinity="0"/>
      <geom name="goal_flag" type="box" size="0.02 0.28 0.18"
            pos="0 0.28 1.55" material="flag" contype="0" conaffinity="0"/>
    </body>

    <!-- The torso is declared at the ORIGIN, not at the start point.
         `rootx`/`rooty` are SLIDE joints, so a body declared at
         pos="(sx, sy)" ends up at (sx, sy) + qpos -- and reset() also
         writes the start into qpos, which placed the robot at TWICE
         its start coordinates. Every distance, every terrain lookup
         and the arrival test were then computed in a frame shifted by
         the start offset: to_goal() read 10.44 m where the true
         distance was 14.78 m, and the privileged observation sampled
         ground the robot was not standing on.

         Declared at the origin, qpos[0:2] IS the world position and
         the two agree by construction. -->
    <body name="torso" pos="0 0 {START_Z}">
      <joint name="rootx"   type="slide" axis="1 0 0" limited="false"
             armature="0" damping="0"/>
      <joint name="rooty"   type="slide" axis="0 1 0" limited="false"
             armature="0" damping="0"/>
      <joint name="rootz"   type="slide" axis="0 0 1" limited="false"
             armature="0" damping="0"/>
      <!-- Yaw is the new degree of freedom. Pitch and roll are absent
           ON PURPOSE; see this module's docstring. -->
      <joint name="rootyaw" type="hinge" axis="0 0 1" limited="false"
             armature="0" damping="0.5"/>
      <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.35"
            size="0.06"/>

      <body name="head" pos="0 0 0.40">
        <inertial pos="0 0 0" mass="2.0" diaginertia="0.02 0.02 0.02"/>
        <geom type="box" pos="-0.05 0 0.10" euler="0 -9 0"
              size="0.13 0.20 0.13" material="shell"
              density="0" contype="0" conaffinity="0"/>
        <geom type="box" pos="0.055 0 0.155" euler="0 -9 0"
              size="0.02 0.201 0.02" material="accent"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.03 0.13 0.10 0.075 0.13 0.095"
              size="0.055" rgba="0.13 0.26 0.52 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.03 -0.13 0.10 0.075 -0.13 0.095"
              size="0.055" rgba="0.58 0.60 0.63 1"
              density="0" contype="0" conaffinity="0"/>
        <!-- Forward and DOWN, as in lab 3: the useful thing to look at
             is the ground you are about to step on. -->
        <camera name="eye" pos="0.09 0 0.10" xyaxes="0 -1 0 0.55 0 0.84"/>
      </body>

      {_leg("L", +HIP_Y)}
      {_leg("R", -HIP_Y)}
    </body>
  </worldbody>
  <actuator>
    {_motors()}
  </actuator>
</mujoco>
"""


def _leg(tag: str, y: float) -> str:
    return f"""
      <body name="thigh{tag}" pos="0 {y} 0">
        <joint name="hip{tag}" type="hinge" axis="0 -1 0" range="-150 20"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{THIGH}" size="0.045"/>
        <body name="shin{tag}" pos="0 0 -{THIGH}">
          <joint name="knee{tag}" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{SHIN}" size="0.04"/>
          <body name="foot{tag}" pos="0 0 -{SHIN}">
            <joint name="ankle{tag}" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot{tag}_geom" type="capsule"
                  fromto="-0.05 0 0 {FOOT_LEN - 0.05} 0 0" size="0.045"/>
          </body>
        </body>
      </body>"""


# The standing pose, in radians, measured in nav_env. Position
# actuators hold THIS when the policy outputs zero.
STAND = {"hip": -0.30, "knee": -0.54, "ankle": 0.15}


def _motors() -> str:
    """
    POSITION actuators, not torque motors.

    Labs 1-3 used torque control, which is fine in a corridor where the
    robot only has to fall forwards usefully. It does not survive the
    jump to navigation: this body needs a specific sustained torque
    pattern just to stand (measured: hip 0.6 / knee 1.0 holds, anything
    less collapses in under 50 steps), and random exploration almost
    never produces it. A 1M-step run on FLAT ground reached 0% arrival
    because the policy spent its whole budget failing to stand up.

    With position control the policy commands joint TARGETS and a PD
    loop supplies the torque, so a zero action means "hold the standing
    pose". Standing becomes the default rather than something to be
    discovered, and the policy learns deviations from it. This is what
    modern legged-locomotion work does, for exactly this reason.
    """
    out = []
    for tag in ("L", "R"):
        for j, kp, rng in (("hip", 260, "-2.6 0.35"),
                           ("knee", 260, "-2.6 0.0"),
                           ("ankle", 110, "-0.78 0.78")):
            out.append(f'<position joint="{j}{tag}" kp="{kp}" '
                       f'ctrlrange="{rng}"/>')
    # Yaw is actuated directly rather than emerging from foot placement.
    # A planar biped with locked roll cannot generate yaw torque through
    # contact in any realistic way, so pretending otherwise would just
    # make turning unlearnable. Stated on the page as a limitation.
    out.append('<motor joint="rootyaw" ctrlrange="-1 1" gear="45"/>')
    return "\n    ".join(out)


def hfield_png(heights: np.ndarray, path) -> None:
    """
    Write the elevation array as the 8-bit PNG MuJoCo's hfield reads.

    Quantisation matters and is worth stating: 256 levels over MAX_H =
    1.2 m is 4.7 mm per level, which is fine next to a 0.12 m step
    threshold but would not be if the threshold were millimetric.
    """
    from PIL import Image

    img = np.clip(heights / MAX_H, 0.0, 1.0)
    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)


def main() -> int:
    import tempfile
    from pathlib import Path

    import mujoco

    from world import generate

    m = generate(3)
    tmp = Path(tempfile.mkdtemp())
    hfield_png(m.heights, tmp / "terrain.png")
    model = mujoco.MjModel.from_xml_string(
        model_xml("terrain.png", m.start, m.goal),
        {"terrain.png": (tmp / "terrain.png").read_bytes()})
    data = mujoco.MjData(model)

    print("=" * 66)
    print("  The navigating biped")
    print("=" * 66)
    print(f"  nq {model.nq}   nv {model.nv}   actuators {model.nu}")
    J = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        act = any(model.actuator_trnid[k][0] == j for k in range(model.nu))
        print(f"    {n:<9} {J[model.jnt_type[j]]:<6} "
              f"{'actuated' if act else '--'}")

    mujoco.mj_forward(model, data)
    for _ in range(400):
        mujoco.mj_step(model, data)
    print()
    print(f"  dropped on the terrain, settles at z = {data.qpos[2]:.3f} m")
    print(f"  ground under it      = {m.h_at(*m.start):.3f} m")
    print(f"  start {tuple(round(v,1) for v in m.start)} -> "
          f"goal {tuple(round(v,1) for v in m.goal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
