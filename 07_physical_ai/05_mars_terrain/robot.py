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

import os

import math
import numpy as np

from mars import preset

# Geometry comes from the Mars generator's rover preset, not from a
# second set of constants. Lab 3 shipped a hand-typed copy of a
# quantity that lived somewhere else, it went stale, and three runs
# died on a shape mismatch while the arms that never touched it
# sailed through.
_P = preset(os.environ.get("MARS_POOL_SCALE", "course"))
ARENA, N = _P["extent"], _P["cells"]
CELL = ARENA / N

# Link geometry, carried over from lab 3 unchanged so the two robots are
# comparable and the gait findings there still apply.
THIGH, SHIN = 0.40, 0.36
FOOT_LEN = 0.22
HIP_Y = 0.07                       # lateral offset of each leg

# ---------------------------------------------------------------------------
# LEG LAYOUTS. (tag, x, y) per leg, in torso-local metres.
#
# ADDED, NEVER SUBSTITUTED: `biped` is byte-identical to what labs 5-7
# measured on, so their published numbers keep describing the same
# robot. The extra layouts exist because three separate labs reached
# the same wall from three directions -- lab 5 part 4 (0% on scree),
# lab 7 (perception cannot help), lab 8 (the traverse stalls) -- and
# each time the body was the answer. Lab 5 already measured it once:
# freeing torso roll and adding hip abduction took flat ground from
# 67-92% WITH falls to 92% with ZERO falls.
#
# The physical argument for more legs is static stability. A biped is
# statically UNSTABLE -- it stays upright only by actively catching
# itself, which is exactly what fails when the ground tilts in two
# directions at once. A quadruped standing on three feet is stable
# without acting; a hexapod is stable on a tripod while three legs
# swing. That is why rough-terrain legged machines have four or six
# legs and not two.
#
# `x` spreads legs fore/aft, which is what actually buys the support
# polygon -- two legs at the same x would be a wider biped, not a
# quadruped.
# `legs` is (tag, x, y); `leg_scale` multiplies THIGH and SHIN.
#
# THE STANCE IS THE POINT, AND THE FIRST VERSION GOT IT WRONG. It
# kept the biped's 0.76 m legs and put four of them on a 0.22 m wide
# base -- a height-to-width ratio of 3.5, against 1.4 for ANYmal and
# 2.0 for Spot. That is not a quadruped, it is a biped on four
# stilts: still tippy, and it threw away the static stability that
# was the entire argument for adding legs. Measured, it was WORSE
# than the biped (83.3% falls vs 58.3%).
#
# Static stability comes from a support polygon that is wide and low
# relative to the centre of mass, so the extra legs come with shorter
# legs and a wider track.
# MAMMAL vs SPRAWL, and the difference is the whole point.
#
# A mammal leg hangs straight DOWN from the torso, so the support
# polygon is only as wide as the hips. Adding legs that way gives a
# biped on stilts -- which is literally what the first attempt built,
# and it fell MORE than the biped (83.3% vs 58.3%).
#
# A sprawling leg reaches OUTWARD first and only then down, so the
# feet plant well outside the body and the body hangs low between
# them. That is what a spider is, and it is why spiders do not fall
# over. Support polygon becomes 2*(r_hip + femur) across while the
# body sits at roughly tibia height.
#
#   kind     legs   foot circle   body height   height/width
#   biped      2       0.14 m        0.71 m         5.4
#   mammal4    4       0.34 m        0.41 m         1.2
#   spider6    6       0.76 m        0.26 m         0.34
#   spider8    8       0.76 m        0.26 m         0.34
#
# `legs` for a sprawl is (tag, azimuth_degrees); the leg is placed on
# a circle and rotated to face outward, so one leg description serves
# every azimuth.
COXA, FEMUR, TIBIA = 0.12, 0.26, 0.26


def _ring(prefix: str, n: int, start_deg: float = 0.0):
    """n legs evenly spaced around the torso."""
    return [(f"{prefix}{i}", start_deg + 360.0 * i / n) for i in range(n)]


LEG_LAYOUTS = {
    "biped": dict(
        kind="mammal",
        legs=[("L", 0.00, +HIP_Y), ("R", 0.00, -HIP_Y)],
        leg_scale=1.0),          # untouched: labs 5-7 measured on this
    "quadruped": dict(
        kind="mammal",
        legs=[("FL", +0.26, +0.17), ("FR", +0.26, -0.17),
              ("HL", -0.26, +0.17), ("HR", -0.26, -0.17)],
        leg_scale=0.58),         # 0.44 m legs on 0.52 x 0.34 m -> 1.3
    "hexapod": dict(
        kind="mammal",
        legs=[("FL", +0.30, +0.18), ("FR", +0.30, -0.18),
              ("ML",  0.00, +0.21), ("MR",  0.00, -0.21),
              ("HL", -0.30, +0.18), ("HR", -0.30, -0.18)],
        leg_scale=0.52),
    # THE SPIDERS. Legs splay outward; the body rides low.
    "spider6": dict(kind="sprawl", legs=_ring("S", 6, 30.0), leg_scale=1.0),
    "spider8": dict(kind="sprawl", legs=_ring("S", 8, 22.5), leg_scale=1.0),
}


# Standing clearance as a FRACTION of leg length, and the fall
# threshold as a fraction of that. Both were biped constants
# (START_DZ -0.294 -> clearance 0.706, FALL_CLEARANCE 0.40) and both
# silently assumed 0.76 m legs.
#
# Giving the quadruped shorter legs while holding the torso at the
# biped's height left its feet hanging ~0.3 m off the ground: the
# robot was dropped every reset and the measured fall rate went UP
# (15% -> 25% -> 35% doing nothing). A constant calibrated for one
# body is not a constant, it is a hidden parameter of that body.
#
# Derived so the biped reproduces its own numbers exactly:
#   0.706 / 0.76 = 0.929      standing clearance / leg length
#   0.40  / 0.706 = 0.567     fall threshold / standing clearance
STAND_FRAC = 0.706 / (0.40 + 0.36)
FALL_FRAC = 0.40 / 0.706


def leg_length(name: str) -> float:
    """
    The length that sets STANDING HEIGHT, in metres.

    For a mammal leg that is the whole limb, because it hangs down.
    For a sprawl it is the TIBIA only: the femur reaches sideways and
    contributes width, not height. Using total limb length for a
    spider would stand it twice as tall as it can reach, which is the
    same dangling-feet bug that made the first quadruped worse.
    """
    d = LEG_LAYOUTS[name]
    if d.get("kind") == "sprawl":
        return TIBIA / STAND_FRAC        # so stand_clearance == TIBIA
    return (THIGH + SHIN) * float(d["leg_scale"])


def stand_pose(name: str) -> dict:
    """
    Joint targets a body holds when the policy outputs zero.

    Per-morphology because a sprawl's neutral is not a mammal's. The
    biped crouches (hip -0.30, knee -0.54); a spider stands with the
    femur horizontal and the tibia vertical, which is all zeros. Lab
    5's `reset` hardcoded the crouch, so a spider given that pose
    folds its legs under itself and collapses on frame one.
    """
    if LEG_LAYOUTS[name].get("kind") == "sprawl":
        return {"hiproll": 0.0, "hip": 0.0, "knee": 0.0, "ankle": 0.0}
    b = 0.30
    return {"hiproll": 0.0, "hip": -b, "knee": -1.8 * b, "ankle": 0.5 * b}


def layout_kind(name: str) -> str:
    return LEG_LAYOUTS[name].get("kind", "mammal")


def stand_clearance(name: str) -> float:
    """Torso clearance this body stands at. Biped: 0.706 m."""
    return STAND_FRAC * leg_length(name)


def fall_clearance(name: str) -> float:
    """Below this, the body has collapsed. Biped: 0.400 m."""
    return FALL_FRAC * stand_clearance(name)


def leg_layout(name: str):
    if name not in LEG_LAYOUTS:
        raise SystemExit(f"unknown morphology {name!r}; "
                         f"have {sorted(LEG_LAYOUTS)}")
    d = LEG_LAYOUTS[name]
    return d["legs"], float(d["leg_scale"])
START_Z = 1.00


def model_xml(hfield_file: str, start: tuple[float, float],
              goal: tuple[float, float], relief: float = 3.0,
              rocks=(), morphology: str = "biped") -> str:
    """
    One MuJoCo model: a height-field floor and a legged robot on it.

    `morphology` selects a layout from `LEG_LAYOUTS` and DEFAULTS TO
    THE BIPED, so every caller written before this parameter existed
    -- all of labs 5, 6 and 7 -- builds exactly the robot they
    measured. Nothing about their numbers changes.

    The height field is loaded from a PNG written by `hfield_png()`.
    `relief` is the vertical scale MuJoCo multiplies the PNG by, and
    it is passed IN rather than being a constant: a Mars surface's
    relief varies per seed, and baking one number here would stretch
    or squash every map that did not happen to match it.

    MuJoCo scales it so white = `relief` metres, which is why the arena
    and the elevation ceiling are shared constants rather than numbers
    typed in two places.
    """
    sx, sy = start
    gx, gy = goal
    # Rocks as tilted boxes. See `mars.Rock`: platy, not spherical.
    rock_xml = "\n      ".join(
        f'<geom type="box" size="{k.rx:.4f} {k.ry:.4f} {k.rz:.4f}" '
        f'pos="{k.x:.4f} {k.y:.4f} {k.z:.4f}" '
        f'euler="{k.roll:.4f} {k.pitch:.4f} {k.yaw:.4f}" '
        f'material="rock" condim="3" friction="0.95 0.1 0.1"/>'
        for k in rocks)
    legs, leg_k = leg_layout(morphology)
    kind = layout_kind(morphology)
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
            size="{ARENA/2} {ARENA/2} {relief} 0.12"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.62 0.44 0.32" rgb2="0.86 0.67 0.49"
             width="512" height="512"/>
    <material name="ground" rgba="0.76 0.44 0.26 1"/>
    <material name="rock"   rgba="0.42 0.34 0.29 1"/>
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
    <light pos="0 0 14" dir="0.45 0.3 -0.84" directional="true"
           diffuse="1.02 0.84 0.68" specular="0.05 0.04 0.03"
           castshadow="true"/>
    <light pos="-7 -7 9" dir="0.5 0.5 -1" directional="true"
           diffuse="0.20 0.16 0.14" specular="0 0 0" castshadow="false"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 0"
          material="ground"/>
    <!-- Boulders, and unlike in Part 1 these COLLIDE.
         In the terrain-only half the rocks were decoration
         (contype="0"), because nothing was walking. Here they are
         obstacles the robot can trip on, stub a foot against or step
         over -- which is the point of scattering them. A rock you can
         walk through is scenery wearing an obstacle's shape. -->
    <body name="rocks" pos="0 0 0">
      {rock_xml}
    </body>

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
      <joint name="rootyaw" type="hinge" axis="0 0 1" limited="false"
             armature="0" damping="0.5"/>
      <!-- PITCH AND ROLL ARE FREE HERE, unlike labs 1-4.
           
           Those labs locked them deliberately: freeing the torso gives
           you the bipedal-balance problem on top of whatever else the
           lab is about, and with two legs it cost return and bought no
           climbing. That reasoning held because their ground was FLAT
           with discrete obstacles standing on it.
           
           Mars is not like that. Its surface is continuously uneven --
           mean slope 10.4 degrees everywhere, not just at the
           obstacles. A torso that cannot pitch or roll cannot put both
           feet on ground that is tilted in two directions at once, and
           the measurement is unambiguous: the same robot walks 9.8 m
           on a plane, 5.3 m at 1.3 degrees of mean slope, and 0.6 m at
           2.6 degrees. It stops dead long before it reaches anything
           Mars would call a hill.
           
           So the two joints come back, and Part 2 is a harder problem
           than lab 4 by design: balance AND navigation, on ground that
           demands both. Damped, because an undamped free torso on a
           12k-parameter policy is a tumbling simulation. -->
      <joint name="rootpitch" type="hinge" axis="0 1 0" limited="true"
             range="-0.9 0.9" armature="0.02" damping="1.2"/>
      <joint name="rootroll"  type="hinge" axis="1 0 0" limited="true"
             range="-0.9 0.9" armature="0.02" damping="1.2"/>
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

      {_chassis(morphology)}
      {"".join(_sprawl_leg(t, az) for t, az in legs) if kind == "sprawl"
        else "".join(_leg(t, yy, xx, leg_k) for t, xx, yy in legs)}
    </body>
  </worldbody>
  <actuator>
    {_motors(legs, kind)}
  </actuator>
</mujoco>
"""


# Structural frame density, kg/m^3. NOT the 1000 the solid limb
# capsules use: a chassis is a frame with electronics and air inside,
# not a billet. 300 puts the hexapod's frame at ~2 kg against its
# ~4.9 kg torso, which is the right order for a machine this size.
CHASSIS_DENSITY = 300.0


def _chassis(morphology: str) -> str:
    """
    Geometry spanning the hips, which did not exist until it was
    noticed missing.

    THE BUG THIS FIXES. The torso is a capsule of radius 0.06 m on the
    centreline, sized for a biped whose hips sit 0.07 m out. The
    hexapod's hips sit at 0.30 m, so nothing at all spanned the gap:
    the legs looked detached because they WERE geometrically
    detached. MuJoCo still attaches child bodies to the parent body
    frame, so the physics was a valid rigid structure and the trained
    policies were correct for the model -- but that model gave the
    multi-legged bodies their limbs for free:

      * no chassis MASS      (a hexapod weighed what a biped weighs)
      * no chassis INERTIA   (Izz identical across all five bodies)
      * no chassis COLLISION (terrain passed between the legs)

    Every one of those errors flatters the multi-legged result, which
    is the direction a measurement must never be wrong in.

    The biped gets NO chassis and is unchanged: its hips are already
    inside the torso capsule, so adding one would alter the body
    labs 5-7 measured on.
    """
    legs, _k = leg_layout(morphology)
    kind = layout_kind(morphology)
    if morphology == "biped":
        return ""
    if kind == "sprawl":
        r = COXA + 0.05
        return (f'<geom name="chassis" type="cylinder" '
                f'pos="0 0 0.05" size="{r:.3f} 0.02" '
                f'density="{CHASSIS_DENSITY}"/>')
    hx = max(abs(x) for _t, x, _y in legs) + 0.05
    hy = max(abs(y) for _t, _x, y in legs) + 0.05
    return (f'<geom name="chassis" type="box" pos="0 0 0.05" '
            f'size="{hx:.3f} {hy:.3f} 0.02" '
            f'density="{CHASSIS_DENSITY}"/>')


def _sprawl_leg(tag: str, az_deg: float) -> str:
    """
    One spider leg: coxa swings horizontally, femur reaches OUT,
    tibia drops DOWN.

    The body is placed on a circle at `az_deg` and rotated to match,
    so the same description serves every azimuth and the joint axes
    stay leg-local. Joint NAME PREFIXES are kept identical to the
    mammal leg (hiproll / hip / knee) because the environment
    resolves joints, actuators and the standing pose by name -- a new
    body needs no plumbing as long as it speaks the same names.
    """
    cx = COXA * math.cos(math.radians(az_deg))
    cy = COXA * math.sin(math.radians(az_deg))
    return f"""
      <body name="thigh{tag}" pos="{cx:.4f} {cy:.4f} 0" euler="0 0 {az_deg}">
        <joint name="hiproll{tag}" type="hinge" axis="0 0 1" range="-40 40"
               armature="0.4" damping="0.6"/>
        <joint name="hip{tag}" type="hinge" axis="0 -1 0" range="-55 55"
               armature="0.4" damping="0.6"/>
        <geom type="capsule" fromto="0 0 0 {FEMUR} 0 0" size="0.035"/>
        <body name="shin{tag}" pos="{FEMUR} 0 0">
          <joint name="knee{tag}" type="hinge" axis="0 -1 0" range="-20 100"
                 armature="0.4" damping="0.6"/>
          <geom name="tibia{tag}" type="capsule"
                fromto="0 0 0 0 0 -{TIBIA}" size="0.032"/>
          <!-- An explicit FOOT at the tip. The shin capsule's
               `geom_xpos` is its CENTRE, so measuring torso-to-foot
               against it reads half a tibia short -- which put every
               spider permanently below the fall threshold (100%
               "falls" while standing still). A tip sphere puts the
               measured foot where the foot actually is. -->
          <geom name="foot{tag}_geom" type="sphere"
                pos="0 0 -{TIBIA}" size="0.034"/>
        </body>
      </body>"""


def _leg(tag: str, y: float, x: float = 0.0, k: float = 1.0) -> str:
    """
    One leg, and it can now move SIDEWAYS.

    Every joint in labs 1-4 used axis="0 -1 0" -- hip, knee and ankle
    all rotating about the pitch axis, so the whole leg lived in the
    sagittal plane. On a corridor that is fine; the robot only ever
    has to fall forwards usefully.

    On ground tilted in two directions it is fatal, and it explains
    the central failure of Part 4. The torso was freed to roll, so the
    body leaned sideways on every slope -- and the legs had no way to
    answer, because not one of their six joints could place a foot
    laterally. The policy fell on 50-92% of episodes across 10M steps
    and never improved, which is not a compute shortage: it is a robot
    that cannot take a sideways step being asked to walk on scree.

    `hiproll` is an abduction joint: it swings the whole leg out from
    the body. Two extra joints, and the difference between a machine
    that can catch itself and one that can only topple.
    """
    return f"""
      <body name="thigh{tag}" pos="{x} {y} 0">
        <joint name="hiproll{tag}" type="hinge" axis="1 0 0" range="-35 35"
               armature="0.6" damping="0.8"/>
        <joint name="hip{tag}" type="hinge" axis="0 -1 0" range="-150 20"/>
        <geom type="capsule" fromto="0 0 0 0 0 -{THIGH*k}" size="0.045"/>
        <body name="shin{tag}" pos="0 0 -{THIGH*k}">
          <joint name="knee{tag}" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -{SHIN*k}" size="0.04"/>
          <body name="foot{tag}" pos="0 0 -{SHIN*k}">
            <joint name="ankle{tag}" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="foot{tag}_geom" type="capsule"
                  fromto="-0.05 0 0 {FOOT_LEN - 0.05} 0 0" size="0.045"/>
          </body>
        </body>
      </body>"""


# The standing pose, in radians, measured in nav_env. Position
# actuators hold THIS when the policy outputs zero.
STAND = {"hiproll": 0.0, "hip": -0.30, "knee": -0.54,
         "ankle": 0.15}


def _motors(legs, kind: str = "mammal") -> str:
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
    if kind == "sprawl":
        # Three joints per leg, and a spider has many legs: eight legs
        # x three is 24 actuators before yaw. Gains are lower than the
        # biped's because each leg carries an eighth of the weight,
        # not a half.
        for tag, _az in legs:
            for j, kp, rng in (("hiproll", 90, "-0.70 0.70"),
                               ("hip", 140, "-0.96 0.96"),
                               ("knee", 140, "-0.35 1.75")):
                out.append(f'<position joint="{j}{tag}" kp="{kp}" '
                           f'ctrlrange="{rng}"/>')
        out.append('<motor joint="rootyaw" ctrlrange="-1 1" gear="45"/>')
        return "\n    ".join(out)
    for tag, _x, _y in legs:
        for j, kp, rng in (("hiproll", 150, "-0.61 0.61"),
                           ("hip", 260, "-2.6 0.35"),
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

    Normalised to the array's OWN range, and the caller must pass the
    same range to `model_xml(relief=...)`. MuJoCo maps PNG value 0 to
    z=0 and 255 to z=relief, so the two have to agree or every height
    in the world is scaled by the ratio between them.

    The caller must also have SHIFTED the array so its minimum is
    zero. If it has not, MuJoCo puts the lowest point of the terrain
    at z=0 while the environment's own `_h_at` still returns the
    unshifted value, and the robot is spawned floating (or buried) by
    exactly that offset -- the same class of frame disagreement that
    invalidated lab 4's published result, in a different coordinate.

    Quantisation: 256 levels over a 3 m relief is 1.2 cm per level,
    which is coarse next to a 0.12 m step threshold but not
    dangerously so; over a 1 m relief it is 4 mm.
    """
    from PIL import Image

    h = np.asarray(heights, dtype=np.float64)
    span = float(h.max() - h.min())
    img = np.zeros_like(h) if span < 1e-9 else (h - h.min()) / span
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
