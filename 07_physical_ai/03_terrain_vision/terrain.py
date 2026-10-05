"""The terrains, on a raised plateau so DOWN is expressible as well as UP.

The approach slab is identical in every one of them, which is the whole
design: a robot standing on it cannot tell from its own joints what
comes next.

`build()` can make four terrains; `KINDS` names the three the lab
actually trains and measures on. See the note above KINDS for why `hop`
is built but excluded -- it is a measured result, not a preference.
"""
PLATEAU_Z = 0.45          # the robot starts up here; descent is now expressible
START_X   = 0.0
EVENT_X   = 1.20          # where the terrain changes
# Long enough that a 600-step episode cannot reach the end. It was 2.40,
# which put the edge of the world at x=3.60 -- and a 6-second episode at
# ~1 m/s travels further than that. Robots were CROSSING the gap and then
# walking off the end of the terrain, which scored as a fall. Every
# measurement taken before this fix is contaminated by it.
RUN       = 9.00

def build(kind: str, rise: float = 0.09, gap: float = 0.45) -> str:
    """kind in {flat, up, down, hop}. Everything before EVENT_X is identical."""
    P, E, R = PLATEAU_Z, EVENT_X, RUN
    # approach slab: the robot always starts on this, identical in all four
    g = [f'<geom name="approach" type="box" pos="{(E-2.0)/2} 0 {P/2}" '
         f'size="{(E+2.0)/2} 0.8 {P/2}" rgba="0.30 0.34 0.40 1"/>']
    # NO overhead barrier, deliberately. One was tried here, to force a
    # crouch on `flat` and `down` while `up` kept the tall gait, so that
    # a single gait could not clear everything. Swept rather than
    # reasoned: 1.40 m changed nothing, 1.30 broke one terrain, 1.20
    # broke both. Every height either did nothing or made a terrain
    # impossible for BOTH arms, which is a broken terrain rather than a
    # discrimination. The asymmetry the lab needed turned out to be
    # `down` -- you cannot feel for a descent -- and needed no ceiling
    # at all.
    if kind == "flat":
        g.append(f'<geom name="run" type="box" pos="{E+R/2} 0 {P/2}" '
                 f'size="{R/2} 0.8 {P/2}" rgba="0.32 0.36 0.42 1"/>')
    elif kind == "up":
        for i in range(3):
            top = P + (i + 1) * rise
            cx = E + TREAD_GAP + TREAD / 2 + i * PITCH
            g.append(f'<geom name="s{i}" type="box" pos="{cx} 0 {top/2}" '
                     f'size="{TREAD/2} 0.8 {top/2}" rgba="0.32 0.36 0.42 1"/>')
        g.append(f'<geom name="run" type="box" pos="{E+3*PITCH+TREAD_GAP+R/2} 0 '
                 f'{(P+3*rise)/2}" size="{R/2} 0.8 {(P+3*rise)/2}" '
                 f'rgba="0.32 0.36 0.42 1"/>')
    elif kind == "down":
        for i in range(3):
            top = P - (i + 1) * rise
            cx = E + TREAD_GAP + TREAD / 2 + i * PITCH
            g.append(f'<geom name="s{i}" type="box" pos="{cx} 0 {top/2}" '
                     f'size="{TREAD/2} 0.8 {top/2}" rgba="0.32 0.36 0.42 1"/>')
        g.append(f'<geom name="run" type="box" pos="{E+3*PITCH+TREAD_GAP+R/2} 0 '
                 f'{(P-3*rise)/2}" size="{R/2} 0.8 {(P-3*rise)/2}" '
                 f'rgba="0.32 0.36 0.42 1"/>')
    elif kind == "hop":
        g.append(f'<geom name="run" type="box" pos="{E+gap+R/2} 0 {P/2}" '
                 f'size="{R/2} 0.8 {P/2}" rgba="0.32 0.36 0.42 1"/>')
    else:
        raise ValueError(kind)
    return "\n    ".join(g)

# The BD-1 head: BIG on purpose, and physically identical to the small
# box it replaced.
#
# A head is not free. The original 17x22x11 cm box weighed 4.25 kg of a
# 22.7 kg robot -- 19% -- because MuJoCo derives mass from geom volume,
# so scaling it up to look like the droid would have multiplied that by
# about six and invalidated 1.5M steps of training for a cosmetic
# change. Two things make the enlargement free:
#
#   density="0" contype="0" conaffinity="0"
#       the shapes carry no mass and collide with nothing.
#   <inertial>
#       the mass, centre of mass and inertia tensor of the ORIGINAL head,
#       measured from the old model and pinned here literally.
#
# So the robot the policy drives is unchanged and only the picture is
# different. `tests/test_terrain_vision.py` asserts that -- identical total
# mass, identical head inertia, and identical depth images, because the
# second risk here is subtler than mass: the camera is mounted IN this
# body, and decorative geometry that drifts into its own field of view
# would corrupt every depth frame the student ever sees while looking
# like nothing more than a nicer render.
HEAD = """
        <inertial pos="0.003131217 0.00005429857 0" mass="4.252544"
                  quat="0.500663 0.499336 -0.500663 0.499336"
                  diaginertia="0.02797353 0.02100553 0.01533584"/>
        <geom type="box" pos="-0.132 0 0.240" euler="0 -9 0"
              size="0.200 0.320 0.250" rgba="0.86 0.86 0.84 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="box" pos="-0.150 0 0.490" euler="0 -9 0"
              size="0.176 0.308 0.026" rgba="0.31 0.34 0.39 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="box" pos="0.030 0 0.365" euler="0 -9 0"
              size="0.038 0.322 0.030" rgba="0.72 0.20 0.17 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="box" pos="-0.300 0 0.240" euler="0 -9 0"
              size="0.046 0.323 0.248" rgba="0.72 0.20 0.17 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="box" pos="-0.132 0 -0.045" euler="0 -9 0"
              size="0.186 0.312 0.022" rgba="0.38 0.40 0.44 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.010 0.232 0.240 0.092 0.232 0.228"
              size="0.140" rgba="0.90 0.90 0.88 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.048 0.232 0.236 0.100 0.232 0.228"
              size="0.098" rgba="0.13 0.26 0.52 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.014 -0.232 0.240 0.094 -0.232 0.229"
              size="0.112" rgba="0.58 0.60 0.63 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.050 -0.232 0.236 0.098 -0.232 0.229"
              size="0.070" rgba="0.30 0.32 0.35 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0.020 0.040 0.155 0.086 0.040 0.146"
              size="0.070" rgba="0.10 0.14 0.30 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="capsule" fromto="-0.300 0.240 0.490 -0.372 0.258 0.730"
              size="0.013" rgba="0.18 0.19 0.21 1"
              density="0" contype="0" conaffinity="0"/>
        <geom type="capsule" fromto="-0.300 -0.240 0.490 -0.372 -0.258 0.730"
              size="0.013" rgba="0.18 0.19 0.21 1"
              density="0" contype="0" conaffinity="0"/>
"""


def world(kind: str, rise=0.09, gap=0.45) -> str:
    return f"""
<mujoco model="t4">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.002"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="3" contype="1" friction="0.9 0.1 0.1"
          rgba="0.42 0.62 0.82 1" solimp="0.95 0.95 0.01" solref="0.01 1"/>
  </default>
  <worldbody>
    <light pos="0 -1.5 5" dir="0 0.3 -1" diffuse="0.9 0.9 0.9" castshadow="true"/>
    <light pos="4 2 4" dir="-0.5 -0.4 -1" diffuse="0.4 0.45 0.55" castshadow="false"/>
    <geom name="void" type="plane" pos="0 0 -3" size="60 4 0.1" rgba="0.08 0.09 0.11 1"/>
    {build(kind, rise, gap)}
    <body name="torso" pos="{START_X} 0 {PLATEAU_Z + 1.0}">
      <joint name="rootx" type="slide" axis="1 0 0" limited="false" armature="0" damping="0"/>
      <joint name="rootz" type="slide" axis="0 0 1" limited="false" armature="0" damping="0"/>
      <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.35" size="0.06"/>
      <body name="head" pos="0 0 0.40">
        {HEAD}
        <camera name="eye" pos="0.10 0 0.0" xyaxes="0 -1 0 0.64 0 0.77"/>
      </body>
      <body name="thighL" pos="0 -0.07 0">
        <joint name="hipL" type="hinge" axis="0 -1 0" range="-150 20"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.045"/>
        <body name="shinL" pos="0 0 -0.40">
          <joint name="kneeL" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.04"/>
          <body name="footL" pos="0 0 -0.36">
            <joint name="ankleL" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="footL_geom" type="capsule" fromto="-0.08 0 0 0.14 0 0" size="0.045"/>
          </body></body></body>
      <body name="thighR" pos="0 0.07 0">
        <joint name="hipR" type="hinge" axis="0 -1 0" range="-150 20"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.045"/>
        <body name="shinR" pos="0 0 -0.40">
          <joint name="kneeR" type="hinge" axis="0 -1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.04"/>
          <body name="footR" pos="0 0 -0.36">
            <joint name="ankleR" type="hinge" axis="0 -1 0" range="-45 45"/>
            <geom name="footR_geom" type="capsule" fromto="-0.08 0 0 0.14 0 0" size="0.045"/>
          </body></body></body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hipL" ctrlrange="-1 1" gear="120" ctrllimited="true"/>
    <motor joint="kneeL" ctrlrange="-1 1" gear="120" ctrllimited="true"/>
    <motor joint="ankleL" ctrlrange="-1 1" gear="70" ctrllimited="true"/>
    <motor joint="hipR" ctrlrange="-1 1" gear="120" ctrllimited="true"/>
    <motor joint="kneeR" ctrlrange="-1 1" gear="120" ctrllimited="true"/>
    <motor joint="ankleR" ctrlrange="-1 1" gear="70" ctrllimited="true"/>
  </actuator>
</mujoco>"""


# HOLLOW stairs: each tread is a thin plate with a VOID between it and the
# next. This is the structure the literature names as the failure case for
# proprioception -- "reactive nature fails when encountering sparse and
# discontinuous structures like hollow stairs, because the absence of
# predictive awareness leads to catastrophic missteps into the gaps"
# (StairMaster, arXiv 2606.25765). Four solid-terrain designs before this
# all came back null: a robot can feel its way up anything continuous.
TREAD = 0.22          # walkable depth of each plate
TREAD_GAP = 0.18      # void between consecutive plates
PITCH = TREAD + TREAD_GAP
FOOT_LEN = 0.22       # the foot can ALMOST bridge a gap -- placement must be right

# The terrains the lab actually trains and measures on.
#
# `hop` -- a gap to leap -- is still BUILDABLE by `build()` and `world()`,
# and a reader can train on it with `--terrain hop`. It is deliberately
# NOT in this tuple, and the reason is a measured result rather than a
# preference: on `hop` the lab's own ablation REVERSES.
#
#     seed       0     1     2
#     blind   1.00  1.00  1.00      <- the arm that cannot see
#     privileged  0.50  1.00  0.00  <- the arm handed the geometry
#
# That is not noise, it is the task. You do not need to SEE a gap to
# clear one: committing to a maximal leap works whether or not you know
# the gap is there, and knowing its width mostly gives the policy a
# second thing to overfit. Three specialists trained on `hop` alone
# scored 0.75 / 0.50 / 0.25, so it is unreliable even unopposed.
#
# Including it would have meant publishing a four-terrain average in
# which one terrain silently argued the opposite of the other three.
# `11_moe` shipped exactly that mistake once -- a finding that reversed
# at world size 2 -- so the terrain is excluded and the negative result
# is written down instead. `tests/test_terrain_vision.py` asserts the
# exclusion, because re-adding it is a one-word edit that would quietly
# invalidate every number in the README.
KINDS = ("flat", "up", "down")


def ground_height(kind: str, x: float, rise: float = 0.09,
                  gap: float = 0.45) -> float:
    """
    Height of the walkable surface at x. One source of truth.

    Termination is judged against THIS rather than an absolute height --
    otherwise descending three steps looks identical to falling, which
    would make `down` unlearnable by construction.
    """
    P, E = PLATEAU_Z, EVENT_X
    if x < E:
        return P
    if kind == "flat":
        return P
    if kind in ("up", "down"):
        sign = 1 if kind == "up" else -1
        span = 3 * PITCH + TREAD_GAP
        if x >= E + span:
            return P + sign * 3 * rise
        for i in range(3):
            lo = E + TREAD_GAP + i * PITCH
            if lo <= x < lo + TREAD:
                return P + sign * (i + 1) * rise
        # Solid stairs: the tread you are over, or the one before it.
        # No voids -- lab 2's geometry, which is what this builds on.
        for i in reversed(range(3)):
            if x >= E + TREAD_GAP + i * PITCH:
                return P + sign * (i + 1) * rise
        return P
    if kind == "hop":
        return -3.0 if x < E + gap else P     # the void, then the far side
    raise ValueError(kind)
